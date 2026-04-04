"""Layer 16 Bayesian Online Change Point Detection (BOCPD).

Detects abrupt shifts in conversation risk profiles using the
Adams & MacKay (2007) algorithm.  Pure-Python implementation with
no external dependencies.

Taxonomy IDs: D1.24 (Conversation Change Point)
"""

from __future__ import annotations

import math
from typing import Dict, List

from na0s.layer16 import config as layer16_config
from na0s.layer16.detectors.base_detector import MultiTurnDetector
from na0s.layer16.models import Alert, ConversationState

# Safety bound: maximum run length vector size to prevent unbounded growth.
_MAX_RUN_LENGTH = 500

# Minimum float to avoid log(0) and division by zero.
_EPS = 1e-300


class BOCPD:
    """Bayesian Online Change Point Detection.

    Computes the posterior probability that a change point occurred
    at each observation.  Uses a Normal-Gamma conjugate prior for
    tractable online updates of a Gaussian observation model.

    Reference: Adams & MacKay (2007) "Bayesian Online Changepoint Detection"

    Args:
        hazard_rate: 1/expected_run_length.  E.g. 0.02 means expect a
            change every ~50 observations.
        mu_prior: Prior mean for observations.
        kappa_prior: Prior strength (pseudo-observations for the mean).
        alpha_prior: Prior shape for precision (>0).
        beta_prior: Prior rate for precision (>0).
    """

    def __init__(
        self,
        hazard_rate: float = 0.02,
        mu_prior: float = 0.1,
        kappa_prior: float = 1.0,
        alpha_prior: float = 1.0,
        beta_prior: float = 0.1,
    ) -> None:
        if hazard_rate <= 0.0 or hazard_rate > 1.0:
            raise ValueError("hazard_rate must be in (0.0, 1.0]")
        if kappa_prior <= 0.0:
            raise ValueError("kappa_prior must be > 0")
        if alpha_prior <= 0.0:
            raise ValueError("alpha_prior must be > 0")
        if beta_prior <= 0.0:
            raise ValueError("beta_prior must be > 0")

        self._hazard = hazard_rate
        self._mu0 = mu_prior
        self._kappa0 = kappa_prior
        self._alpha0 = alpha_prior
        self._beta0 = beta_prior

        # Run length distribution: R[i] = P(r_t = i | x_{1:t})
        self._run_length_probs: List[float] = [1.0]

        # Sufficient statistics for each run length hypothesis
        self._mu: List[float] = [mu_prior]
        self._kappa: List[float] = [kappa_prior]
        self._alpha: List[float] = [alpha_prior]
        self._beta: List[float] = [beta_prior]

        self._last_cp_prob: float = 0.0
        self._observation_count: int = 0
        # Track MAP run length for change point detection
        self._prev_map_rl: int = 0
        # Short run length threshold for change detection
        self._short_rl_k: int = 3

    def update(self, x: float) -> float:
        """Process one observation, return P(change point at this step).

        Returns the probability mass allocated to run length 0 after
        normalization, which represents the posterior probability that
        a change point occurred at this time step.
        """
        if not isinstance(x, (int, float)):
            raise TypeError("observation must be a number")
        if math.isnan(x) or math.isinf(x):
            raise ValueError("observation must be finite")

        self._observation_count += 1
        n = len(self._run_length_probs)

        # 1. Evaluate predictive probability under each run length
        pred_probs = []
        for i in range(n):
            nu = 2.0 * self._alpha[i]
            if nu <= 0:
                pred_probs.append(_EPS)
                continue
            sigma2 = (self._beta[i] * (self._kappa[i] + 1.0)) / (
                self._alpha[i] * self._kappa[i]
            )
            if sigma2 <= 0:
                sigma2 = _EPS
            pred_probs.append(
                self._student_t_pdf(x, self._mu[i], sigma2, nu)
            )

        # 2. Growth probabilities (existing run continues)
        growth_probs = [
            self._run_length_probs[i] * pred_probs[i] * (1.0 - self._hazard)
            for i in range(n)
        ]

        # 3. Change point probability (all runs collapse to r=0)
        cp_prob = sum(
            self._run_length_probs[i] * pred_probs[i] * self._hazard
            for i in range(n)
        )

        # 4. New run length distribution: [cp_prob, growth_probs...]
        new_rl = [cp_prob] + growth_probs

        # 5. Normalize
        total = sum(new_rl)
        if total > 0:
            new_rl = [p / total for p in new_rl]
        else:
            # Fallback: uniform restart
            new_rl = [1.0 / len(new_rl)] * len(new_rl)

        # 6. Update sufficient statistics for each run length
        new_mu = [self._mu0]
        new_kappa = [self._kappa0]
        new_alpha = [self._alpha0]
        new_beta = [self._beta0]

        for i in range(n):
            old_mu = self._mu[i]
            old_kappa = self._kappa[i]
            old_alpha = self._alpha[i]
            old_beta = self._beta[i]

            new_k = old_kappa + 1.0
            new_m = (old_kappa * old_mu + x) / new_k
            new_a = old_alpha + 0.5
            new_b = old_beta + (old_kappa * (x - old_mu) ** 2) / (
                2.0 * new_k
            )

            new_mu.append(new_m)
            new_kappa.append(new_k)
            new_alpha.append(new_a)
            new_beta.append(new_b)

        # 7. Truncate to prevent unbounded growth
        if len(new_rl) > _MAX_RUN_LENGTH:
            # Keep the most probable run lengths; merge tail into last entry
            new_rl = new_rl[:_MAX_RUN_LENGTH]
            new_mu = new_mu[:_MAX_RUN_LENGTH]
            new_kappa = new_kappa[:_MAX_RUN_LENGTH]
            new_alpha = new_alpha[:_MAX_RUN_LENGTH]
            new_beta = new_beta[:_MAX_RUN_LENGTH]
            # Re-normalize after truncation
            total = sum(new_rl)
            if total > 0:
                new_rl = [p / total for p in new_rl]

        self._run_length_probs = new_rl
        self._mu = new_mu
        self._kappa = new_kappa
        self._alpha = new_alpha
        self._beta = new_beta

        # Change point signal: mass on short run lengths (r < k).
        # After a change point, the posterior mass shifts to short runs
        # because the fresh prior explains the new data better than
        # long-running hypotheses.  We subtract the baseline expected
        # mass at short runs under no-change to get a cleaner signal.
        short_mass = sum(new_rl[:self._short_rl_k]) if len(new_rl) >= self._short_rl_k else 0.0
        # Baseline: in steady state, P(r<k) ~ 1 - (1-H)^k for geometric dist
        baseline = 1.0 - (1.0 - self._hazard) ** self._short_rl_k
        # CP probability = excess short-run mass, clamped to [0, 1]
        self._last_cp_prob = max(0.0, min(1.0, (short_mass - baseline) / max(1.0 - baseline, _EPS)))

        return self._last_cp_prob

    def _student_t_pdf(
        self, x: float, mu: float, sigma2: float, nu: float
    ) -> float:
        """Student-t predictive probability density.

        Uses log-gamma for numerical stability.
        """
        if nu <= 0 or sigma2 <= 0:
            return _EPS

        try:
            # log of the normalization constant
            log_norm = (
                math.lgamma((nu + 1.0) / 2.0)
                - math.lgamma(nu / 2.0)
                - 0.5 * math.log(nu * math.pi * sigma2)
            )
            # log of the kernel
            z = (x - mu) ** 2 / (nu * sigma2)
            log_kernel = -((nu + 1.0) / 2.0) * math.log(1.0 + z)

            result = math.exp(log_norm + log_kernel)
        except (ValueError, OverflowError):
            return _EPS

        # Guard against NaN/Inf
        if math.isnan(result) or math.isinf(result):
            return _EPS
        return max(result, _EPS)

    @property
    def change_point_probability(self) -> float:
        """Most recent change point probability."""
        return self._last_cp_prob

    @property
    def observation_count(self) -> int:
        """Number of observations processed."""
        return self._observation_count

    def reset(self) -> None:
        """Reset to prior state."""
        self._run_length_probs = [1.0]
        self._mu = [self._mu0]
        self._kappa = [self._kappa0]
        self._alpha = [self._alpha0]
        self._beta = [self._beta0]
        self._last_cp_prob = 0.0
        self._observation_count = 0
        self._prev_map_rl = 0


class ChangePointDetector(MultiTurnDetector):
    """Detects conversation change points using BOCPD.

    Feeds risk_score from each turn into the BOCPD algorithm.
    When the change point probability exceeds a threshold,
    it signals that the conversation has shifted (e.g. from benign
    to adversarial).

    Taxonomy IDs: D1.24 (Conversation Change Point)
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, BOCPD] = {}
        # Track how many turns we've fed per session to avoid re-feeding
        self._session_turn_counts: Dict[str, int] = {}

    def analyze(self, state: ConversationState) -> List[Alert]:
        """Run BOCPD on conversation risk scores."""
        if not layer16_config.ENABLE_CHANGE_POINT:
            return []

        session_id = state.session_id
        turns = state.turns

        if len(turns) < layer16_config.BOCPD_MIN_TURNS:
            return []

        # Get or create BOCPD instance
        if session_id not in self._sessions:
            self._sessions[session_id] = BOCPD(
                hazard_rate=layer16_config.BOCPD_HAZARD_RATE,
            )
            self._session_turn_counts[session_id] = 0

        bocpd = self._sessions[session_id]
        fed_count = self._session_turn_counts[session_id]

        # Feed only new (unseen) turns, tracking max CP prob
        max_cp_prob = 0.0
        max_cp_turn_idx = fed_count
        for idx, turn in enumerate(turns[fed_count:], start=fed_count):
            cp_prob = bocpd.update(turn.risk_score)
            # Skip warmup period (first 2 observations produce spurious spikes)
            if bocpd.observation_count <= 2:
                continue
            if cp_prob > max_cp_prob:
                max_cp_prob = cp_prob
                max_cp_turn_idx = idx
        self._session_turn_counts[session_id] = len(turns)

        # Check if change point detected
        alerts: List[Alert] = []
        if max_cp_prob > layer16_config.BOCPD_CHANGE_POINT_THRESHOLD:
            severity = "medium" if max_cp_prob < 0.8 else "high"
            alerts.append(
                Alert(
                    alert_type="change_point",
                    severity=severity,
                    confidence=min(max_cp_prob, 1.0),
                    description=(
                        f"Conversation change point detected at turn {max_cp_turn_idx + 1} "
                        f"(P={max_cp_prob:.3f}). Risk profile has shifted significantly."
                    ),
                    turn_range=(max(0, max_cp_turn_idx - 1), max_cp_turn_idx),
                    evidence=[
                        f"BOCPD change point probability: {max_cp_prob:.4f}",
                        f"Threshold: {layer16_config.BOCPD_CHANGE_POINT_THRESHOLD}",
                    ],
                )
            )

        # Cleanup: prune sessions not seen recently
        # (Lazy cleanup: only when session count exceeds 2x a reasonable limit)
        if len(self._sessions) > 20000:
            self._cleanup_stale_sessions(state)

        return alerts

    def _cleanup_stale_sessions(self, current_state: ConversationState) -> None:
        """Remove sessions that are no longer active."""
        # Keep only the current session during cleanup (conservative)
        current_id = current_state.session_id
        stale = [
            sid for sid in self._sessions if sid != current_id
        ]
        # Remove oldest half
        for sid in stale[: len(stale) // 2]:
            self._sessions.pop(sid, None)
            self._session_turn_counts.pop(sid, None)

    def remove_session(self, session_id: str) -> None:
        """Explicitly remove a session's BOCPD state."""
        self._sessions.pop(session_id, None)
        self._session_turn_counts.pop(session_id, None)

    def reset(self) -> None:
        """Clear all internal state."""
        self._sessions.clear()
        self._session_turn_counts.clear()

    @property
    def detector_name(self) -> str:
        return "ChangePointDetector"

    @property
    def taxonomy_ids(self) -> List[str]:
        return ["D1.24"]
