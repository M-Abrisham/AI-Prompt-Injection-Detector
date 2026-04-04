"""Cross-turn mutual information detector (T3.7).

Information-theoretic measure of turn relatedness that catches
distributional anomalies -- a different signal from embedding drift.
Uses character-level entropy and mutual information (pure Python).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List

from na0s.layer16 import config as layer16_config
from na0s.layer16.models import Alert, ConversationState
from na0s.layer16.detectors.base_detector import MultiTurnDetector


# ---------------------------------------------------------------------------
# Information-theoretic primitives
# ---------------------------------------------------------------------------


def _char_distribution(text: str) -> Dict[str, float]:
    """Character frequency distribution (probability mass function)."""
    if not text:
        return {}
    counts = Counter(text.lower())
    total = sum(counts.values())
    return {ch: count / total for ch, count in counts.items()}


def _entropy(dist: Dict[str, float]) -> float:
    """Shannon entropy H(X) = -sum(p * log2(p)).

    Returns 0.0 for empty distributions. Guards against log2(0).
    """
    if not dist:
        return 0.0
    result = -sum(p * math.log2(p) for p in dist.values() if p > 0)
    # Guard against NaN/Inf from floating point edge cases
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result


def _entropy_of_text(text: str) -> float:
    """Convenience: entropy of a text's character distribution."""
    return _entropy(_char_distribution(text))


def _joint_entropy(text1: str, text2: str) -> float:
    """Joint entropy H(X,Y) using character bigram co-occurrence.

    Builds a joint distribution from aligned character pairs of the
    two texts.  If texts differ in length, uses the shorter length.
    Returns 0.0 if either text is empty.
    """
    if not text1 or not text2:
        return 0.0
    t1 = text1.lower()
    t2 = text2.lower()
    min_len = min(len(t1), len(t2))
    if min_len == 0:
        return 0.0

    # Build joint distribution from aligned character pairs
    pair_counts: Counter = Counter()
    for i in range(min_len):
        pair_counts[(t1[i], t2[i])] += 1

    total = sum(pair_counts.values())
    if total == 0:
        return 0.0

    result = -sum(
        (c / total) * math.log2(c / total)
        for c in pair_counts.values()
        if c > 0
    )
    if math.isnan(result) or math.isinf(result):
        return 0.0
    return result


def mutual_information(text1: str, text2: str) -> float:
    """I(X;Y) = H(X) + H(Y) - H(X,Y).

    Returns 0.0 if either text is empty or MI is negative due to
    length mismatch truncation.
    """
    if not text1 or not text2:
        return 0.0
    hx = _entropy_of_text(text1)
    hy = _entropy_of_text(text2)
    hxy = _joint_entropy(text1, text2)
    mi = hx + hy - hxy
    # MI should be non-negative; clamp to handle floating point issues
    if math.isnan(mi) or math.isinf(mi):
        return 0.0
    return max(0.0, mi)


def normalized_mutual_information(text1: str, text2: str) -> float:
    """NMI = 2 * I(X;Y) / (H(X) + H(Y)), range [0, 1].

    Returns 0.0 when both entropies are zero (e.g. empty or
    single-character texts).
    """
    if not text1 or not text2:
        return 0.0
    hx = _entropy_of_text(text1)
    hy = _entropy_of_text(text2)
    denom = hx + hy
    if denom == 0.0:
        return 0.0
    mi = mutual_information(text1, text2)
    nmi = 2.0 * mi / denom
    # Clamp to [0, 1] for safety
    if math.isnan(nmi) or math.isinf(nmi):
        return 0.0
    return max(0.0, min(1.0, nmi))


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class MutualInformationDetector(MultiTurnDetector):
    """Detect distributional anomalies via cross-turn mutual information.

    Two signals:
    1. NMI drop: NMI between consecutive turns drops below threshold,
       indicating an abrupt topic/style shift.
    2. Entropy anomaly: a turn's entropy deviates significantly from
       the conversation mean (very repetitive or obfuscated text).
    """

    @property
    def detector_name(self) -> str:
        return "MutualInformationDetector"

    @property
    def taxonomy_ids(self) -> List[str]:
        return ["T3.7"]

    def reset(self) -> None:
        pass  # Stateless detector -- all state comes from ConversationState

    def analyze(self, state: ConversationState) -> List[Alert]:
        if not layer16_config.ENABLE_MUTUAL_INFORMATION:
            return []

        min_turns = layer16_config.MI_MIN_TURNS
        if state.turn_count < min_turns:
            return []

        window = layer16_config.MI_WINDOW
        nmi_threshold = layer16_config.MI_NMI_DROP_THRESHOLD
        entropy_factor = layer16_config.MI_ENTROPY_DEVIATION_FACTOR

        # Get the relevant window of turns
        turns = state.turns[-window:] if len(state.turns) > window else list(state.turns)

        alerts: List[Alert] = []

        # ------------------------------------------------------------------
        # Signal 1: NMI drop between consecutive turns
        # ------------------------------------------------------------------
        nmi_values: List[float] = []
        for i in range(1, len(turns)):
            nmi = normalized_mutual_information(turns[i - 1].text, turns[i].text)
            nmi_values.append(nmi)

        if nmi_values:
            # Check the most recent NMI value
            latest_nmi = nmi_values[-1]
            if latest_nmi < nmi_threshold and len(nmi_values) >= 2:
                # Compare to previous NMI values to confirm it's a *drop*
                prior_mean = sum(nmi_values[:-1]) / len(nmi_values[:-1])
                if prior_mean > nmi_threshold:
                    confidence = min(1.0, (prior_mean - latest_nmi) / max(prior_mean, 0.01))
                    confidence = max(0.1, confidence)
                    severity = "medium" if confidence < 0.6 else "high"
                    alerts.append(
                        Alert(
                            alert_type="mutual_information_anomaly",
                            severity=severity,
                            confidence=confidence,
                            description=(
                                f"NMI dropped from {prior_mean:.3f} to "
                                f"{latest_nmi:.3f} (threshold: {nmi_threshold}), "
                                f"indicating abrupt distributional shift"
                            ),
                            turn_range=(state.turn_count - 2, state.turn_count - 1),
                            evidence=[
                                f"nmi_latest={latest_nmi:.4f}",
                                f"nmi_prior_mean={prior_mean:.4f}",
                            ],
                        )
                    )

        # ------------------------------------------------------------------
        # Signal 2: Entropy anomaly on the latest turn
        # ------------------------------------------------------------------
        entropies = [_entropy_of_text(t.text) for t in turns]
        if len(entropies) >= 2:
            latest_entropy = entropies[-1]
            # Mean of all turns except the latest
            prior_entropies = entropies[:-1]
            mean_entropy = sum(prior_entropies) / len(prior_entropies)

            if mean_entropy > 0:
                # Check for high entropy (encoded/obfuscated) or very low (repetitive)
                ratio = latest_entropy / mean_entropy if mean_entropy > 0 else 0.0
                is_anomaly = False
                anomaly_desc = ""

                if ratio >= entropy_factor:
                    is_anomaly = True
                    anomaly_desc = (
                        f"Turn entropy ({latest_entropy:.3f}) is {ratio:.1f}x "
                        f"the conversation mean ({mean_entropy:.3f}) -- "
                        f"possible encoded/obfuscated content"
                    )
                elif mean_entropy > 0 and ratio <= (1.0 / entropy_factor):
                    is_anomaly = True
                    anomaly_desc = (
                        f"Turn entropy ({latest_entropy:.3f}) is {ratio:.2f}x "
                        f"the conversation mean ({mean_entropy:.3f}) -- "
                        f"highly repetitive content"
                    )

                if is_anomaly:
                    confidence = min(1.0, abs(ratio - 1.0) / entropy_factor)
                    confidence = max(0.1, confidence)
                    severity = "medium" if confidence < 0.6 else "high"
                    alerts.append(
                        Alert(
                            alert_type="mutual_information_anomaly",
                            severity=severity,
                            confidence=confidence,
                            description=anomaly_desc,
                            turn_range=(state.turn_count - 1, state.turn_count - 1),
                            evidence=[
                                f"turn_entropy={latest_entropy:.4f}",
                                f"mean_entropy={mean_entropy:.4f}",
                                f"ratio={ratio:.4f}",
                            ],
                        )
                    )

        return alerts
