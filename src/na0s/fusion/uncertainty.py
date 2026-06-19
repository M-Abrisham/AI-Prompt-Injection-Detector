"""GAP-12: low-margin / signal-disagreement abstain band.

A single hard threshold makes a 0.5499-vs-0.5500 input a coin-flip, and it hides
cases where the detectors DISAGREE (ML says safe while rules say malicious)
behind one fused number.  ``assess_uncertainty`` marks such borderline verdicts
as ``abstained`` with an ``uncertainty`` score so the embedding application can
escalate (LLM judge / human review) instead of trusting the flip.

It deliberately does NOT change the verdict: the abstain DEFAULT (block-on-
uncertainty vs keep-verdict) and the band WIDTH are TPR/FPR tradeoffs that must
be tuned on the eval harness — they are surfaced here as env-overridable config,
not guessed constants baked into the verdict.
"""

from __future__ import annotations

import os
import statistics

#: Base half-width of the low-margin band around the decision threshold.  Within
#: this distance the verdict is a near-coin-flip -> abstain.  EVAL-TUNABLE.
ABSTAIN_BAND = float(os.environ.get("NA0S_ABSTAIN_BAND", "0.05"))

#: How much MAXIMAL signal disagreement extends the abstain band.  Disagreement
#: only WIDENS the band near the threshold — it never makes a confident,
#: far-from-threshold verdict abstain (that would be wrong: the fused score is
#: confident even if one weak signal dissents).  EVAL-TUNABLE.
DISAGREEMENT_WIDEN = float(os.environ.get("NA0S_DISAGREEMENT_WIDEN", "0.10"))

#: Disagreement (population stdev of P(malicious) signals) is normalized against
#: this "fully split" reference (0.5 = a perfect even split, e.g. {0,1}).
_MAX_DISAGREEMENT = 0.5


def assess_uncertainty(composite, threshold, signal_probs=()):
    """Return ``(abstained: bool, uncertainty: float)`` for a verdict.

    Parameters
    ----------
    composite : float
        Final risk score in [0, 1].
    threshold : float
        Decision threshold the verdict was taken at.
    signal_probs : iterable of float or None
        Per-signal P(malicious) estimates in [0, 1].  ``None`` entries (absent
        signals — e.g. the embedding model not loaded, so its 0.0 is "no info"
        not "agrees safe") are IGNORED, so absence never fakes agreement.

    A verdict abstains when its distance from the threshold is within the
    *effective* band = ``ABSTAIN_BAND`` widened by signal disagreement.  A
    confident verdict (large margin) never abstains, however much the signals
    split — only borderline verdicts do, and disagreement widens how borderline
    counts.  ``uncertainty`` is how deep inside the effective band the verdict is.
    """
    margin = abs(float(composite) - float(threshold))

    present = [float(p) for p in signal_probs if p is not None]
    disagreement = statistics.pstdev(present) if len(present) >= 2 else 0.0
    widen = DISAGREEMENT_WIDEN * min(disagreement / _MAX_DISAGREEMENT, 1.0)
    effective_band = ABSTAIN_BAND + widen

    abstained = bool(effective_band > 0 and margin < effective_band)
    if abstained:
        uncertainty = round(max(0.0, 1.0 - margin / effective_band), 4)
    else:
        uncertainty = 0.0
    return abstained, uncertainty
