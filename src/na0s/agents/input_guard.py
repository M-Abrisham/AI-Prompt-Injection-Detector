"""Dogfood Na0S's own injection detector on untrusted agent inputs.

The deploy-approval agents sit on the "lethal trifecta": they ingest adversarial
text (canary ``errors``, quarantine rows, synthetic samples), feed it to the
Claude API, and gate a deploy on an iMessage reply. The text-fencing in
``claude_gate_analyzer`` *contains* that text; this module *detects* it by
running Na0S's own calibrated ``predict()`` over the untrusted strings before
they ever reach the model prompt.

Defense-in-depth, not a replacement for the ``<UNTRUSTED_CI_DATA>`` fence:
fencing keeps adversarial text from being followed as an instruction; this
guard surfaces *that it is* adversarial so the human approver sees it.

The guard is deliberately **fail-safe**: if the ML pipeline is unavailable
(models missing, import error, …) it returns an unflagged, ``"unscanned"``
result and logs a warning. The daemon must keep running even when the detector
can't load — an unavailable scanner must never crash the deploy-approval loop.
"""

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class GuardResult:
    """Outcome of scanning one untrusted string with Na0S's own detector.

    Attributes:
        text: The original (unmodified) input string.
        flagged: True iff Na0S's detector returned a ``MALICIOUS`` verdict.
            This is the detector's *own* calibrated decision — there is no
            second numeric threshold layered on top of it here.
        risk_score: Probability-of-malicious in ``[0, 1]``. For a ``MALICIOUS``
            verdict this is the model's confidence; for ``SAFE`` it is
            ``1 - confidence``. ``0.0`` when unscanned/empty.
        label: The detector's verdict (``"MALICIOUS"``/``"SAFE"``/``"BLOCKED"``),
            or ``"unscanned"`` when the detector could not run, or ``"empty"``
            for empty/whitespace input.
        technique_ids: Layer-0 / rule technique identifiers, when available.
        safe_text: The text to forward downstream. When ``flagged``, this is the
            original text prefixed with a visible warning annotation so the
            human approver (and the model, behind the fence) sees the verdict.
            Otherwise it is the original text unchanged.
    """

    text: str
    flagged: bool
    risk_score: float
    label: str
    technique_ids: List[str] = field(default_factory=list)
    safe_text: str = ""


def scan_untrusted(text, *, source: str = "input") -> GuardResult:
    """Scan one untrusted string with Na0S's own ``predict()`` detector.

    The ML pipeline is imported **lazily inside this function** so that merely
    importing an agent module does not drag in the heavy detector at process
    start — it is loaded on first scan and cached thereafter by ``predict``.

    The ``flagged`` decision comes solely from ``predict``'s own verdict
    (``label == "MALICIOUS"``); no additional numeric threshold is invented
    here — ``predict`` already applies calibrated thresholds internally.

    Args:
        text: The untrusted string to scan (e.g. a canary error preview).
        source: A short label identifying where the text came from, used only
            in the warning annotation (e.g. ``"canary_error"``).

    Returns:
        A :class:`GuardResult`. Never raises — on any failure to load or run
        the detector it returns a fail-safe unflagged ``"unscanned"`` result.
    """
    raw = "" if text is None else str(text)

    # Empty / whitespace-only input is an unflagged passthrough — there is
    # nothing to scan and nothing to annotate.
    if not raw.strip():
        return GuardResult(
            text=raw,
            flagged=False,
            risk_score=0.0,
            label="empty",
            technique_ids=[],
            safe_text=raw,
        )

    try:
        # Lazy import: keep the ML pipeline out of agent import time.
        from na0s.predict import predict, _get_cached_models

        vectorizer, model = _get_cached_models()
        label, prob, l0 = predict(raw, vectorizer, model)

        # ``prob`` is the model's confidence in its OWN predicted class, so a
        # SAFE verdict also carries a high ``prob``. Normalize to a
        # probability-of-malicious so risk_score is monotone in maliciousness.
        confidence = float(prob)
        if label == "MALICIOUS":
            risk_score = confidence
        elif label == "SAFE":
            risk_score = 1.0 - confidence
        else:  # BLOCKED (layer-0 reject) — treat as max risk.
            risk_score = 1.0

        # Pull any technique/rule identifiers Layer-0 surfaced, best-effort.
        technique_ids: List[str] = []
        for attr in ("technique_ids", "techniques", "flags", "reasons"):
            value = getattr(l0, attr, None)
            if value:
                technique_ids = [str(v) for v in value]
                break

        flagged = label == "MALICIOUS" or label == "BLOCKED"
        if flagged:
            safe_text = (
                f"[⚠️ na0s flagged as likely injection "
                f"(score={risk_score:.2f}): {source}] {raw}"
            )
        else:
            safe_text = raw

        return GuardResult(
            text=raw,
            flagged=flagged,
            risk_score=risk_score,
            label=label,
            technique_ids=technique_ids,
            safe_text=safe_text,
        )

    except Exception as e:
        # Fail-safe: the detector being unavailable must NOT take down the
        # agent. Forward the text unchanged and let the existing fence contain
        # it; record that it went un-scanned so the gap is visible in logs.
        logger.warning(
            "na0s input_guard could not scan untrusted %s input (%s: %s); "
            "passing through unscanned",
            source,
            type(e).__name__,
            e,
        )
        return GuardResult(
            text=raw,
            flagged=False,
            risk_score=0.0,
            label="unscanned",
            technique_ids=[],
            safe_text=raw,
        )
