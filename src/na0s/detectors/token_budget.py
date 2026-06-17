"""D8.1 Token-budget / context-length / system-prompt-eviction accounting.

Closes gap D8-G03: every other Na0S signal measures input size by WORD or
TURN count, never by TOKENS.  An attacker who pads the input toward the
model's context window can push the customer's system prompt / safety
preamble out of the live window (context-flooding -> system-prompt
eviction), neutralising the very guardrails Na0S protects.

This module measures the input against the CUSTOMER's model window in
tokens and raises a D8.1 signal as the input approaches the eviction
threshold.

Na0S is a DEFENSIVE SDK embedded in the customer's application, so the
"model window" here is the customer's deployed model -- it is configured,
not detected.  Counting is exact when ``tiktoken`` is installed and falls
back to a conservative heuristic otherwise (the SDK must never hard-require
tiktoken; it is an optional dependency, see ``layer0/tokenization.py``).

Pure, fast, side-effect-free: no network and no file I/O at call time.

Integration: intended to be called from the cascade for every input so the
boost can be folded into the aggregate score (see final report -- wiring is
done by the orchestrator, not this module).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# tiktoken is OPTIONAL.  When absent we degrade to a heuristic rather than
# crash -- identical policy to src/na0s/layer0/tokenization.py:11-19.
try:  # pragma: no cover - import guard is trivial
    import tiktoken

    _HAS_TIKTOKEN = True
except ImportError:  # pragma: no cover - exercised via monkeypatch in tests
    tiktoken = None  # type: ignore[assignment]
    _HAS_TIKTOKEN = False


# ---------------------------------------------------------------------------
# Named constants -- every threshold carries its derivation.
# ---------------------------------------------------------------------------

# Default context window (tokens) when the customer has not configured one.
# 8192 is the smallest window among current mainstream production models, so
# assuming it is CONSERVATIVE: if the real deployed window is larger we may
# warn slightly early (safe), but we never silently under-count and miss a
# real eviction.  Overridable via NA0S_MODEL_CONTEXT_WINDOW for the customer's
# actual model.
_DEFAULT_MODEL_WINDOW = 8192

# Env var the customer sets to their deployed model's real window.
_WINDOW_ENV_VAR = "NA0S_MODEL_CONTEXT_WINDOW"

# Fraction of the window reserved for the system prompt + safety preamble +
# room for the model's own response.  The guardrail budget is therefore half
# the window: anything past this point is eating into headroom that the
# defender needs.  0.5 is a deliberately generous reservation -- real system
# prompts + responses routinely consume 30-60% of a small window, so half is
# the round, defensible midpoint.  Overridable per call via guardrail_budget.
_DEFAULT_GUARDRAIL_FRACTION = 0.5

# Eviction risk threshold: at >=90% of the window only ~10% of tokens remain
# for everything else, so a system prompt of any realistic size is at genuine
# risk of being truncated/evicted from the live window.  This is the D8.1
# DETECT line.
_EVICTION_RATIO = 0.90

# Heuristic chars-per-token when tiktoken is unavailable.  ~4 chars/token is
# the well-known English rule of thumb for BPE tokenizers (and what
# judge/llm_judge.py and layer0 assume); we additionally floor the estimate at
# the word count so whitespace-heavy / CJK text is never under-counted.
_CHARS_PER_TOKEN = 4

# Boost ceiling for the DETECT (eviction) case.  Capped at 0.25 to match the
# sibling D8 detector (context_manipulation.py:274 caps padding/hijack boosts
# at 0.25); a single structural signal should nudge, not unilaterally decide.
_MAX_DETECT_BOOST = 0.25

# Per-point-over-budget boost slope for the DETECT case.  At exactly the
# eviction ratio boost starts near zero and climbs to the cap by ~2x window;
# 2.5 means full cap is reached at ratio ~= _EVICTION_RATIO + 0.1 (i.e. at the
# window itself), so being AT the window already earns the cap.
_DETECT_BOOST_SLOPE = 2.5

# Boost for the weaker WATCH case (over guardrail budget but below eviction).
# Intentionally small -- this is "you are spending your headroom", not an
# attack on its own, and must not drive a benign long document to a block.
_WATCH_BOOST = 0.05


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TokenBudgetResult:
    """Result of token-budget / context-eviction analysis.

    Attributes
    ----------
    detected:
        True only for the D8.1 eviction-risk case (ratio >= _EVICTION_RATIO).
        The weaker WATCH signal leaves this False but populates ``reason``.
    token_count:
        Number of tokens in ``text`` (exact via tiktoken, else estimated).
    estimated:
        True when tiktoken was unavailable and the heuristic was used.
    model_window:
        The window (tokens) the input was measured against.
    ratio:
        token_count / model_window.
    boost:
        Score boost to fold into the aggregate (0.0 when nothing fires).
    technique_ids:
        Taxonomy ids; ["D8.1"] on detection, [] otherwise.
    reason:
        Human-readable explanation -- always populated.
    """

    detected: bool = False
    token_count: int = 0
    estimated: bool = False
    model_window: int = _DEFAULT_MODEL_WINDOW
    ratio: float = 0.0
    boost: float = 0.0
    technique_ids: list = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

# Module-level encoder cache.  We do NOT load it at import time: tiktoken's
# get_encoding() can hit the network on a cold cache, and import-time network
# I/O breaks offline/air-gapped installs (same reasoning as
# layer0/tokenization.py:30-33).  Loaded lazily on first count instead.
_ENCODER = None
_ENCODER_TRIED = False


def _get_encoder():
    """Lazily return a cl100k_base encoder, or None if unavailable.

    Any failure (not installed, cold-cache network error, etc.) returns None
    so the caller transparently falls back to the heuristic.  The attempt is
    memoised so we never repeat a failing/slow load.
    """
    global _ENCODER, _ENCODER_TRIED
    if not _HAS_TIKTOKEN:
        return None
    if _ENCODER is not None or _ENCODER_TRIED:
        return _ENCODER
    _ENCODER_TRIED = True
    try:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:
        # Installed but unusable (e.g. offline cold cache) -> heuristic.
        _ENCODER = None
    return _ENCODER


def _count_tokens(text: str) -> tuple[int, bool]:
    """Return (token_count, estimated).

    Uses tiktoken when available; otherwise the chars/4 heuristic floored at
    the word count.  ``estimated`` is True for the heuristic path.
    """
    enc = _get_encoder()
    if enc is not None:
        try:
            return len(enc.encode(text)), False
        except Exception:
            # Defensive: fall through to heuristic on any encode failure.
            pass
    word_count = len(text.split())
    heuristic = max(len(text) // _CHARS_PER_TOKEN, word_count)
    return heuristic, True


def _resolve_window(model_window: int | None) -> int:
    """Resolve the effective window: explicit arg > env var > default."""
    if model_window is not None:
        return model_window
    env = os.getenv(_WINDOW_ENV_VAR)
    if env:
        try:
            parsed = int(env)
            if parsed > 0:
                return parsed
        except (TypeError, ValueError):
            pass  # malformed env -> fall back to the conservative default
    return _DEFAULT_MODEL_WINDOW


# ---------------------------------------------------------------------------
# Main detection function
# ---------------------------------------------------------------------------

def analyze_token_budget(
    text: str,
    *,
    model_window: int | None = None,
    guardrail_budget: int | None = None,
) -> TokenBudgetResult:
    """Analyse ``text`` against the customer's token budget (D8.1).

    Parameters
    ----------
    text:
        The input to measure.
    model_window:
        The customer's model context window in tokens.  Defaults to the
        ``NA0S_MODEL_CONTEXT_WINDOW`` env var, then ``_DEFAULT_MODEL_WINDOW``.
    guardrail_budget:
        Token budget reserved for system prompt + safety preamble + response.
        Defaults to ``_DEFAULT_GUARDRAIL_FRACTION`` of the resolved window.

    Returns
    -------
    TokenBudgetResult
        ``detected`` is True only for the eviction-risk case.  A weaker
        over-budget WATCH signal returns detected=False with a populated
        ``reason`` and a small ``boost``.
    """
    window = _resolve_window(model_window)

    if guardrail_budget is None:
        guardrail_budget = int(window * _DEFAULT_GUARDRAIL_FRACTION)

    token_count, estimated = _count_tokens(text)
    # window is guaranteed > 0 by _resolve_window / default constant.
    ratio = token_count / window

    # --- DETECT: approaching / past system-prompt eviction (D8.1) ---
    if ratio >= _EVICTION_RATIO:
        # Scale by how far past the eviction line we are, capped.  At the line
        # the boost is near zero; it reaches the cap by the window itself.
        over = ratio - _EVICTION_RATIO
        boost = min(_MAX_DETECT_BOOST, _DETECT_BOOST_SLOPE * over * _MAX_DETECT_BOOST)
        # At/over the window itself, always award the full cap.
        if ratio >= 1.0:
            boost = _MAX_DETECT_BOOST
        return TokenBudgetResult(
            detected=True,
            token_count=token_count,
            estimated=estimated,
            model_window=window,
            ratio=ratio,
            boost=boost,
            technique_ids=["D8.1"],
            reason=(
                f"Input is {token_count} tokens "
                f"({ratio:.0%} of the {window}-token window); within "
                f"{(1 - _EVICTION_RATIO):.0%} of the limit -- system prompt "
                f"at risk of eviction (D8.1)."
            ),
        )

    # --- WATCH: over the guardrail budget but below eviction ---
    if token_count > guardrail_budget:
        return TokenBudgetResult(
            detected=False,
            token_count=token_count,
            estimated=estimated,
            model_window=window,
            ratio=ratio,
            boost=_WATCH_BOOST,
            technique_ids=[],
            reason=(
                f"Input is {token_count} tokens, over the "
                f"{guardrail_budget}-token guardrail budget but below the "
                f"eviction threshold -- watch only."
            ),
        )

    # --- Clean: comfortably within budget ---
    return TokenBudgetResult(
        detected=False,
        token_count=token_count,
        estimated=estimated,
        model_window=window,
        ratio=ratio,
        boost=0.0,
        technique_ids=[],
        reason=(
            f"Input is {token_count} tokens, within the "
            f"{guardrail_budget}-token guardrail budget."
        ),
    )
