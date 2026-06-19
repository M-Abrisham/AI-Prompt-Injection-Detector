"""Global test isolation for Na0S.

Two detection components keep *process-global* mutable state that the pipeline
deliberately persists across `scan()` calls in production:

  * the known-malicious ``FingerprintStore`` (``na0s.input.tokenization``):
    every MALICIOUS verdict registers its sanitized text, and a later scan that
    matches a stored fingerprint is floored to MALICIOUS by ``predict.py``.
  * the ``ConversationSecurityMonitor`` (``na0s.predict``): accumulates
    per-session turns, risk scores, and alerts.

Both are singletons that live for the whole pytest process. Without a reset
between tests, fingerprints and session risk registered by one test file leak
into later files, so a near-threshold scan can be flipped or mis-attributed
purely by *test order*. That is invisible while the embedding signal dominates
but surfaces the moment embeddings are degraded/offline (``NA0S_EMBEDDING_ENABLED=0``)
and scans sit near the decision threshold — the exact order-dependent flakiness
that made the embedding-off CI runs non-deterministic.

This autouse fixture resets both singletons before every test, making the whole
suite order-independent with respect to this state. (``test_false_positives.py``
already did this locally; this hoists the same guarantee to the entire suite.)
The model/vectorizer caches are intentionally NOT reset — they are immutable
artifacts, not accumulating state, and reloading them per test would be wasteful
with no correctness effect.
"""

import pytest


@pytest.fixture(autouse=True)
def _reset_pipeline_global_state():
    """Clear cross-test detection state before each test (see module docstring)."""
    # Fingerprint store: drop the singleton so the next access rebuilds it empty.
    try:
        import na0s.input.tokenization as _tok

        _tok._default_store = None
    except Exception:
        pass

    # Conversation monitor: cleanup + drop the singleton.
    try:
        import na0s.predict as _predict

        _predict._reset_conversation_monitor()
    except Exception:
        pass

    yield
