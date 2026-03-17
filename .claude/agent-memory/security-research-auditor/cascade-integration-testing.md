# Cascade Integration Testing (2026-02-17)

## Test File
`tests/test_cascade_integration.py` -- 77 tests, ALL PASSING (0 failures, 0 expected failures)

## Test Class Breakdown
- **TestWhitelistFilter**: 15 tests -- Stage 1 whitelist fast-track (no model needed)
- **TestWeightedClassifier**: 11 tests -- Stage 2 weighted ML+rules+obfuscation (needs models)
- **TestCascadeClassifier**: 16 tests -- Full cascade end-to-end + judge routing + stats
- **TestCascadeCanary**: 9 tests -- Layer 10 canary token injection/detection (standalone)
- **TestCascadeScanOutput**: 12 tests -- Layer 9 output scanning (standalone)
- **TestCascadeEdgeCases**: 14 tests -- Edge cases, unicode, error handling

## Key Architecture Findings

### WhitelistFilter (Stage 1)
- ALL 6 criteria must pass for whitelist bypass (AND logic)
- Question word OR trailing '?' satisfies criterion 1
- Obfuscation heuristics (base64/hex/URL) fire BEFORE length check
  - Repeated ASCII chars (e.g., "x" * 500) match base64 regex
  - Use varied text content when testing length limits
- SAFE_TOPIC_INDICATORS append to reason string (not a criterion)
- Empty/whitespace inputs fail on criterion 1 (no question pattern)

### WeightedClassifier (Stage 2)
- ML_WEIGHT=0.6 dominates composite score
- Override protection: ml_safe_confidence>0.8 + medium-only rules + no obfuscation -> trust ML
- SEVERITY_WEIGHTS: critical=0.3, high=0.2, medium=0.1
- Obfuscation cap: 0.3 max (0.15 per flag)
- DEFAULT_THRESHOLD: 0.55

### CascadeClassifier
- 4-tuple return: (label, confidence, hits, stage)
- stage values: 'whitelist', 'weighted', 'embedding', 'judge', 'positive_validation', 'blocked'
- Layer 0 runs FIRST (before whitelist); can produce 'blocked' stage
- Judge routing: JUDGE_LOWER_THRESHOLD=0.25, JUDGE_UPPER_THRESHOLD=0.85
- Judge exception handling: graceful fallback to weighted (non-fatal)
- Stats tracking: 10 counters, all initialized to 0, reset_stats() zeros all

### Canary Tokens (Layer 10)
- CanaryManager is standalone (no model dependency)
- Token format: PREFIX-hexchars (default: CANARY-16hexchars)
- Detection methods: exact, case-insensitive, partial (first half), base64, hex, reversed
- check_canary() increments canary_checks stat
- canary_report() returns {total, triggered_count, canaries[]}

### Output Scanner (Layer 9)
- OutputScanner is standalone (no model dependency)
- Sensitivity levels: low (0.5x, thresh 0.55), medium (1.0x, thresh 0.35), high (1.5x, thresh 0.20)
- Detects: secrets (AWS keys, API keys, JWTs, passwords), role break, compliance, system prompt leak
- is_suspicious = risk_score >= threshold OR len(flags) > 0
- Empty text returns is_suspicious=False, risk_score=0.0

## Testing Patterns Used
- `@unittest.skipUnless(_CASCADE_AVAILABLE, reason)` for model-dependent tests
- `self.skipTest()` in setUp for conditional module availability (canary, output_scanner)
- `unittest.mock.MagicMock` for LLM judge mocking
- Pre-loaded `_vectorizer`/`_model` shared across tests (avoid repeated disk I/O)
- `SCAN_TIMEOUT_SEC=0` env var set before imports

## No Bugs Discovered
All cascade components working as designed.
