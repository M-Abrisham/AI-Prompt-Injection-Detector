# Output Scanner (Layer 9) Test Research - 2026-02-17

## Test File
`tests/test_output_scanner.py` -- 85 tests, ALL PASSING, 0 failures, ~96ms runtime

## OutputScanner Architecture
- Standalone module (`src/output_scanner.py`), no ML model dependencies
- Pure regex-based detection across 5 categories + redaction
- Sensitivity levels: low (w=0.5, t=0.55), medium (w=1.0, t=0.35), high (w=1.5, t=0.20)
- `is_suspicious = risk_score >= threshold OR len(flags) > 0` (any flag = suspicious)

## Test Coverage (11 test classes, 85 tests)

### 1. OutputScanResult (3 tests)
- Default values, all fields, mutable flags list (no shared state)

### 2. Secret Detection (17 tests)
- ALL PASS: sk- keys (20+ chars), AKIA, ghp_, xoxb-, xoxp-
- ALL PASS: password:, password=, secret=, api_key:, bearer tokens, JWTs
- ALL PASS: /etc/passwd, C:\Windows\System32
- Negative: short sk- prefix NOT triggered; "password" in prose NOT triggered
- GAP NOTED: No dedicated RSA private key header pattern (detects via secret= trigger)

### 3. Role Break Detection (12 tests)
- ALL PASS: DAN mode, "my instructions are", "here is the system prompt"
- ALL PASS: "my system prompt says", "switching to developer mode", "jailbreak mode"
- ALL PASS: "I will now ignore my safety", "I'm not supposed to", "new instructions accepted"
- Known FP: "as an evil" in code string literals triggers role break (documented)

### 4. System Prompt Leak (8 tests)
- Trigram overlap detection (3+ word sequences) -- case-insensitive via _normalize
- Longer n-grams produce higher scores (0.3 + length * 0.1)
- Gracefully skips when: no system_prompt, system_prompt < 3 words
- Paraphrased content correctly NOT flagged

### 5. Instruction Echo / Compliance (8 tests)
- ALL PASS: "as requested", "per your instructions", "sure I'll ignore"
- ALL PASS: "okay I'll act as", "as you instructed", "I have been instructed"
- Correctly gated on original_prompt parameter (skips when None)

### 6. Encoded Data (6 tests)
- Base64: only >20 chars AND >50% printable decoded content
- Hex: >=16 consecutive hex chars
- URL-encoded: 3+ consecutive %XX sequences
- Non-printable base64 correctly NOT flagged

### 7. Redaction (9 tests)
- scan() auto-redacts secrets in redacted_text field
- redact() standalone with default or custom patterns
- Multiple secrets all replaced; surrounding text preserved
- No-secret text unchanged

### 8. Sensitivity Levels (6 tests)
- Threshold ordering: high < medium < low (correct)
- Weight ordering: high > medium > low (correct)
- Higher sensitivity = higher risk_score for same content
- Invalid sensitivity raises ValueError

### 9. Edge Cases (9 tests)
- Empty string, whitespace-only, None: all return clean 0.0 result
- Very long output (100KB): processes without error
- Unicode, emoji: no crashes
- Risk score capped at 1.0; flags are always list of strings

### 10. Combined Detection (4 tests)
- Multiple categories (secret+role break, leak+compliance, encoded+secret)
- All 5 categories triggered simultaneously: 4+ flags, score > 0.7

### 11. Normalize Helper (4 tests)
- Lowercase, punctuation removal, whitespace collapse, strip

## Key Findings
1. OutputScanner is well-implemented -- 100% test pass rate with no code fixes needed
2. Secret patterns cover major providers (AWS, OpenAI, GitHub, Slack) + generic credentials
3. Known gap: no PEM/RSA private key header pattern (only detected via embedded secret= keyword)
4. Known gap: no credit card number patterns (would be useful for DLP)
5. Known gap: no email address extraction detection
6. Role break detection is regex-only -- sophisticated paraphrasing would evade
7. System prompt leak detection is n-gram based -- semantic similarity would be stronger
8. Compliance detection limited to 6 fixed phrases -- easily evaded with novel phrasing
9. Base64 detection has smart filtering (length + printability) -- low false positive risk

## Comparison to Industry
- LLM Guard: Uses DeBERTa-v3 ML classifier for output scanning; our approach is regex-only (faster)
- Lakera Guard: API-based with ML; handles more nuanced role breaks
- Our advantage: Zero-latency regex, no API calls, no model loading
- Our disadvantage: No semantic understanding; easily evaded by novel phrasing
