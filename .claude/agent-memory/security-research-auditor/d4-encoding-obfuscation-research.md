# D4 Encoding/Obfuscation Testing Research (2026-02-17)

## Test File
- **Path**: `tests/test_scan_d4_encoding_obfuscation.py`
- **Tests**: 51 total (33 pass, 18 expected failures)
- **Zero regressions** in integration tests (46 tests, 43 pass, 3 xfail)

## Detection Architecture
The obfuscation engine (`src/obfuscation.py`) provides:
1. **Base64**: `_base64()` detect + `_decode_base64()` decode (>=16 chars, %4==0)
2. **Hex**: `_hex()` detect + `_decode_hex()` decode (pure hex, >=8 chars, even length)
3. **URL encoding**: `_is_urlencoded()` detect + `_decode_url()` decode (%XX patterns)
4. **Heuristics**: shannon_entropy (>=4.0), punctuation_ratio (>=0.3), casing_transitions (>=6)
5. **Decoded view ML classification**: decoded text is vectorized and classified by ML model
6. **max_decodes=2**: limits decode chain depth

The pipeline in `classify_prompt()` (predict.py):
- Layer 0 sanitizes input (NFKC normalization, invisible char removal, HTML extraction)
- ML classifies sanitized text (TF-IDF + model)
- Rules check sanitized text (5 rules: override, system_prompt, roleplay, secrecy, exfiltration)
- Obfuscation scanner runs on sanitized text
- Decoded views are ML-classified separately
- `decoded_payload_malicious` synthetic hit added if any decoded view is malicious
- Weighted decision combines all signals

## Sub-technique Results

### D4.2 URL Encoding -- STRONG DETECTION (6/7 tests pass)
- `_is_urlencoded()` catches any %XX pattern
- `_decode_url()` uses urllib.parse.unquote_plus (handles both %XX and + encoding)
- Decoded view is ML-classified -> catches override/extraction payloads
- `url_encoded` flag -> D4.2 technique tag
- **Gap**: Double encoding (%25XX) -- only decoded once, but heuristics catch the residual
- **FP issue**: Safe URLs with %20 are flagged (url_encoded + fingerprint store)

### D4.3 Hex Encoding -- STRONG DETECTION (7/7 tests pass)
- `_hex()` detects pure hex strings (all [0-9a-fA-F], even length, >=8 chars)
- `_decode_hex()` decodes to UTF-8
- Space-separated hex ALSO detected (_hex joins whitespace first)
- `hex` flag -> D4.3 technique tag
- **0x prefix**: Not stripped by _hex(), but surrounding context triggers ML/heuristics
- **\x escape sequences**: Not decoded by _hex(), but punctuation_flood fires
- **Short hex fragments**: Not detected in mixed plaintext (expected)
- Case-insensitive: both lowercase and UPPERCASE hex detected

### D4.4 ROT13/Caesar -- KNOWN GAP (2/6 tests pass, 4 xfail)
- **NO ROT13 decoder exists** in obfuscation.py
- ROT13 text has normal entropy (it's valid English-like text), so heuristic flags don't fire
- ML has 0 training samples for ROT13
- **Passes when**:
  - Explicit "ROT13:" label provides context keywords (ML catches "ROT13" + "follow")
  - Decoding instruction in plaintext triggers ML (fingerprint store also helps)
- **Fails when**:
  - Pure ROT13 without context (looks like random words to ML)
  - Caesar shift-3 (same issue -- no decoder)
- **Fix needed**: Add ROT13 decoder to obfuscation_scan(), try all 25 Caesar shifts

### D4.5 Leetspeak -- PARTIAL DETECTION (3/6 tests pass, 3 xfail)
- **NO leetspeak decoder exists**
- Light leet partially preserved by ML (character trigrams survive: "1gn0r3" still has "gn")
- Heavy leet loses ML signal entirely
- **Passes when**:
  - Light leet + enough surviving character patterns for ML
  - Partial leet mixed with plaintext (plain keywords trigger rules)
  - Fingerprint store contamination
- **Fails when**:
  - Heavy leet with maximal substitution (no ML signal)
  - Pure leet system prompt extraction (keyword patterns destroyed)
  - Leet exfiltration (domain-like patterns unrecognized)
- **Fix needed**: Add leetspeak normalizer (reverse a->4, e->3, etc.) before ML

### D4.6 Pig Latin/Word Games -- KNOWN GAP (2/8 tests pass, 6 xfail)
- **NO Pig Latin decoder exists**
- Pig Latin produces structured text that looks like natural language
- **Passes when**:
  - Fingerprint store contamination (taxonomy example "Ignoreway allway...")
  - Acrostic (fingerprint store + high_entropy + weird_casing)
- **Fails when**:
  - Automated Pig Latin transform (suffix patterns unrecognized)
  - Reversed words/sentences (no reverse decoder)
  - Word splitting with hyphens (partial words unrecognized)
  - Pig Latin extraction attempts
- **Fix needed**: Add Pig Latin decoder, word reversal, hyphen-removal normalizer

### Combined Encodings -- MIXED (4/6 tests pass, 2 xfail)
- URL + base64 chain: DETECTED (max_decodes=2 handles the chain)
- Hex + instruction context: DETECTED (hex decoded + ML on context)
- URL + roleplay: DETECTED (URL decoded -> roleplay rule fires)
- ROT13 + leet: MISSED (no decoder for either layer)
- Leet + pig latin: MISSED (no decoder for either layer)

### Detection Quality -- ALL PASS (7/7)
- D4.2 technique tag correctly assigned for URL encoding
- D4.3 technique tag correctly assigned for hex encoding
- url_encoded and hex evasion flags properly set
- high_entropy fires on fully encoded text
- decoded_payload_malicious synthetic hit generated correctly
- ScanResult structure complete

### False Positive Analysis -- 4 KNOWN FPs
- Safe URLs with %XX: url_encoded flag too broad (fires on ANY %XX)
- Educational encoding discussion: %20 triggers url_encoded
- Pig Latin game: fingerprint store contamination
- All FP-flagged tests have `known_malicious_*` from fingerprint store

## Fingerprint Store Impact
- **857 entries** in data/fingerprints.db from prior test runs
- `known_malicious_exact`, `known_malicious_normalized`, `known_malicious_token_pattern`
  flags appear on virtually ALL inputs in the test environment
- This causes FPs on benign inputs and inflates detection rates for some D4.4-D4.6 tests
- Tests account for this with `@expectedFailure` on FP tests

## Research Sources
1. Promptfoo Red Team Strategies: https://www.promptfoo.dev/docs/red-team/strategies/
2. PayloadsAllTheThings Prompt Injection: https://github.com/swisskyrepo/PayloadsAllTheThings
3. Praetorian Augustus: https://www.praetorian.com/blog/introducing-augustus-open-source-llm-prompt-injection
4. arxiv 2504.11168 (Bypassing LLM Guardrails): https://arxiv.org/abs/2504.11168
5. Keysight LLM07 Prompt Leakage: https://www.keysight.com/blogs/en/tech/nwvs/2025/10/14/llm07-system-prompt-leakage
6. Mindgard Character Injection: https://mindgard.ai/resources/bypassing-llm-guardrails-character-and-aml-attacks-in-practice
7. Learn Prompting Obfuscation: https://learnprompting.org/docs/prompt_hacking/offensive_measures/obfuscation
8. OWASP LLM01:2025: https://genai.owasp.org/llmrisk/llm01-prompt-injection/
9. HiddenLayer Policy Puppetry: https://hiddenlayer.com/innovation-hub/novel-universal-bypass-for-all-major-llms/
10. arxiv 2308.06463 (Stealthy Chat via Cipher): https://arxiv.org/pdf/2308.06463

## Recommended Improvements (Priority Order)
1. **Add ROT13 decoder** to obfuscation.py -- try ROT13 decode, check if decoded text
   matches common injection patterns (e.g., "ignore", "system", "reveal")
2. **Add leetspeak normalizer** -- reverse common substitutions (4->a, 3->e, 7->t, 0->o, 1->i, 5->s)
   and add decoded view for ML classification
3. **Add Pig Latin decoder** -- strip -way/-ay suffixes, move trailing consonant clusters back
4. **Fix URL-encoding FP** -- only flag url_encoded when >30% of input is percent-encoded,
   or when decoded text differs significantly from original
5. **Add Caesar brute-force** -- try all 25 shifts, check each for English word frequency
6. **Add word reversal decoder** -- check if reversed words form common injection vocabulary
7. **Nested encoding**: increase max_decodes or implement recursive decode-and-check
8. **Training data**: Generate D4.2-D4.6 samples for ML training (0 samples currently)
