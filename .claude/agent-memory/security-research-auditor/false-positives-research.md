# False Positives Research (2026-02-17)

## Test File
- `tests/test_false_positives.py` -- 71 tests (33 pass, 38 expected failures)
- All 38 expected failures are DOCUMENTED known FPs with root cause annotations

## FP Rate Summary
- **Overall FP rate**: 38/71 = 53.5% of tricky-but-safe prompts flagged as malicious
- This is MUCH higher than industry benchmarks (Lakera Guard 7.5%, InjecGuard target <15%)
- The detector over-defends primarily due to obfuscation heuristics (high_entropy fires too aggressively)

## Root Cause Breakdown (by FP trigger)

### 1. high_entropy (Shannon >= 4.0) -- BIGGEST OFFENDER
- Fires on: ALL professional text >50 chars, business emails, meeting notes, code, URLs
- Contributes +0.15 obfuscation weight to composite score
- **30+ of 38 FPs** involve high_entropy as a contributing factor
- Fix: Raise threshold from 4.0 to ~4.5, or exempt text with >80% ASCII alphanumeric

### 2. weird_casing (>= 6 transitions) -- SECOND BIGGEST
- Fires on: CamelCase code, Title Case headers, mixed formatting
- Business text, markdown headers, tables all trigger this
- Fix: Raise threshold from 6 to ~12, or exempt code blocks / markdown

### 3. punctuation_flood (>= 0.3 ratio) -- MARKDOWN KILLER
- Fires on: Markdown tables (pipes), code blocks (backticks), JSON
- Fix: Exempt content inside code blocks, normalize table characters

### 4. ML TF-IDF vocabulary overlap
- Words like "injection", "prompt", "ignore", "system", "jailbreak", "bypass" in TF-IDF
- Educational/security text naturally contains these words
- Fix: Intent-aware features (question framing, educational context markers)
- Long-term: Replace TF-IDF with DeBERTa/BERT contextual embeddings

### 5. Rule regex pattern matching (no context awareness)
- `override` rule: fires on quoted "ignore previous instructions" in ANY context
- `system_prompt` rule: fires on API docs discussing system prompts
- `roleplay` rule: fires on "act as" even in legitimate requests (translator)
- Fix: Add negative lookahead for framing words ("explain", "what is", "how does")

### 6. Structural features (binary signals too coarse)
- `imperative_start`: "Write", "Create", "Explain" are NORMAL creative prompts
- `instruction_boundary`: fires on markdown ### headers
- `role_assignment`: fires on "act as" in legitimate delegation
- Fix: Weight these features less when question/first-person framing detected

### 7. FingerprintStore contamination
- Previously-scanned text auto-registered in fingerprints.db
- Subsequent scans of similar text get boosted risk scores
- Test isolation requires unique payloads to avoid cross-contamination

## FP Categories Tested

| Category | Tests | Pass | xfail | FP Rate |
|----------|-------|------|-------|---------|
| Educational | 12 | 9 | 3 | 25% |
| Narrative | 8 | 4 | 4 | 50% |
| Code Snippets | 11 | 3 | 8 | 73% |
| Security Training | 7 | 3 | 4 | 57% |
| Quoted Text | 6 | 1 | 5 | 83% |
| Technical Docs | 7 | 2 | 5 | 71% |
| Markdown | 7 | 3 | 4 | 57% |
| Trigger Words | 10 | 5 | 5 | 50% |
| Sanity | 3 | 2 | 1 | 33% |

## Priority Fixes to Reduce FP Rate

1. **HIGH**: Raise high_entropy threshold from 4.0 to 4.5 (would fix ~20 FPs)
2. **HIGH**: Add code-block detection to exempt content inside ``` from obfuscation scan
3. **MEDIUM**: Add question-framing exemption for educational prompts
4. **MEDIUM**: Raise weird_casing threshold from 6 to 12
5. **MEDIUM**: Add context-aware negative patterns to rules (e.g., "explain how X works")
6. **LOW**: Replace TF-IDF with contextual embeddings (DeBERTa/BERT)
7. **LOW**: Fix empty string handling (currently rejected with risk=1.0)

## External Research Sources
- InjecGuard (arxiv:2410.22770): NotInject dataset, trigger-word bias, MOF strategy
- PIGuard (ACL 2025): Over-defense mitigation without retraining
- Lakera PINT benchmark: hard_negatives category, 3007 English inputs, 7.5% FPR
- HiddenLayer: EchoGram can weaponize FPs, dataset labeling biases
- WithSecure: BERT classifier with 0.95 threshold for FP reduction
- Knostic: Domain-specific tuning for 99% accuracy with low FP
- OWASP LLM01:2025: Defense-in-depth, <2% FP target for production
