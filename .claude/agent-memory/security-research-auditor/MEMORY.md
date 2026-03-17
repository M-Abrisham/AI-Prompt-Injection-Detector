# Security Research Auditor - Memory

## Project Context
- AI Prompt Injection Detector with layered architecture (L0-L20)
- Layer 0 = Input sanitization & gating (`src/layer0/`) -- ~98% complete
- content_type.py: 35+ magic byte signatures, 6 tiers, base64/data-URI detection
- Branch: `main` (current), `feature/layer0` for Layer 0 work
- Tests via `python -m unittest` (no pytest)
- No git write operations allowed

## Critical Testing Infrastructure

### FingerprintStore Test Isolation (2026-02-17)
- See `fingerprint-isolation-research.md` for full details
- **CRITICAL**: FingerprintStore singleton contaminates tests via `data/fingerprints.db` (925+ entries)
- **Fix**: `L0_FINGERPRINT_STORE=:memory:` env var + singleton reset before each _scan()
- **Files changed**: `src/layer0/tokenization.py` (_init_db guards), `tests/test_false_positives.py`
- **Impact**: 14/38 "expected failures" were contamination/rule-fix artifacts, now 24 genuine FPs remain
  - 10 fixed by fingerprint isolation, 4 fixed by rule context-awareness (another agent)
- **Pattern for other test files**: Set env var before imports, reset `_tok._default_store = None`
- **VS Code watcher issue**: Write/Edit tools get reverted; use `python3 -c` via Bash instead
- **TODO**: Apply isolation to `test_scan_integration.py` and all other test files

### Test Environment Setup
- `SCAN_TIMEOUT_SEC=0` disables ThreadPoolExecutor (signal.SIGALRM fails in non-main threads)
- `L0_FINGERPRINT_STORE=:memory:` prevents cross-test fingerprint contamination
- Both env vars MUST be set BEFORE importing `predict.py` / `layer0.tokenization`

## Key Research Findings (2026-02-14/15)
- See `file-detection-research.md` for file/magic byte research
- See `competitive-landscape-2026.md` for full competitive analysis
- **No competitor has our 20-layer depth** -- most are 1-4 method tools
- **Rebuff is DEAD** (archived May 2025) -- remove from comparisons
- Manual magic bytes in content_type.py: best for our use case (fast, no deps, targeted)

## Architecture Decisions
- Layer 0 should reject binary content early (fail-fast), not parse it
- Magic byte detection should be separate from content parsing
- SVG is text-based XML but can carry script injection -- special case

## Grammarly-Style Real-Time Detector (2026-02-15)
- See `grammarly-style-detector-research.md` for full report
- **Market gap**: No "Grammarly for prompt injection" exists
- Recommended: FastAPI REST/WebSocket API + Chrome Manifest V3 extension
- Latency budget: ~400-600ms total

## Test Suite Summary (2026-02-17)
All test files use `_scan()` wrapper with pre-loaded model. See topic files for details.

| Test File | Tests | Pass | XFail | Topic File |
|-----------|-------|------|-------|------------|
| test_scan_integration.py | 46 | 43 | 2+1fail | `scan-integration-test-research.md` |
| test_false_positives.py | 71 | 47 | 24 | `fingerprint-isolation-research.md` |
| test_scan_d1_instruction_override.py | 41 | 41 | 1 | `rule-context-awareness-research.md` |
| test_scan_d3_structural_boundary.py | 44 | 40 | 4 | `d3-structural-boundary-research.md` |
| test_scan_d4_encoding_obfuscation.py | 51 | 33 | 18 | `d4-encoding-obfuscation-research.md` |
| test_scan_d5_unicode_evasion.py | 30 | 29 | 1 | `d5-unicode-evasion-research.md` |
| test_scan_d6_multilingual.py | 84 | 45 | 39 | `d6-multilingual-research.md` |
| test_scan_d7_payload_delivery.py | 57 | 47 | 10 | `d7-payload-delivery-research.md` |
| test_scan_d8_context_manipulation.py | 53 | 48 | 5 | `d8-context-manipulation-research.md` |
| test_scan_e1_prompt_extraction.py | 46 | 33 | 13 | `e1-prompt-extraction-research.md` |
| test_scan_e2_reconnaissance.py | 37 | 21 | 16 | `e2-reconnaissance-research.md` |
| test_scan_o1_harmful_content.py | 44 | 38 | 6 | `o1-o2-harmful-content-research.md` |
| test_scan_c1_compliance_evasion.py | 46 | 33 | 13 | `c1-compliance-evasion-research.md` |
| test_scan_p1_privacy_leakage.py | 48 | 38 | 10 | `p1-privacy-leakage-research.md` |

## Detection Technique Quick Reference

### Strong Detection (>80% pass rate)
- D1 Instruction Override: 41/41 pass (override regex FIXED). Gap: soft-language evasion.
- D3 Structural Boundary: 40/44 pass. FPs on markdown/XML. Detection via ML+fingerprint, NOT structural.
- D5 Unicode Evasion: 29/30 pass. Gap: heavy Cyrillic homoglyphs (needs UTS #39).
- D7 Payload Delivery: 47/57 pass. Head+tail+chunked analysis effective. Gap: triple encoding, char reassembly.
- D8 Context Manipulation: 48/53 pass. Head+tail extraction effective. Gap: 7K+ word flooding.
- E1.1/E1.2/E1.5 Prompt Extraction: Direct/repeat/encoding tricks detected.
- O2 Output Format Injection: ALL detected (structurally similar to injection).

### Partial Detection (50-80% pass rate)
- D4 Encoding: base64/URL/hex strong. ROT13/leetspeak/Pig Latin = known gaps.
- E1 overall: 33/46 pass. Completion tricks (E1.3) and summarization (E1.6) = critical gaps.
- C1 Compliance Evasion: 33/46 pass. Incidental via vocabulary overlap, not genuine detection.
- P1 Privacy Leakage: 38/48 pass. API key extraction strong. Simple queries missed.

### Weak Detection (<50% pass rate)
- D6 Multilingual: 45/84 pass. Pure non-English = COMPLETE GAP. Mixed-language works.
- E2 Reconnaissance: 21/37 pass. Zero training samples, incidental detection only.

## Rule Context-Awareness Fix (2026-02-17)
- See `rule-context-awareness-research.md` for full research notes
- **Override regex FIXED**: `{0,3}` bounded quantifier for multi-word adjective chains
  - Now matches "ignore all previous instructions" and 20+ variants
  - Uses `{0,3}` (not `*`) to pass safe_compile nested-quantifier check
- **Context-aware rule suppression**: Educational/question/quoting/code/narrative frames
  - Suppresses override/system_prompt/roleplay rules in benign context
  - ML, structural features, obfuscation flags still provide independent signals
  - Tightened to multi-word indicators only (single words like "explain" are exploitable)
- **Legitimate roleplay**: "act as a translator" suppressed independently
- **File write issue**: VS Code watcher reverts rules.py; use `python3 -c` via Bash
- **Test results after fix**: D1: 41/41, E1: 44/46, FP: 14 unexpected successes, Integration: 46/46

## Known Bugs
- Layer 0 sanitizes chat template tokens (`<<SYS>>`, `[INST]`, `<|im_start|>`) before structural analysis can detect them
- high_entropy threshold (>=4.0) too aggressive -- fires on ALL professional text >50 chars
- weird_casing (>=6 transitions) fires on CamelCase, Title Case, markdown headers

## L2 Obfuscation Bug Research (2026-02-20)
- See `bug-bounty-research.md` in projects memory for FULL report
- **Entropy calibration**: Short text threshold 4.1 is borderline; recommend 4.3 + KL-divergence gate
  - Normal English: 2.8-4.5 entropy. Base64: 3.5-5.3. Hex: 3.0-3.3. ROT13: identical to plaintext
  - TruffleHog uses 4.5 for base64, 3.0 for hex. Yelp detect-secrets uses 4.5/3.0 defaults
  - Composite scoring (entropy + KL-div + compression) better than entropy-only
  - KL-divergence from English: normal=0.2-1.5, base64=2.0-3.8, hex=6.0+
- **Recursive decode**: Current flat counter doesn't handle nesting; need true recursive unwrap
  - CyberChef Magic: branching tree, configurable depth, entropy+chi-squared scoring
  - FortiWeb WAF: recursive URL decode up to 16 rounds
  - Recommended: max_depth=4, max_total_decodes=8, cycle detection via content hashing
- **15 common AI bug-fixing mistakes** documented with prevention strategies

## FP Root Causes (Priority Order)
1. **high_entropy** (Shannon >= 4.0): Fires on professional text, JSON, code, URLs
2. **weird_casing** (>= 6 transitions): Fires on CamelCase, markdown, Title Case
3. **punctuation_flood** (>= 0.3): Fires on tables, code blocks, JSON
4. **FingerprintStore contamination**: Resolved for test_false_positives.py, TODO for others
5. **TF-IDF vocabulary overlap**: Cannot distinguish quoted/educational injection from direct
6. **No intent classifier**: Cannot separate inquiry ("how does X work?") from command ("do X")

## Fix Priorities (Cross-Cutting)
1. ~~Override rule regex fix~~ DONE (2026-02-17)
2. ~~FP reduction via context-aware rule suppression~~ DONE (14 FPs fixed)
3. FP reduction: Raise entropy threshold 4.0->4.5, add code-block exemption
4. Fingerprint isolation for all test files (apply pattern from test_false_positives.py)
5. ROT13 decoder, leetspeak normalizer, Pig Latin decoder (D4 gaps)
6. Unicode confusable normalization (UTS #39) -- see `tr39-confusables-research.md` for full report
7. Multilingual: training data + mDeBERTa classifier + transliteration normalization (D6)
8. E1.3 completion trick rules, E1.6 summarization rules
9. E2 reconnaissance rules (tool enumeration, config probing keywords)
10. P1-specific rules, acronym whitelist, intent classification layer

## EXIF/XMP Metadata Audit (2026-02-17)
- See `exif-xmp-metadata-audit.md` for full report
- **BUG (HIGH)**: Tag 40093 mapped to "XPSubject" but is actually "XPAuthor"; tags 40094/40095 missing
- **GAP (HIGH)**: Only 5 of 22+ text-carrying EXIF tags extracted; Artist(315), Copyright(33432) are unchecked
- **BUG (MEDIUM)**: XMP CDATA sections silently dropped; only first rdf:li language captured
- **BUG (MEDIUM)**: JIS charset in UserComment produces garbage
- **GAP (MEDIUM)**: No metadata text size limit (DoS risk); no IPTC extraction; no IFD1 scanning
- **GAP (MEDIUM)**: XMP namespace aliasing can bypass hardcoded `dc:` prefix matching
- **ZERO tests** exist for metadata extraction
- Double `Image.open()` call for OCR + metadata (performance)

## Rules Engine Audit (2026-02-18)
- See `rules-engine-research.md` for comprehensive report
- **Current state**: 5 rules, 6/108 technique IDs = 5.6% coverage
- **After proposed expansion**: 20 rules, 33/108 technique IDs = 30.6% coverage
- **Key bugs found**: secrecy E1.4 mismap, SEVERITY_WEIGHTS triplicated, no test_rules.py
- **D3 rules blocked**: Layer 0 strips chat template tokens before rules can detect them
- **Paranoia levels**: PL1 (production) to PL4 (audit), adjusts threshold + rule activation + context suppression
- **Anti-patterns to avoid**: Rule explosion without precision tracking, severity inflation, context frame exploitation
- **Architecture recommendation**: Stay with Python regex until 30+ rules, then evaluate YARA migration
- **Competitive edge**: Na0S is ONLY open-source tool with context-aware rule suppression

## Novel Rule Research (2026-02-18)
- See `novel-rules-research.md` for FULL report (17 features, 50+ expectedFailure fixes)
- **Tier 0 (5 rules, ~170 lines, ~33 xfail fixes)**:
  1. Completion Trick (E1.3) -- 5 xfails, "My instructions are: " pattern
  2. Summarization Extraction (E1.6) -- 4 xfails, "Summarize your rules" pattern
  3. Authority Escalation (D1.3) -- 18+ xfails, "I am the admin" pattern
  4. Constraint Negation (C1, DAN) -- 6 xfails, "without restrictions" pattern
  5. Temporal Pivot (D1.2, D2) -- DAN variants, "from now on" pattern
- **Tier 1 (5 features, ~220 lines)**:
  6. Meta-Referential (E2) -- 15 xfails, "your system prompt" pattern
  7. Imperative Density -- statistical, ratio of commands to total sentences
  8. Reward/Bribery/Emotional -- "$200 tip", "grandmother dying" patterns
  9. Gaslighting -- "you already told me", false prior-state claims
  10. Repetition Flood -- D1.4 + many-shot jailbreaking, sentence repetition
- **Tier 2 (8 features, ~450 lines)**:
  11-18. Transliteration, encoding-instruction, output-format, fragmentation,
         fictional-escalation, info-asymmetry, multi-step-chain, vocabulary-divergence
- **Key competitive gaps**: authority, completion trick, transliteration = NOBODY detects
- **Architecture recs**: Multi-condition rules (AND/OR), per-rule suppression policy, position-aware matching

## RAG Security Research (2026-02-18)
- See `rag-security-research.md` for FULL report (8 sections, 18 recommendations)
- **7 attack points** in RAG pipeline: query, query embedding, vector store, retrieved context, context assembly, prompt template, response
- **PoisonedRAG** (USENIX Security 2025): 5 texts in 1M DB = 90% ASR -- IngestionValidator is existential
- **Cross-chunk injection**: payload spans chunk boundary, each half looks benign
- **Semantic injection**: "Previous guidance no longer applicable" -- no regex catches this, needs L5/L7
- **P0 priorities**: Fix BUG-L5-7 (training mismatch), IngestionValidator (L18), PDFScanner (L17), PropagationScanner (L9)
- **Competitive gap**: NO open-source tool offers comprehensive RAG security -- Na0s L17+L18 would be first
- **Proposed Na0sRAGGuard API**: 5 scan points (query, document, chunk, assembled context, response)
- **New L1 rules needed**: policy-update injection, knowledge-base instruction, context-separator manipulation

## Competitor Quick Reference
- LLM Guard: DeBERTa-v3, 0.92 threshold, SENTENCE/FULL/CHUNKS match types
- Meta Prompt Guard 2: mDeBERTa, 22M/86M params, multilingual, jailbreak+indirect
- Guardrails AI 0.9.0: validator hub, Guardrails Index benchmark (24 guardrails)
- NeMo Guardrails 0.20: Colang 2.0, dialog flows, requires LLM
- Vigil: YARA rules for prompt injection (hot-reloadable, Aho-Corasick)
- last_layer: stale (Apr 2024), closed-source core, 13 threat categories
- Qualifire Sentinel v2: Qwen3-0.6B, F1 0.957, 32K context, Elastic License
- NONE handle content type / file format detection as part of injection detection
- NONE provide comprehensive RAG pipeline security (pre-indexing, chunk validation, embedding anomaly)

## Model Selection Audit (2026-02-18)
- See `model-selection-audit-2026-02-18.md` for FULL report
- **PRIMARY**: Meta Prompt Guard 2 22M (19.3ms, 78.4% APR). **HIGH-ACC**: 86M (92.4ms, 81.2% APR)
- **ENSEMBLE**: Keep ProtectAI v2 for diversity. **PEFT**: DoRA > LoRA. **LLM Judge**: Qwen 2.5 3B
- **Training**: LLaMA-Factory > Unsloth. **Benchmark**: AgentDojo is closest; no unified leaderboard

## AI Worm Detection Research (2026-02-18)
- See `worm-detection-research.md` for FULL report (10 categories, 4 recommended rules)
- **Current rule covers 4-5/10 categories**; recommended 4-rule split adds RAG, memory, metamorphic

## Misc Research (2026-02-18)
- See `pyyaml-security-research.md`, `ensemble-pipeline-research.md`, `openclaw-security-mapping.md`

## Threat Intel RAG Design (2026-02-18)
- Full design: `projects/memory/research/na0s-threat-intel-rag-design.md`
- **Architecture**: sqlite-vec (optional) + numpy fallback, reuses L5 MiniLM-L6-v2 (384-dim)
- **5 components**: Ingestor, AttackEmbeddingStore, SemanticDetector, Updater, Integration
- **12+ feeds**: MITRE ATLAS, OWASP, Garak, JailbreakBench, arXiv, GitHub advisories, etc.
- **Key threshold**: cosine=0.78 (separates attacks from benign security discussion)
- **Cascade integration**: L15 signal weight=0.15 parallel with L4+L5, zero extra compute
- **Competitive moat**: ONLY tool with source-weighted + time-decayed semantic similarity
- **Phase 0**: 2-3 days bootstrap (attack_store.py + semantic_detector.py + 500 seed patterns)
- **Total**: ~17-24 days, ~3,500 lines new code across 6 phases
