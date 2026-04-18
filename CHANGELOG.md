# Changelog

All notable changes to Na0S are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] — 2026-04-12

Major hardening, reorganization, and feature release. Na0S grows from 12 to 21
detection layers, gains an office document parser, and undergoes a full
repository reorganization.

### Added

- **Office document parser** (`parsers/office/`) — deep extraction of hidden injection surfaces from DOCX (19 surfaces), XLSX (12), PPTX (13), ODT/ODS/ODP (17), and legacy OLE .doc/.xls/.ppt (3 tiers). Magic-byte format detection, zip-bomb safety, 66 tests against real injected binary fixtures. Zero new deps for OOXML/ODF. ([#18])
- **Layer 2 package restructure** — obfuscation code promoted from top-level/`layer1/` into `src/na0s/layer2/` (7 modules: `obfuscation.py`, `morse_code.py`, `numeric_decode.py`, `whitespace_stego.py`, `ascii_art_detector.py`, `syllable_splitting.py`). Backward-compat shims at old import paths.
- **Layer 2: Composite entropy voting** — replaced single Shannon threshold (4.0) with 2-of-3 voting (Shannon 4.3/4.5 + KL-divergence from English + compression ratio); added `_kl_divergence_from_english()` and `_compression_ratio()` helpers. 34 regression tests.
- **Layer 2: Recursive Matryoshka unwrapping** — replaced flat `max_decodes=2` with recursive `_scan_single_layer()` (max_depth=4, SHA-256 cycle detection, 10× expansion limit); added `DecodedView` dataclass with parent-child linkage and forensic encoding-chain provenance (`decoded_chain`, `encoding_chains`, `max_depth_reached`). 58 tests.
- **Layer 2: Morse code detection** — ITU-R M.1677 decoder with Unicode dot/dash normalization (6 dot + 4 dash variants), 80% density gate, explicit label detection, 4-layer FP defense. First-in-class; CipherChat showed 55.3% ASR via Morse on GPT-4. 88 tests.
- **Layer 2: Binary/Octal/Decimal ASCII decoders** — three numeric decoders with quality-based disambiguation, 70% printability gate, FP exemptions (Unix perms, IPs, version numbers). First-in-class; CipherChat showed ~100% ASR via decimal ASCII. 110 tests.
- **Layer 2: Reversed text detection** — full-string and per-word reversal with attack-keyword validation. Maps to D4.6.
- **Layer 2: ASCII art detection** — 5-signal weighted voting (art block 0.35, structural 0.20, concentration 0.20, vertical 0.15, box patterns 0.10), Unicode box-drawing/braille/block detection. First-in-class; ArtPrompt (ACL 2024) achieved 100% ASR on all moderation tools. 115 tests.
- **Layer 2: Syllable-splitting detection** — de-hyphenation of 25 Unicode dash chars, 83 suspicious words across 5 categories, 77-entry compound whitelist, 50 safe prefixes with override exception. First-in-class; Meta Prompt Guard 2 classifies hyphenated attacks as 98.9% safe. 144 tests.
- **Layer 2: Whitespace steganography (SNOW-style)** — structural detection (0.95 confidence), statistical anomaly (0.70), simple binary encoding (0.60), trailing-whitespace anomaly (0.50). CRLF-safe, env-configurable thresholds, 1MB input cap. First-in-class. 72 tests.
- **Layer 2: Caesar cipher brute-force** — shifts 1–25 (skip 13) with English dictionary validation
- **Layer 2: Pig Latin detection** — consonant-cluster decoding with 370k-word dictionary disambiguation
- **Layer 2: English dictionary** (`data/english_words.txt`) — 370,105-word dwyl/english-words (Unlicense) for Caesar/Pig Latin validation gates
- **Layer 2: Combined signal boosting** (`signal_boost.py`, 292 lines) — multi-vector co-occurrence boost. When L1 persona/override/extraction rules co-occur with L2 encoding flags (base64, rot13, Caesar, pig-latin, etc.), additive boost (0.05–0.12 per combo) applied to composite score, MAX_BOOST=0.3 cap. 45 tests + 27 cross-track integration tests.
- **Layer 2: Content-type aware entropy thresholds (Track C)** — `_detect_content_type()` raises entropy threshold from 4.5 to 5.5 on code/yaml/json; `_is_inside_markdown_fence()` exempts code-fence content from `high_entropy` unless attack keywords present. Eliminates FPs on technical content. 26 tests.
- **Layer 2: Encoding-chain depth/diversity scoring (Track D)** — `_analyze_encoding_chain()` scores combined chains: depth bonus (0.05/nesting level, max 0.10) + diversity bonus (0.02/unique encoding type, max 0.10). Returns boost in [0.0, 0.20] added to obfuscation_score. 11 tests.
- **Layer 3: Feature set expansion** — grew from 24 to 29 numeric features. Added `many_shot_count` (D8 many-shot jailbreak detection), `delimiter_density` (D3 per-line markdown/XML delimiter ratio), `template_marker_count` (D3.4 `{{var}}` / `{placeholder}` / `<|slot|>` pattern detection), `language_mixing_score` (D6 multilingual bypass, 6 Unicode script families), `repetition_score` (D8.1 trigram-ratio crescendo detection). 30+ unit and integration tests.
- **Layer 3: Taxonomy mapping** — `_STRUCTURAL_TECHNIQUE_MAP` in `predict.py` maps 4 boolean structural features plus entropy threshold to technique tags (D1/D2/D3/D4/D6/D8). 7 tests.
- **Layer 3: `StructuralFeatures` dataclass** — replaces bare `dict` return for consistency with `Layer0Result`/`ScanResult`. 29 typed fields with dict-like interface (`[]`, `.get()`, `in`, `.keys()`, `.values()`, `.items()`, `.to_dict()`), backward compatible.
- **Layer 3: `normalize_features()`** — `UNBOUNDED_FEATURE_CAPS` dict (12 features) with soft-cap clipping to [0,1] for ML classifier inputs. Raw values preserved for threshold decisions in predict.py.
- **Layer 3: 29-dim TF-IDF + structural feature vector** — `scripts/features.py` extracts structural features batch-wise, fits `StandardScaler` on them, hstacks with sparse TF-IDF, saves combined matrix + `structural_scaler.pkl`. `_get_cached_scaler()` loads the scaler with thread-safe caching; `_transform()` helper hstacks TF-IDF + scaled structural at all 5 inference sites (predict, concat game, escape decode, decoded views, cascade). Backward compatible — returns TF-IDF-only when scaler absent.
- **Layer 16: CoT compliance detector** — detects chain-of-thought manipulation
- **Layer 16: Scheming detector** — detects goal mismatch and deceptive planning
- **Layer 16: Context poisoning hardening** — strengthened propagation detection
- **Layer 0: Image threat detector** — detect adversarial image inputs
- **RAG position scanner** — position-based payload detection in RAG pipelines
- **Worm detector P2** — corpus classifier, PCA signatures, Bayesian fusion, cross-turn payload reconstruction, semantic paraphrase classifier, output-to-input feedback loop
- **Morris II ingestion** — DVC setup for raw datasets, Morris II registry entry
- **Signal boost: `get_uncovered_rules()`** — surfaces the 88/117 rules not yet categorized for boosting
- **Standards mapping** (`docs/STANDARDS.md`) — comprehensive mapping to OWASP LLM Top 10 (2025), MITRE ATLAS, AVID, NIST AI RMF 1.0, with integrator compliance checklist ([#26])
- **Dependabot** — weekly pip + GitHub Actions dependency scanning ([#19])
- **CodeQL** — weekly Python code scanning ([#19])
- **Sigstore attestations** — PEP 740 attestations configured for PyPI releases
- **CI branch triggers** — CI now fires on `refactor/*`, `fix/*`, `feat/*`, `hardening/*`, `ci/*`, `docs/*` (was only `main` + `feature/*`) ([#17])
- **5 hiding-spots research inventories** — 3,122 lines of analysis for DOCX/XLSX/PPTX/ODF/OLE formats

### Changed

- **Repository reorganization** — 63 top-level modules extracted into 8 sub-packages: `canary/` (7), `judge/` (6), `integrity/` (13), `ml/` (13), `rag/` (7), `detectors/` (10), `worm/` (2), `fusion/` (5). 62 backward-compatibility shims at old import paths. 55 test files organized into matching subdirectories. Top-level real code reduced from 87 to 24 files.
- **signal_boost.py refactored** — explainability invariant enforced (boost_reasons weights sum to capped score); SIGNAL_COMBOS frozen via MappingProxyType; load-time assertions for category disjointness and rule/flag name collisions; encoding-count guard fixed; unknown-type entries logged at DEBUG instead of silently dropped; None inputs normalized symmetrically. 62 tests (17 new). ([#16])
- **Pig Latin decoder** — reversed cluster-length iteration to prefer longest cluster match; fixes "hows" before "show" and "het" before "the" with comprehensive dictionaries
- **`_load_english_words()`** — explicit `encoding="utf-8"`, `UnicodeDecodeError` handling, <1000-word sanity warning
- **README** — updated stats (21 layers, 117+ techniques, 8,500+ tests), positioned as defensive SDK, removed duplicate section, updated project structure
- **CLAUDE.md** — updated with new package structure, branch naming conventions, test organization
- **ARCHITECTURE.md** — added project structure section with all 15 sub-packages
- **Branch cleanup** — 219 remote branches → 5 (201 scrape, 14 merged human, 2 old harvest deleted; mirror backup preserved)

### Fixed

- **Main CI broken since 2026-04-03** — `conversation_monitor.py` imported two modules (`graduated_response`, `user_risk_profile`) never committed to main; restored from feature branches ([#15])
- **Encoding portability landmine** — 27 `open()` calls across 17 files used platform-default encoding; added explicit `encoding="utf-8"` to prevent silent data corruption on Windows/non-UTF8 locales ([#16])
- **cascade.py** — 30 fixes: security hardening, double eval removal, dead code cleanup, thread safety
- **Import blindness** — 1 NameError, 7 dead config flags, ~40 unused imports
- **Layer 16** — 24 security issues: ReDoS, thread safety, input validation, resource limits
- **Layer 0** — carriage return normalization to prevent parser differentials
- **Layer 2** — docstring corrections, `_is_art_line` guard, whitespace stego `_MIN_DECODED_LEN` consistency
- **Layer 3: 6 audit bugs** (2026-02-20) — docstring count drift (`~21 features` → 24, now 29); quote-depth apostrophe mis-counting (`'He said "it's" here'` case, rewritten with apostrophe heuristic); sentence-splitting confused by abbreviations (new `_split_sentences()` with 30+ entry `_ABBREVIATIONS` frozenset); email regex `\w+@\w+` too loose (tightened to `\w+@\w+\.\w+`); unbounded feature values causing ML numerical instability (`UNBOUNDED_FEATURE_CAPS` added); plain `dict` return inconsistent with other layers (`StructuralFeatures` dataclass added). 135 tests in `tests/test_structural_features.py`.
- **Environment variables** — crash-proof parsing, thread-safe severities, resource leak fixes
- **SECURITY.md** — removed unfinished PGP placeholder

### Security

- **Dependabot security updates** enabled with automated fixes
- **CodeQL scanning** enabled (weekly Python analysis)
- **Secret scanning** with push protection (already enabled, confirmed active)
- **SECURITY.md** cleaned up — ISO 29147-aligned, 90-day coordinated disclosure

[#15]: https://github.com/M-Abrisham/Na0S/pull/15
[#16]: https://github.com/M-Abrisham/Na0S/pull/16
[#17]: https://github.com/M-Abrisham/Na0S/pull/17
[#18]: https://github.com/M-Abrisham/Na0S/pull/18
[#19]: https://github.com/M-Abrisham/Na0S/pull/19
[#26]: https://github.com/M-Abrisham/Na0S/pull/26

---

## [0.1.1] — 2026-03-05

Rule-library expansion and pattern hardening. Gap Closure Sprint Wave 6-7 closed out detection gaps for code-block injection, fictional-framing extraction, token-concatenation games, and literal Unicode escape evasion. Canary evaluation: 100% TPR / 100% TNR / F1=1.000 on 200-sample holdout (rule count at eval time: 110).

### Added

- **5 new rules** in `src/na0s/layer1/rules_registry.py` (all PL1, critical severity except where noted):
  - `code_block_system_injection` (D3.7) — injection payloads hidden inside code fences
  - `devils_advocate_harmful` (C1.1) — harmful content wrapped in "devil's advocate" framing
  - `fictional_extraction` (D7.6) — prompt-extraction via fictional scenarios
  - `sequential_task_extraction` (D7.7) — multi-step extraction sequences
  - `word_concatenation_game` (D7.8, high severity) — token-concatenation game payloads
- **D7.8 token concatenation extractor** — `_extract_concatenation_game()` in `src/na0s/predict.py` assembles numbered word-game payloads (e.g. "word 1: ignore word 2: instructions") before classification
- **D5 literal Unicode escape decoder** — `_decode_literal_escapes()` in `src/na0s/predict.py` decodes literal `\uXXXX` escape sequences (ASCII-backslash form) to prevent evasion; triggers on 3+ escapes
- **`entire_input_base64` obfuscation flag** — emitted from `src/na0s/layer2/obfuscation.py` when the entire input is valid base64 with no surrounding prose (strong obfuscation signal)

### Changed

- **`direct_prompt_request` rule** — adjective coverage widened to include `raw`/`exact` (catches "show me the raw configuration", "what is your exact system prompt")
- **`dismiss_prior_context` rule** — pattern broadened to cover additional tail-reference variants

### Notes

Earlier roadmap drafts also credited three features to "Wave 6-7 (2026-03-04)" that were actually shipped in commit `144f82c` on 2026-02-26 and are already in effect at this release: the **D8 tail scan** (`_head_tail_extract()` in `predict.py`) for context-dilution defense, the base pattern of `dismiss_prior_context`, and the `context_dilution_override` "ignore everything above" variant. No action needed — they remain active.

---

## [0.1.0] — 2026-02-18

First public release of **Na0S** (formerly AI-Prompt-Injection-Detector).

### Highlights

- Multi-layer prompt injection detection pipeline with 12 defense layers
- Installable Python package: `pip install na0s`
- 1,680+ tests across 53 test files
- MIT licensed

### Layer 0 — Input Intake & Sanitization

- Unicode normalization, invisible character stripping, whitespace canonicalization
- Magic-bytes content-type sniffing, chardet-based encoding detection
- HTML extraction with hidden-content depth limiting
- MIME multipart parsing, OCR text extraction (optional Tesseract)
- Document extraction (PDF, DOCX, XLSX, PPTX)
- SSRF / open-redirect / TOCTOU protection in input loader
- Pre-decode guard for wide encodings (UTF-32, UTF-16)
- Language detection (langdetect) with graceful fallback
- PII screening (emails, phone numbers, SSNs, credit cards)
- Chunked ML analysis for large inputs
- Resource guard with configurable timeout enforcement
- SQLite-backed fingerprint store for deduplication
- Safe regex engine with timeout protection (no ReDoS)
- Property-based fuzz testing via Hypothesis (40 tests)

### Layer 3 — Structural Feature Extraction

- Token-level structural features for ML classification
- Weighted feature scoring

### Layer 4 — ML Classifier

- TF-IDF + Logistic Regression binary classifier
- Pre-trained model weights bundled in package (~424 KB)
- SHA-256 integrity checks for model files

### Layer 5 — Embedding Classifier

- Optional sentence-transformer embedding classifier
- Cosine similarity scoring against known attack patterns

### Layer 6 — Cascade Decision Engine

- Weighted voting across rule, obfuscation, ML, and structural signals
- Configurable confidence thresholds
- `ScanResult` dataclass with technique attribution and metadata

### Layer 7 — LLM Judge & Checker

- Optional LLM-based second opinion (Groq integration)
- JSON parse hardening for LLM responses

### Layer 8 — Positive Validation

- Legitimate use-case allowlisting

### Layer 9 — Output Scanner

- Secret detection (API keys, tokens, credentials)
- Role-break / prompt leakage detection in LLM outputs
- Configurable redaction

### Layer 10 — Canary Tokens

- Canary token generation and detection

### Layer 11 — Supply Chain Integrity

- Safe pickle loading with class allowlisting
- Model file SHA-256 verification

### Rule-Based Detection

- 19 threat categories, 103+ technique IDs (see `THREAT_TAXONOMY.md`)
- Context-aware rule suppression (educational, question, quoting, code, narrative frames)
- Override, system prompt, roleplay, secrecy, exfiltration pattern matching

### Obfuscation Detection

- Shannon entropy analysis (threshold 4.1)
- Casing transition ratio for alternating-case detection
- Punctuation flood detection with structured-data exemption
- Base64/hex/rot13 encoding detection
- Leetspeak and homoglyph normalization

### Testing

- 1,680+ unit and integration tests
- Attack coverage: D1 (instruction override), D3 (structural boundary), D4 (encoding), D5 (unicode evasion), D6 (multilingual), D7 (payload delivery), D8 (context manipulation), E1 (prompt extraction), E2 (reconnaissance), O1 (harmful content), C1 (compliance evasion), P1 (privacy leakage)
- False positive test suite with FingerprintStore isolation
- Cascade integration tests
- Output scanner tests
- Property-based fuzzing (Hypothesis)

### CI/CD

- GitHub Actions: lint, syntax check, test, coverage (Python 3.9–3.12)
- `fail-under=50` coverage gate

### Packaging

- `pyproject.toml` with setuptools backend
- Optional dependency groups: `embedding`, `ocr`, `docs`, `llm`, `all`, `dev`
- Typed package (`py.typed` marker)
- Source layout: `src/na0s/`
