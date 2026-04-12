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
- **Layer 2: Caesar cipher brute-force** — shifts 1–25 (skip 13) with English dictionary validation
- **Layer 2: Pig Latin detection** — consonant-cluster decoding with 370k-word dictionary disambiguation
- **Layer 2: English dictionary** (`data/english_words.txt`) — 370,105-word dwyl/english-words (Unlicense) for Caesar/Pig Latin validation gates
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
