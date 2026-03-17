# Na0S Benchmark Roadmap

**Last updated**: 2026-02-28
**Goal**: Produce a credible, publication-grade benchmark for Na0S

---

## What's Done

| Item | Status |
|------|--------|
| CLI (`na0s scan`) | Done — 30 tests |
| `ScanResult.to_dict()` / `.to_json()` | Done |
| Configurable threshold | Done — `scan(text, threshold=0.55)` |
| Latency measurement (`elapsed_ms`) | Done |
| Benchmark harness (`scripts/benchmark.py`) | Done |
| Competitor wrappers (LLM Guard, Prompt Guard 2) | Done — not yet run |
| Regression tests | Done — 45 golden tests (43 pass, 2 xfail) |
| Holdout datasets generated | Done — 502 safe, 200 malicious, 574 adversarial |
| Makefile | Done — 9 targets |
| Dockerfile | Done — multi-stage, non-root |
| PyPI publish workflow | Done — OIDC trusted publishers |
| README updated | Done — CLI, detection capabilities |

---

## Phase 1 — CRITICAL (No credibility without these)

- [ ] **BM-1**: Download and run on real datasets
  - `deepset/prompt-injections` (662 samples) from HuggingFace
  - `tatsu-lab/alpaca` (52K benign) for negative samples
  - `databricks/dolly-15k` (15K human-written benign) for negative samples
  - Run `scripts/benchmark.py` against each

- [ ] **BM-2**: Run benchmark on our own holdout datasets
  - `data/holdout/safe_holdout.jsonl` (502 samples) — exists, never benchmarked
  - `data/holdout/malicious_holdout.jsonl` (200 samples) — exists, never benchmarked
  - `data/benchmark/adversarial_evasion.jsonl` (574 samples) — exists, never benchmarked

- [ ] **BM-3**: Competitor head-to-head comparison
  - Run LLM Guard on same datasets via `scripts/wrappers/llm_guard.py`
  - Run Prompt Guard 2 on same datasets via `scripts/wrappers/prompt_guard.py`
  - Side-by-side table: F1, Recall, FPR, Latency
  - Use a third-party benchmark (not our own) for credibility

- [ ] **BM-4**: Per-attack-category breakdown
  - Report recall by: D1, D2, D3, D4, D5, D6, D8, E1, E2, C1, P1
  - Not just aggregate F1 — show strengths and weaknesses honestly

- [ ] **BM-5**: Over-defense / false positive stress test
  - Create NotInject-style dataset (339+ benign prompts with trigger words: "ignore", "forget", "system prompt")
  - Report "over-defense accuracy" as a separate metric
  - Current SOTA detectors drop to ~60% on these

- [ ] **BM-6**: Report "Recall @ fixed FPR"
  - Recall @ 1% FPR and Recall @ 0.1% FPR
  - This is the #1 metric enterprises look at
  - Meta Prompt Guard 2 uses this as their headline number

---

## Phase 2 — HIGH PRIORITY (Serious vs amateur)

- [ ] **BM-7**: ROC curve and PR curve
  - Full curves, not single-threshold numbers
  - Overlay Na0S vs competitors on same plot
  - PR curve more informative for imbalanced data

- [ ] **BM-8**: Confusion matrix
  - At threshold=0.55 and at optimal F1 threshold

- [ ] **BM-9**: Threshold sweep
  - Test at [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
  - Table + plot showing F1, precision, recall, FPR at each

- [ ] **BM-10**: Multilingual evaluation
  - Test on non-English prompts (Meta tests 8 languages)
  - Use OpenAssistant/oasst1 (161K messages, 35 languages) for benign
  - 40 known gaps in D6 — quantify honestly

- [ ] **BM-11**: Error analysis
  - 5-10 false positive examples with explanations
  - 5-10 false negative examples with explanations
  - Categorize failure modes

- [ ] **BM-12**: Limitations section
  - Map to OWASP LLM01:2025
  - Acknowledge untested attacks (GCG/AutoDAN/PAIR, agent/tool abuse, indirect injection)

---

## Phase 3 — PUBLICATION-GRADE

- [ ] **BM-13**: LODO evaluation (Leave-One-Dataset-Out)
  - Train on N-1 datasets, test on held-out one
  - Standard evaluation inflates AUC by 8.4% ("When Benchmarks Lie")

- [ ] **BM-14**: Large real-world datasets
  - Lakera Gandalf (279K real attacks from DEF CON)
  - Lakera Mosscap (130K attacks from DEF CON 31)
  - TensorTrust (126K+ human-generated attacks + defenses)
  - BIPIA (86K indirect injection — Microsoft)
  - SaTML CTF (137K multi-turn attack conversations)
  - JailbreakBench (100 curated jailbreak behaviors)
  - AdvBench (520 harmful behaviors — GCG/AutoDAN standard)

- [ ] **BM-15**: Adversarial robustness
  - Generate GCG adversarial suffixes
  - Generate AutoDAN semantically-meaningful jailbreaks
  - Generate PAIR black-box attacks
  - Per-evasion-type detection rates

- [ ] **BM-16**: Latency under load
  - Throughput (requests/sec at 1, 10, 50, 100 concurrent)
  - Memory footprint (RSS idle vs under load)
  - Compare: Na0S (~10ms) vs LLM Guard (~1s) vs Prompt Guard 2 (~92ms)

- [ ] **BM-17**: Map technique tags to MITRE ATLAS
  - Map D1-D8, E1-E2 to MITRE ATT&CK/ATLAS technique IDs

- [ ] **BM-18**: Version comparison tracking
  - Show v0.1 vs v0.2 vs v0.3 side by side as we improve

---

## Phase 4 — IMPROVEMENTS TO EXISTING COMPONENTS

- [ ] **IMP-1**: Rewrite BENCHMARK_RESULTS.md with real data
  - Executive summary, methodology, reproducibility
  - Follow Meta Prompt Guard 2 model card format

- [ ] **IMP-2**: Upgrade benchmark harness
  - `--threshold-sweep`, `--output-roc`, `--per-category`, `--recall-at-fpr`
  - Confidence interval computation

- [ ] **IMP-3**: Create `scripts/benchmark_report.py`
  - Auto-generate results from benchmark JSON
  - Generate ROC/PR/confusion matrix PNGs
  - One command: `make bench-full`

- [ ] **IMP-4**: Expand regression tests to 100+
  - Add indirect injection, multilingual, hard-negative tests
  - Fix the 2 xfail tests

- [ ] **IMP-5**: Expand holdout datasets
  - Safe: 502 → 2,000+ samples
  - Malicious: 200 → 500+ samples
  - Add hard-negatives, indirect injection, multilingual

- [ ] **IMP-6**: More competitor wrappers
  - Deepset DeBERTa (most-deployed open-source)
  - LlamaGuard 3
  - Simple regex baseline (floor)

- [ ] **IMP-7**: Build Layer 5 embedding model
  - Train or download MiniLM model
  - Create `model_embedding.pkl`
  - Re-benchmark L4+L5 ensemble vs L4-only

---

## Known Detection Gaps

129 `@expectedFailure` tests document what Na0S currently misses (down from 152):

| Category | Count | Fixed | Nature |
|----------|-------|-------|--------|
| D6 Multilingual | 40 | **-11** | English-only ML model (multilingual_handler.py added) |
| C1 Compliance Evasion | 21 | **-1** | Fictional framing (fictional_frame_detector.py added) |
| E1 Prompt Extraction | 16 | **-10** | Indirect extraction (extraction_detector.py added) |
| E2 Reconnaissance | 16 | 0 | Capability probing |
| P1 Privacy Leakage | 13 | 0 | Inference attacks |
| FP (False Positives) | 11 | 0 | Security docs flagged |
| D4 Obfuscation | 9 | 0 | Combined encoding + structural |
| D8 Context Manipulation | 6 | 0 | Document overflow |
| D1 Instruction Override | 5 | 0 | Subtle paraphrased overrides |
| D7 Payload Delivery | 5 | **new** | Fragmented payloads (payload_assembly_detector.py added) |
| O1 Harmful Content | 4 | **new** | Output manipulation (harmful_intent_detector.py added) |
| D3 Structural Boundary | 2 | **new** | Subtle markers (semantic_system_marker rule added) |
| D5 Unicode Evasion | 1 | 0 | Edge case |
| **Adversarial ML** | **0** | 0 | GCG, AutoDAN, PAIR — untested |
| **Agent/Tool Abuse** | **0** | 0 | Tool-call injection — untested |

### Recent Detection Improvements (2026-02-28)

Three new detection modules were added, closing 22+ known gaps:

1. **multilingual_handler.py** — D6 multilingual injection patterns for 20+ languages
   (French, Spanish, German, Chinese, Japanese, Arabic, Korean, Hindi, Russian,
   Portuguese, Italian, Turkish, Dutch, Polish, Vietnamese, Thai, Indonesian,
   Swedish, Czech, Hebrew, and romanized transliterations)

2. **fictional_frame_detector.py** — C1 compliance evasion via fictional/hypothetical/
   academic/emotional/authority framing with inner attack extraction

3. **extraction_detector.py** — E1 indirect system prompt extraction via completion
   tricks (E1.3), translation tricks (E1.4), encoding tricks (E1.5), summarization
   tricks (E1.6), reference manipulation, and constraint probing

### D7/O1/D3 Detection Improvements (2026-02-28)

Three additional modules address payload delivery, harmful content, and structural boundary gaps:

4. **payload_assembly_detector.py** — D7 fragmented payload detection:
   token-split assembly (D7.1), code-block weaponization (D7.3),
   comment/metadata payload hiding (D7.4), cross-encoding fragment
   assembly (D7.5), multi-turn stub (D7.2, ready for ConversationSecurityMonitor)

5. **harmful_intent_detector.py** — O1 harmful intent + injection combination:
   CSAM always-flag (O1.2), violence/social-eng/disinfo only when combined
   with injection techniques. Na0S-scoped: injection+harmful detection,
   not general content moderation.

6. **semantic_system_marker** rule (D3.6) — natural-language fake system
   boundary detection: authority+boundary framing, pseudo-official headers,
   supersession language. PL2, context-suppressible.

---

# Na0S Benchmark Sprint — Clean Roadmap

**Generated**: 2026-02-27 | **Last Cleaned**: 2026-03-03
**Goal**: Ship Na0S v0.2.0 with public benchmark numbers, CLI, and pip-installable package
**Pipeline Under Benchmark**: `predict.scan()` (L0 + L1 + L2 + L3 + L4 weighted composite)

---

## DONE — Completed Tasks (39/53)

### Core Engine (CP-1 to CP-5)

- [x] **CP-1**: `ScanResult.to_dict()` / `.to_json()` — 19 tests (2026-02-27)
- [x] **CP-2**: `OutputScanResult.to_dict()` / `.to_json()` (2026-02-27)
- [x] **CP-3**: Configurable `threshold=0.55` on `scan()` / `classify_prompt()` — 15 tests (2026-02-27)
- [x] **CP-4**: `elapsed_ms` field on `ScanResult` with wall-clock measurement — 8 tests (2026-02-27)
- [x] **CP-5**: Fixed `evaluate_probes.py` JSON bug (`_project_root` undefined) (2026-02-27)

### Benchmark Harness & Results

- [x] **CP-6**: `scripts/benchmark.py` — unified harness (JSONL in, metrics out, supports `--tool`, `--threshold`, `--max-samples`) (2026-02-27)
- [x] Baseline benchmark on `test_sample.jsonl` — F1=1.0, p50=9.93ms (2026-02-28)
- [x] `BENCHMARK_RESULTS.md` — 10-section report: F1=0.84 at t=0.40, FPR 0.5-5%, per-technique recall, evasion resistance, avg latency ~33ms (2026-02-28)
- [x] Threshold sweep at [0.40–0.65] — `scripts/threshold_sweep.py`, 27 tests. Best F1=0.97 at t=0.40 (2026-02-28)
- [x] Per-technique-tag analysis — `scripts/technique_analysis.py`, 24 tests. D1 recall=70% (2026-02-28)
- [x] Holdout split benchmark run — results in `benchmarks/results/` (2026-02-28)
- [x] `tests/test_benchmark_regression.py` — 45 golden tests (20 mal + 20 safe + 5 tags), 43 pass + 2 xfail (2026-02-28)
- [x] `make bench-fast` in CI — non-blocking, Python 3.12 only (2026-02-28)

### Competitor Wrappers

- [x] **CP-7**: `scripts/wrappers/base.py` — CompetitorWrapper ABC (2026-02-27)
- [x] **CP-8**: `scripts/wrappers/llm_guard.py` — lazy import + singleton (2026-02-27)
- [x] **CP-9**: `scripts/wrappers/prompt_guard.py` — score flipping for consistent semantics (2026-02-27)

### CLI & Packaging

- [x] **CP-10**: `src/na0s/cli.py` — `na0s scan`, `na0s version`, JSON/CSV/text output, exit codes 0-3 — 30 tests (2026-02-27)
- [x] CLI registered in `pyproject.toml` (2026-02-27)
- [x] `tests/test_cli.py` — 30 tests across 14 classes (2026-02-27)
- [x] **CP-12**: `Makefile` — 9 targets (help, install, test, lint, bench, bench-fast, build, clean, publish) (2026-02-27)
- [x] **CP-13**: `requirements-benchmark.txt` — 78 pinned packages (2026-02-27)
- [x] `.github/workflows/publish.yml` — OIDC trusted publishers (2026-02-28)
- [x] `README.md` updated — install, quick-start, benchmark table, CLI usage, 15-layer summary (2026-02-28)

### Datasets

- [x] **CP-11**: `scripts/download_datasets.py` — deepset, alpaca (2000), dolly (2000) — 40 tests (2026-02-27)
- [x] `data/holdout/safe_holdout.jsonl` — 502 samples (instructional, code, support, creative, technical) (2026-02-28)
- [x] `data/holdout/malicious_holdout.jsonl` — 200 samples (D1:40, D2:30, D3:30, E1:25, D4:25, D5:25, D8:25) (2026-02-28)
- [x] `data/benchmark/adversarial_evasion.jsonl` — 574 samples (82 prompts x 7 evasion types) (2026-02-28)
- [x] `data/holdout/README.md` — provenance, license, split methodology (2026-02-28)
- [x] `Dockerfile` — multi-stage build, non-root user, `.dockerignore` — 39 tests (2026-02-27)

### Gap Closure Sprint (155 → 128 xfails, 27 flipped)

- [x] **Track A**: `recon_detector.py` — E2.1-E2.5, 39 tests, 2 xfails flipped (2026-02-28)
- [x] **Track B**: `privacy_probe_detector.py` — P1.1-P1.6, 38 tests, 3 xfails flipped (2026-02-28)
- [x] **Track C**: FP Reduction — ML zone cap + `safe_content.py` + content-type entropy, 26 tests, 4 xfails flipped (2026-02-28)
- [x] **Track D**: D4 Combined Obfuscation — encoding chain scoring, 11 tests, 4 xfails flipped (2026-02-28)
- [x] **Track E**: D8 Context Manipulation — `context_manipulation_detector.py`, 20 tests, 1 xfail flipped (2026-02-28)
- [x] **Track F**: D1 Subtle Overrides — `subtle_override_rules.py` (4 rules), 40 tests (2026-02-28)
- [x] Pipeline integration — 81 rules, 0 duplicates, 10 context-suppressible additions (2026-02-28)
- [x] Security audit — all `safe_compile()`, 50K input caps, bounded quantifiers (2026-02-28)
- [x] Signal boosting + Caesar brute-force + Pig Latin detection — 13 xfails flipped (2026-02-27)

---

## TODO — Remaining Tasks (14)

### 1. Benchmark Execution

- [ ] **Add PINT adapter** to `benchmark.py` (handle PINT label format if dataset available)
- [ ] **Identify missed attack patterns** — if recall < 80% on any category, check if existing layers should fire
- [ ] **Document threshold changes** with before/after evidence (no overfitting to tune split)

### 2. Dataset Gaps

- [ ] **Create 70/30 stratified split** — tune vs holdout by label + category
- [ ] **Download `deepset_pi.jsonl`** — benchmark dataset not yet on disk
- [ ] **Download `benign_alpaca.jsonl`** — benchmark dataset not yet on disk
- [ ] **Download `benign_dolly.jsonl`** — benchmark dataset not yet on disk

### 3. Release & Packaging

- [ ] **Prepare PyPI release** — bump to 0.2.0 (`pyproject.toml` + `_version.py`), verify metadata, `python -m build`, `twine check`, test install in clean venv
- [ ] **Final integration test** — `pip install dist/*.whl` → `na0s scan "test"` → correct JSON output
- [ ] **Verify Dockerfile** builds and runs benchmark end-to-end
- [ ] **Tag `v0.2.0`** and publish to TestPyPI (production after manual review)

### 4. Quality Gates

- [ ] **Raise coverage gate** — 50% → 60% (immediate), target 70% post-sprint
- [ ] **Create `scripts/benchmark_report.py`** — automated report generation (file not yet created)

### 5. Integration

- [ ] **Cross-workstream sync** — validate datasets work with benchmark harness, merge all work

---

## Current Benchmark Numbers

| Metric | Value | Notes |
|--------|-------|-------|
| F1 (t=0.40) | 0.84 | Best operating point |
| F1 (t=0.55) | 0.76 | Default threshold |
| FPR | 0.5-5% | Varies by dataset |
| Warm Latency | ~33ms avg | CPU-only, Apple Silicon |
| Cold Start | ~3.5s | Model loading from .pkl |
| Regression Tests | 43/45 pass | 2 xfail (known gaps) |
| Expected Failures | 128 | Down from 155 (27 closed) |
| Total Tests | 4,901 | 0 failures |

## Known Detection Gaps (128 xfails)

| Category | Count | Nature |
|----------|-------|--------|
| D6 Multilingual | 40 | 0 non-English training data |
| C1 Compliance Evasion | 21 | Fictional framing |
| E1 Prompt Extraction | 17 | Indirect extraction |
| E2 Reconnaissance | 14 | Capability probing |
| P1 Privacy Leakage | 11 | Inference attacks |
| FP (False Positives) | 7 | Educational content flagged |
| D4 Obfuscation | 6 | Combined encoding |
| D1/D3/D5/D7/D8/O1 | 12 | Various edge cases |
| A Adversarial ML | 0 tests | GCG/AutoDAN/PAIR untested |
| T Agent/Tool Abuse | 0 tests | Tool-call injection untested |
