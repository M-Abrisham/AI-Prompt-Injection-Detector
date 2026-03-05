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
