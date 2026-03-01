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
| D7 Payload Delivery | 5 | 0 | Fragmented payloads |
| O1 Harmful Content | 4 | 0 | Output manipulation |
| D3 Structural Boundary | 2 | 0 | Subtle markers |
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

