# Attack Spec: Adversarial-ML / Automated Jailbreak (A / GCG) — "PAYOFF"

| Field | Value |
|-------|-------|
| **Attack** | Adversarial-ML / automated jailbreak — GCG suffixes, AutoDAN, PAIR, TAP, HarmBench |
| **Taxonomy** | A (A1.1 GCG suffix), D7.5; OWASP LLM01 / ATLAS T0054 (automated) |
| **Matrix row** | INJ-0023 (was *estimated 48*) → **measured 35%** |
| **Skills** | `eval-harness`, `data-harvesting`, `detector-authoring` |
| **Branch** | `hardening/payoff-gcg-enable` |
| **Status** | 🟡 **PARTIAL — measured + data-pipeline enabled; retrain CI-deferred (blocked locally)** |

GCG-family attacks append an optimizer-found adversarial suffix (or run an
automated jailbreak search — AutoDAN/PAIR/TAP) so an aligned model complies
with a harmful request. They are *fluent or high-perplexity gibberish* that the
keyword/rule layer can't generalize to — the durable fix is a **retrained ML
classifier** with category-A coverage, not more regex.

---

## Measure-first (this worktree, existing probe battery via `scan()`)

| Slice | Measured recall | Benign FP |
|---|---|---|
| **A** (adversarial-ML/GCG, INJ-0023) | **35% (59/168)** | 0/15 |
| **AB** (adversarial benchmarks) | **59% (71/120)** | 2/28 |

The matrix's `estimated 48` overstated A by +13 (same pattern as every other
measure-first in this campaign). A is **not** in the recall harness `per_category`
(D/E/O/P/C1 only) — these numbers come from running the shipped
`scripts/taxonomy/adversarial_ml.py` / `adversarial_benchmarks.py` probes
through `scan()` (defensive measurement, no attack content generated).

## The binding blocker (why the retrain is CI-deferred, not done here)

The durable recall fix is **harvest GCG/AdvBench → decontaminate → retrain the ML
classifier → validate on a held-out gate**. That cannot run in this checkout:

- **No training corpus** — `data/raw`, `data/aggregated`, `data/processed` are
  all empty (0 files). Nothing to retrain on.
- **No ML backbone** — `sentence-transformers` and `torch` do not import;
  the embedding classifier serves the degraded TfidfCentroid fallback.
- **Policy boundary** — *generating* GCG/jailbreak attack content (e.g. authoring
  new suffix patterns) trips the usage-policy violative-content classifier; the
  policy-clear route is **data acquisition + data→weights training**, which is
  exactly what is blocked locally by the two points above.

## To-do (GENERAL Prompt, applicable items only)

- [x] **1-2. Explore + measure.** A=35%/AB=59% measured; A1.1 detector is NOT on
  main (the earlier `adversarial_suffix.py` overfit work never merged); A not in harness.
- [x] **3. Root-cause.** Generalization needs retrained weights with category-A
  coverage; rule-detector route is policy-blocked + overfits the holdout.
- [x] **Enable the data pipeline (policy-clear).** `weekly_harvest.py` now queues
  GCG/AdvBench/AutoDAN/HarmBench/PAIR/TAP across HF/arXiv/GitHub + seeds the
  `walledai/AdvBench` and `walledai/HarmBench` known-dataset IDs, so a data/CI
  environment acquires the category-A corpus for the retrain.
- [x] **10. Matrix honesty.** INJ-0023 `estimated 48` → `measured 35%`.
- [ ] **CI-DEFERRED: harvest → decontaminate → retrain → held-out gate.** Run in
  a data/GPU environment (corpus present, `sentence-transformers`+`torch`
  installed). Gate: A recall ↑ on a **decontaminated** held-out split with the
  pooled benign FPR held ≤ baseline. (`scripts/process_data.py` +
  `scripts/optimize_threshold.py` already exist; they need the corpus.)
- [ ] **Add A to the recall harness `per_category`** so the gain is measured, not estimated.

## Documented residuals
- PAIR/TAP/AutoDAN/LatentBreak (fluent) evade the perplexity gate; only the retrain closes them.
- The local retrain is impossible (corpus + ML backbone absent) — this is an environment limit, not a design gap.

## Q&A verification
1. **Can Na0S catch it?** Partially — **A 35%, AB 59%** (measured). GCG-suffix + benchmark-structured cases; fluent automated jailbreaks evade.
2. **Pipeline wired?** A1.1 tokenization + perplexity (gated) on the input path; no dedicated detector (the overfit one was not merged).
3. **Harvester?** Now **queues** the category-A families (this PR) — acquisition enabled; ingestion→retrain runs in CI.
4. **Scorer / retrain?** The fix is retrained weights, **CI-deferred** (no corpus/backbone locally).
5. **Taxonomy/matrix?** INJ-0023 corrected to measured 35%; A/A1.1 valid.
