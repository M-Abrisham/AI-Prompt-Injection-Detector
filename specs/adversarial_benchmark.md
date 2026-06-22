# Attack Spec: Adversarial Benchmark / Automated Jailbreak Suites (AB, A/INJ-0023)

| Field | Value |
|-------|-------|
| **Attack** | Adversarial benchmark cross-validation + automated jailbreak suites (HarmBench / JailbreakBench / TensorTrust / AdvBench / GCG-PAIR-TAP) |
| **Taxonomy** | `AB` (Adversarial Benchmarks, 12 techniques AB1.1–AB4.3) + `A` (Adversarial ML, A1.1–A1.5). Both tag `owasp-llm:2025:llm01`. |
| **Matrix row** | **INJ-0023** = category **A** (Adversarial-ML / automated jailbreak GCG/PAIR/TAP), estimated **48**, PARTIAL. **`AB` has NO matrix row** — it is a Layer-12 cross-validation probe, not an INJ class. |
| **Skills** | `eval-harness`, `detector-failure-analysis`, `na0s-review-checklist`, `data-harvesting` |
| **Branch** | TBD |
| **Status** | 🔍 **SCOPED** (2026-06-22, read-only audit) — partial detection, **unmeasured** (neither A nor AB is in the recall harness) |

"Adversarial benchmark suites" are not one attack — they are *curated test sets* of other
attack classes drawn from public benchmarks. Na0S models them two ways:

- **`AB` probe** (`scripts/taxonomy/adversarial_benchmarks.py`, 148 original samples
  "inspired by" HarmBench/JailbreakBench/TensorTrust/public disclosures). 12 techniques:
  AB1 HarmBench-style (direct-harm / template / GCG-suffix), AB2 JailbreakBench-style
  (DAN / roleplay / translation), AB3 TensorTrust-style (prompt-extraction / override /
  delimiter-escape), AB4 real-world (indirect-web / multi-step / context-flood).
- **`A` probe** (`scripts/taxonomy/adversarial_ml.py`): the *mechanism* family —
  A1.1 GCG-suffix, A1.2 AutoDAN, A1.3 PAIR, A1.4 TAP, A1.5 model-inversion.

The two overlap by design (AB1.3 ≈ A1.1; AB2.1 ≈ D2.1; AB3.1 ≈ E1.1; AB4.3 ≈ D8). The
defensive signal is whatever the *underlying* class triggers; "AB" itself is an eval-set
label, not a detector target.

---

## To-do list (GENERAL Prompt, per item)

- [x] **1. Explore current rules + find gaps/edge-cases.** AB samples decompose into
  classes Na0S already detects via structure (override / DAN / delimiter / chat-template
  tokens / hidden-HTML). **Empirical gap:** the *harmful-content-request* slice
  (AB1.1 direct harm, AB2.2 fiction-wrapped roleplay, AB4.2 incremental "educational"
  escalation) carries **no injection markers** and is MISSED — because Na0S is a
  prompt-injection detector, not a content-harm/O1 classifier. Edge cases: GCG gibberish
  suffixes (A1.1) lean on perplexity (gated off by default); fluent PAIR/TAP/AutoDAN
  (A1.3/A1.4/A1.2) evade perplexity entirely.
- [x] **2. Read roadmap/taxonomy/matrix/source.** taxonomy.yaml: `AB` (L690-719) and `A`
  (L280-295) both exist; AB has 12 techniques + benign counterparts. COVERAGE_MATRIX:
  **only INJ-0023 (A)**, no AB row. ROADMAP_V2: A3 (HarmBench/WildJailbreak export
  adapters, P1), HarmBench/AutoDAN/AdvBench dataset-ingest TODOs (L1522-1525, open),
  R2D2-overfit warning (L1557, M3). **Stale/missing:** the brief's `detectors/
  adversarial_suffix.py` **does not exist** (it is `adversarial_ml.py` + L5
  `generate_gcg_samples.py`); brief conflated AB↔A. `eval/probes` path is wrong — probes
  live in `scripts/taxonomy/`.
- [x] **3. Root-cause plan.** Two distinct root causes: **(R1)** AB1.1/AB2.2/AB4.2 miss
  because they are content-harm not injection — closing this needs an *output/content*
  signal (O1 `harmful_intent.py`), not an injection rule, and carries the heaviest FP
  risk. **(R2)** A1.1 GCG gibberish relies on `ml/perplexity.py`, which is **gated off by
  default** in `predict.py` (L1353-1365) — so raw-gibberish GCG is undetected on the
  default path unless the suffix also trips override/structural rules (the in-the-wild
  ones usually do — see empirical AB1.3 = 1.0). **Measurement root cause:** neither A nor
  AB is wired into the recall harness, so the 48 estimate is unverified.
- [ ] **4. Implement + wire (parity).** N/A for this audit (read-only). *Recommended* (no
  new detector first): (a) wire `A` + `AB` into `gen_all_datasets.py`/`technique_analysis`
  so the 48 estimate becomes measured before any code change (measure-first); (b) only if
  measured recall is low on the *injection-structured* AB slice, harden — do NOT add a
  content-harm classifier to chase AB1.1/AB2.2 (FP-unsafe, out of injection scope).
- [ ] **5. Datasets → isolated tests.** `weekly_harvest.py` discovers HF/arXiv/GitHub by
  `jailbreak`/`prompt injection` keywords (would surface HarmBench/JailbreakBench-named
  repos) but does **not** ingest AdvBench/HarmBench *behaviors*; `threat_intel/
  jailbreakbench_sync.py` only diffs dataset *listings*. The canonical AdvBench
  `harmful_behaviors.csv` ingest is an **open ROADMAP TODO** (L1522). Real datasets are
  available (MIT/public) — ingest is gated behind the A3 schema work.
- [ ] **6. Test cases (code + use-case).** AB probe has per-probe validation via
  `tests/test_generate_taxonomy_samples.py` (asserts `_technique_to_category("AB1.1")=="AB"`)
  + `test_l12_buffs.py`. **No scan()-level recall test** exists for AB or A.
- [ ] **7. File/dir cleanup + refactor.** N/A — read-only. Note: AB↔A overlap is
  intentional (cross-validation), not a duplicate to merge.
- [ ] **8. Update roadmap.** N/A this audit. If measured: upgrade INJ-0023 basis
  `estimated`→`measured`; consider an explicit AB cross-validation note in the matrix.
- [ ] **9. README/benchmark.** N/A until measured — AB is a cross-validation eval set, not
  a customer-facing INJ class; do not publish a recall number that the harness hasn't
  produced.
- [ ] **10. Taxonomy + matrix + threshold.** AB rows in taxonomy.yaml are correct and
  match the probe. **No new taxonomy code.** Flag: INJ-0023 carries an **arbitrary `48`
  estimate** with no measurement; perplexity gate threshold `PERPLEXITY_THRESHOLD`
  (`ml/perplexity.py`) is off-by-default and unvalidated against GCG recall.
- [ ] **11. Open PR (held-out gate).** N/A this audit. Any future PR must run the held-out
  gate AND guard against R2D2-style overfit (ROADMAP M3 / arXiv 2402.04249): a fix that
  memorizes the 148 AB strings without generalizing is a regression dressed as progress.

---

## FP-safety (the binding constraint)

- The AB probe **ships 28 benign counterparts** (`AB*_benign`: "I'm studying prompt
  injection defenses for my thesis…", "Our red team needs to understand adversarial
  suffix attacks…", "compare HarmBench/JailbreakBench academically…"). These are the
  exact NotInject-style hard negatives. **Empirically, the benign "I am a security
  researcher at Anthropic…" sample scores 0.132 → ALLOWED** — good, the current
  pipeline already avoids the obvious over-block.
- The dangerous fix is chasing the AB1.1/AB2.2/AB4.2 misses with a harmful-keyword or
  content classifier: "synthesize methamphetamine", "make explosives", "phishing email"
  appear in *both* the attack and the benign-research samples → high FP risk on the
  legitimate security-research / CTF / blue-team traffic that is Na0S's customer base.
  Any such signal MUST be measured against the 28 benign counterparts first.

## Documented residuals (left uncovered)

- **Direct harmful-content requests with no injection structure** (AB1.1, AB2.2 fictional
  framing, AB4.2 incremental "educational" escalation): MISSED on the default path
  (0.02–0.15). Out of injection scope; belongs to the O1/content-safety surface, not this
  detector. Honest residual, not a quick fix.
- **Raw GCG gibberish suffix (A1.1)** without a co-occurring override/structural cue:
  relies on perplexity, **gated off by default** → undetected unless the prefix request
  trips another rule (in practice it usually does).
- **Fluent automated jailbreaks (A1.2 AutoDAN / A1.3 PAIR / A1.4 TAP)**: by construction
  read as natural language → evade perplexity and most structural rules; semantic-only.
- **Measurement gap**: INJ-0023 `48` is an unverified estimate; AB has no number at all.

## Q&A verification

1. **Can Na0S catch it?** **Partial.** Injection-structured AB samples BLOCK (AB1.2
   maintenance-mode 1.0, AB1.3 GCG-suffix 1.0, AB2.1 DAN 1.0, AB2.3 translation 0.91,
   AB3.1 extraction 0.83, AB3.2 override 1.0, AB3.3 chat-template 1.0, AB4.1 hidden-HTML
   0.95). Pure-content-harm AB samples MISS (AB1.1 0.08–0.15, AB2.2 0.14, AB4.2 0.02,
   security-researcher benign 0.13). Verified via `scan()` on the TF-IDF fallback
   classifier (live sentence-transformer unavailable; uncalibrated 0.55 threshold —
   numbers will shift with the real model).
2. **Cleanup done?** N/A — read-only audit; no duplicate to merge (AB↔A overlap is
   intentional cross-validation).
3. **Pipeline wired?** Underlying classes are (override/structural/HTML rules + ML in
   `predict.py`/`cascade.py`). **AB-as-a-class is NOT wired** — and shouldn't be; it's an
   eval label. A1.1 perplexity path is wired but **off by default**.
4. **Tested (code + use-case)?** Probe-generation tests yes
   (`test_generate_taxonomy_samples.py`, `test_l12_buffs.py`); **no scan()-recall test**
   for AB or A.
5. **Harvester?** Discovers jailbreak/PI datasets by keyword (would find HarmBench/JBB
   repos) but does **not** ingest AdvBench/HarmBench *behaviors*; `jailbreakbench_sync.py`
   diffs listings only. AdvBench `harmful_behaviors.csv` ingest is an open ROADMAP TODO.
   Datasets are public/available.
6. **Taxonomy/matrix match, no dupes?** taxonomy.yaml `AB`+`A` correct and internally
   consistent. **Mismatch:** COVERAGE_MATRIX has **no AB row** (only INJ-0023=A) — this is
   defensible (AB is cross-validation, not a class) but should be noted explicitly so the
   148-sample probe isn't mistaken for an unmeasured detector gap.
7. **Scorer correct?** For injection-structured AB samples, yes (override/structural floor
   to 1.0). For GCG gibberish, perplexity contribution is **off by default**; threshold
   unvalidated.
8. **predict.py/cascade.py refs?** No literal `AB`/`adversarial_benchmark` ref. A1.1 is
   referenced via `perplexity_score` (predict.py L257-365, L1552) and the
   `tokenization_spike→A1.1`/`timeout_*→A1.1` rule→technique map (L1983-2020).
9. **Harvester agent harvests it?** Partially — keyword discovery yes, behavior ingest no;
   the `data-harvesting` skill + A3 export-adapter work (ROADMAP P1) is the path to true
   HarmBench/AdvBench interop.
