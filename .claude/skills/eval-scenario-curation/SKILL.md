---
name: eval-scenario-curation
description: >-
  Curates and promotes scenarios into the Na0S F14 evaluation library
  (data/eval/scenarios/) with provenance and decontamination discipline — the
  "SIFT Protocol". Use when adding, editing, importing, or reviewing eval
  scenarios, wiring the promotion gate, or feeding triaged detector failures back
  into the scenario set. Enforces verifiable source/provenance, semantic
  decontamination against training data, paired benign siblings, calibrated
  difficulty, and per-category TPR/FPR + over-refusal metrics.
---

# Eval Scenario Curation — the SIFT Protocol

The F14 scenario library is the regression gate for the L0–L16 detector
ensemble. Its hardest failure mode is **provenance + contamination**, not
authoring volume. "SIFT" here is the fact-checking discipline — **Stop /
Investigate the source / Find better coverage / Trace claims** — applied to every
scenario before it can gate.

Schema of record: [schema.py](../../../src/na0s/eval/scenarios/schema.py)
(`Scenario` dataclass). Loader: [loader.py](../../../src/na0s/eval/scenarios/loader.py).
Scenarios live in `data/eval/scenarios/v0.1/*.yaml`.

> Read the dataclass before editing — do not trust the field list below if the
> code has changed. Fields marked **(PROPOSED)** do **not** exist yet; add them
> with a schema change + a failing test, don't assume they're present.

---

## The SIFT pass — apply to every scenario before promotion

### S — Stop: don't promote on face value
The promotion gate exists for this. A scenario is a candidate until it has passed
the four steps below. Never add directly to a `v0.1/*.yaml` and assume it gates.

### I — Investigate the source (provenance)
- `source` must be one of the real enum values: `manual`, `shade_arena`,
  `harmbench`, `layer16_fixtures`, `llm_generated`, `matrix_composed`,
  `harvest_pipeline`. It names the *bucket*, not the origin.
- **(PROPOSED)** add `provenance` (origin URL / dataset + license + date) and
  `verified_by` / `verified_at`. A `severity: critical` scenario must have a
  human label sign-off — do not trust `llm_generated` labels unreviewed (an
  imported seed set is ~50% mislabeled until proven otherwise).

### F — Find better coverage (don't just collect payloads)
- Tag `attack_category` from the taxonomy and use `compliance_tags`
  (OWASP / MITRE ATLAS / NIST) to find **taxonomy gaps**, not just to label.
- Every attack scenario should ship its benign sibling: set `paired_benign_id`
  to a scenario with `expected_verdict: allowed` and near-identical surface form.
  This is how over-refusal is measured — a blunt over-block then fails the gate.

### T — Trace the claim
- Tie `expected_verdict` back to a cited technique (e.g. ATLAS AML.T0051) in the
  `description` or `compliance_tags`. If you can't cite *why* it should be
  blocked, it isn't ready.
- Confirm `stable_id` (SHA-256 of NFKC-normalized, whitespace-collapsed content)
  is computed — it's the decontamination handle.

---

## Decontamination (the highest-value integrity step)
1. **Exact match** — `stable_id` already blocks verbatim train/test overlap.
2. **Semantic near-dup (GAP today)** — exact match misses *paraphrased* attacks.
   Before promotion, run an embedding-similarity sweep against training data and
   reject candidates with cosine > ~0.9. A paraphrased attack already in training
   is a leak, not a test. **(PROPOSED)** record `decontam_checked_at` +
   `decontam_method` so the gate can refuse un-audited scenarios.

## Difficulty — calibrate, don't assert
`difficulty` (int 100–400) is currently asserted by the author. Derive it from
**how many ensemble layers must fire to catch the payload**: an L0-regex catch is
easy; one that only the LLM judge catches is hard. Auto-flag scenarios that every
layer trivially catches as retirement candidates — they no longer carry signal.

## Metrics the gate must emit (not a single pass-rate)
- **Per-`attack_category` TPR/FPR** — block on a regression *within any category*
  so a D1 gain can't mask a new D8 hole.
- **Over-refusal rate** = false positives on the `paired_benign_id` siblings,
  reported separately.
- For non-deterministic `evaluator.type == llm_judge` scenarios, probe **3×** and
  record consistency — a 2-of-3 catch is a latent regression.
- For `multi_turn`, exploit per-turn `risk_score`: assert *which turn* detection
  is expected by and fail if caught after the harm turn (the "missed an early
  warning sign" case the schema docstring names).

> Note: in v0.1 `evaluator.check` is free-text documentation and is **not
> executed** — "metrics beyond accuracy" are aspirational until the v0.2 executor
> lands. Prioritize that executor before claiming the gate enforces these.

---

## Error-analysis loop (failures → new scenarios)
1. Dump every gate failure **with per-layer scores**, `attack_category`,
   `source`, `difficulty` — not just pass/fail.
2. Bucket each FN/FP into a fixed taxonomy: **obfuscation-evasion** (encoding/
   split bypassed L0–L2) · **semantic-novelty** (technique not in taxonomy) ·
   **multi-turn-late** (caught after the harm turn) · **over-refusal** (FP on a
   benign sibling). **(PROPOSED)** persist this as a `response_class` tag.
3. Cluster within each bucket by embedding similarity — a dense cluster is a
   systematic hole; a singleton may be a labeling fluke.
4. Root-cause to the specific L0–L16 / ML / judge layer that should have fired
   (the per-layer trace from step 1 makes this mechanical).
5. Promote the cluster exemplar **and its benign sibling** back into the library;
   generate a few parameter-randomized variants so one fix generalizes.
6. **Decontaminate before re-adding** (above) — confirm the new scenario isn't
   already in training data.
7. Re-run the per-category gate: confirm it **fails before** the fix and
   **passes after**, and that no other category regressed.

## What F14 already does well (don't reinvent)
Paired benign siblings (`paired_benign_id`), category slicing (`attack_category`),
and first-class multi-turn (per-turn `expected_label` + `risk_score`) are already
ahead of most public benchmarks. The open work is verification rigor: semantic
decontam, provenance/label-audit fields, calibrated difficulty, repeated-trial
consistency, and executing the v0.2 evaluator.
