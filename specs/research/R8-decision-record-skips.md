---
item: R8
title: "Negative decision record / de-scope — Sigstore-keyless + SLSA + OCI provenance + blanket-skops + pickle-scanner-as-CI-gate"
class: RESEARCH item — SUPPLY-CHAIN / GOVERNANCE infrastructure (NOT a prompt-injection attack class)
classification: Governance DOC only — records SKIP / evaluated-&-de-scoped decisions; ships NO source, NO detector, NO test-of-shipping-code
dedup_status: >
  NEW DOC. RE-VERIFIED against THIS worktree (research-items, off `hardening/rag-poison-wiring`,
  base advanced via #476 / 81db256). A "recommended-against" block ALREADY EXISTS in the roadmap but
  the task's cited line range (~1698-1701) has DRIFTED: the live block is the M8 / research-agent-R5
  callout at **ROADMAP_V2.md:1763-1766** ("What R5 explicitly RECOMMENDED AGAINST adding: TEEs/Nitro
  Enclaves; FROST/N-of-M threshold sigs; reproducible-build attestation — incl. the SLSA-3-provenance
  redundancy note at :1766"). R8 does NOT duplicate that block — it MIRRORS its FORMAT and EXTENDS the
  same governance surface to a NEW set of SKIP rows (Sigstore-keyless model-signing, SLSA, OCI/in-toto
  provenance, blanket-skops migration, pickle-SCANNER-as-CI-gate). #476 added the M8 block and the
  hardening-campaign record but added NO standalone negative-decision-record DOC and NO de-scope cross-link
  for the Key-Libraries "adopt" rows (ROADMAP_V2.md:2759 model-signing/Sigstore, :2760 modelscan, :2763
  safetensors) — those still read as aspirational "adopt", un-reconciled. R8 is that reconciliation.
applicable_steps: [1, 2, 3, 7, 8, 9]   # explore, roadmap/refs, root-cause(=the decision rationale), cleanup, roadmap-update, README/SECURITY reconcile — all governance-doc-shaped
na_steps: [4, 5, 6, 10, 11]            # implement-into-pipeline, harvester, tests-of-shipping-code, taxonomy/coverage, GitHub — see N/A justifications
skills: [na0s-review-checklist]
agents: [layer-9-11-auditor, security-research-auditor]
depends_on:
  - Item 11 (specs/hardening/11-ci-security-gate.md) — R8 de-scopes ONLY the pickle-SCANNER half of #11 (modelscan/fickling, §3.2 steps 82-83). The bandit / ruff-S / CodeQL-security-extended / pip-audit dep-audit / no-raw-loader grep-test (§3.2 step 85 + §3.4) SURVIVE and remain the real gate. R8 must land its decision AFTER #11 lands (or alongside) so the surviving controls are concrete to point at.
  - Item 13 §3.4 steps 130-134 (specs/hardening/13-format-migration-skops-safetensors.md:130-134) — the SURVIVING "pin `skops>=0.10,<1` if/when used" guidance. R8 de-scopes BLANKET skops migration (do not rip every pickle to skops as a campaign) but PRESERVES skops as a targeted, parity-gated option for the model-load artifact when Item 13/R6 elects it; R8 explicitly does NOT delete that line.
  - Item 13 / R6 (R6-split-serialize.md) — R8's "blanket-skops SKIP" must not contradict R6's targeted split-serialize fork; R8 records that skops is EVALUATED (kept as an option), not mandated.
  - ROADMAP_V2.md:1763-1766 (M8 / R5 recommended-against block) — R8 mirrors its format; the two blocks must cross-link, not collide.
local_only: true   # NO GitHub — no push / PR at any execution step
---

# R8 — Negative decision record / de-scope (Sigstore-keyless · SLSA · OCI provenance · blanket-skops · pickle-scanner-as-CI-gate)

## 0. One-sentence framing
R8 is a **governance DOC**: it formally records a SKIP / "evaluated & de-scoped" decision for four
supply-chain controls that the roadmap currently lists aspirationally as "adopt" — **Sigstore-keyless
model-signing, SLSA provenance levels, OCI/in-toto artifact provenance, and a blanket sklearn→skops
migration** — plus the **pickle-SCANNER-as-a-CI-gate** sliver of Item 11; it reconciles the stale
"adopt" rows to "evaluated & de-scoped", and it **preserves** the controls that survive
(bandit / ruff-S / CodeQL-security-extended / pip-audit / no-raw-loader grep-test from #11, and the
"pin skops if used" line from Item 13). It ships **no source, no detector, no test of shipping code**.

## Dedup re-verification (against THIS worktree)
- **A recommended-against block already exists** — `ROADMAP_V2.md:1763-1766` (M8, research-agent R5):
  TEEs/Nitro Enclaves, FROST/N-of-M threshold signatures, and reproducible-build attestation are
  already recorded as "RECOMMENDED AGAINST", and :1766 already notes "SLSA-3 provenance of the *runner*
  is fine but mostly redundant with pinned-SHA Actions." **R8 does NOT re-litigate those three rows** —
  it MIRRORS the block's format and adds the **distinct** set below (model-signing/Sigstore-keyless,
  SLSA-as-a-release-gate for the *model artifact* (not the runner), OCI/in-toto provenance, blanket-skops,
  pickle-scanner CI gate). The two blocks cross-link.
- **The "adopt" rows are un-reconciled** — the Key Libraries table still reads (verified, THIS worktree):
  - `ROADMAP_V2.md:2759` `| model-signing | Sigstore signing v1.0 | L11 |`
  - `ROADMAP_V2.md:2760` `| modelscan | Pickle content scanning | L11 |`
  - `ROADMAP_V2.md:2763` `| safetensors | Secure model format | L11 |`
  and the source list at `:2782` (Sigstore Model Signing v1.0) and `:2801` (SLSA v1.1). These read as
  intended adoptions with no "evaluated & de-scoped" annotation. **R8's roadmap edit (Step 8) is to
  annotate these rows, NOT delete them** (they remain valid *research sources*; the verdict changes from
  "adopt" → "evaluated & de-scoped, see R8").
  > **Drift note for the executor:** the task brief cited the adopt rows at `~:2692-2696/2715/2734` and
  > the recommended-against block at `~:1698-1701`; in THIS worktree (#476-advanced) they are at
  > `:2759/2760/2763` and `:1763-1766` respectively. **Re-grep at execution** — do not trust the stale
  > line numbers. Anchor on the literal strings `model-signing | Sigstore`, `modelscan | Pickle content`,
  > `safetensors | Secure model format`, and `RECOMMENDED AGAINST adding`.
- **#11 SCOPE SPLIT (the load-bearing nuance):** de-scope ONLY the **pickle-SCANNER** part of Item 11 —
  i.e. `modelscan`/`fickling` (`specs/hardening/11-ci-security-gate.md` §3.2 step 82-83, README row 11
  "no fickling/modelscan"). The **bandit** (§3.2 step 80), **ruff-S** (step 81), **CodeQL
  security-extended** (§3.3 step 6 / `codeql.yml`), **pip-audit dep-audit** (step 84), and the
  **no-raw-loader grep-test** (`tests/integrity/test_no_raw_loaders.py`, §3.4 steps 7-8) all **STAY**
  and remain Item 11's real, keyless, deterministic gate. R8 must say this explicitly so a future reader
  does not gut all of #11.
- **#476 (81db256) check:** present in this worktree's `git log` (`81db256 docs(roadmap): record
  hardening campaign (8 PRs)…`); it recorded the M8 recommended-against block but added **no** standalone
  negative-decision DOC and **no** de-scope annotation on the Key-Libraries adopt rows. **Dedup status
  unchanged: R8 is NEW.**

---

## The four+one de-scope decisions (the substance of R8)

> Each row records: **decision**, **one-line rationale grounded in THIS worktree's constraints**,
> **what survives instead**. Keyless / subscription-only host (no raw API key — project memory) is the
> dominant constraint and is cited per row where load-bearing.

| # | Control (current "adopt" framing) | Decision | Rationale (Na0S-grounded) | What survives instead |
|---|---|---|---|---|
| 1 | **Sigstore-keyless model-signing** (`model-signing` lib, ROADMAP_V2.md:2759; source :2782) | **SKIP / de-scope** | Sigstore *keyless* signing binds to an OIDC identity at sign time and the Fulcio/Rekor transparency-log path; Na0S ships 4 small `.pkl` inside the package, deploys via PyPI Trusted Publishing, and has a **subscription-only / no-raw-key host** — the keyless flow's CI-identity + network-to-Rekor dependency adds release-pipeline surface for a single-publisher artifact whose tamper-protection is already covered by the SHA-256 `KNOWN_HASHES` pin (`src/na0s/models/__init__.py:26-31`) + per-file sidecars. Marginal defense over the existing pin is low; operational + network surface is non-trivial. | PyPI **Trusted Publishing** (already the release path, ROADMAP_V2.md `publish.yml`) provides provenance at the *package* level; the hardcoded `KNOWN_HASHES` + `safe_load` digest gate provides per-artifact tamper-evidence. |
| 2 | **SLSA provenance levels** (as a *model-artifact* release gate; source ROADMAP_V2.md:2801 SLSA v1.1) | **SKIP / de-scope** (artifact gate) | The roadmap already records (`:1766`) that "SLSA-3 provenance of the *runner* is fine but mostly redundant with pinned-SHA Actions"; R8 extends that to the *model artifact* — a full SLSA build-provenance attestation requires a hermetic, attested builder, which is overkill for a 4-pickle bundle retrained by `auto-retrain.yml`. No regulated-customer requirement is on file (SECURITY.md does not promise SLSA). | Pinned-SHA third-party Actions + `KNOWN_HASHES` + the `auto-retrain` → `deploy_model.py` `_write_sidecar` provenance trail (`.sha256` regenerated on every deploy, F-AR6). |
| 3 | **OCI / in-toto artifact provenance** (container/registry provenance) | **SKIP / de-scope** | Na0S is a **pip-installed defensive SDK**, not a container-distributed service (project positioning memory); there is no OCI registry in the distribution path, so OCI/in-toto provenance has no artifact to attest. Pure speculative plumbing. | Same as #2; the Dockerfile/`docker-compose.yml` are dev/eval-only, not the shipped distribution. |
| 4 | **Blanket sklearn→skops migration** (rip *every* pickle to skops as a campaign) | **De-scope the BLANKET form; KEEP skops as a targeted, parity-gated option** | Item 13 / R6 already own the *durable* fix for the model-load artifact behind a **byte-identical `predict_proba` parity gate** (Item 13 §2 "Parity / no detection drift"). A blanket campaign to skops-ify every pickle (incl. training-only intermediates `(X,y)`, FAISS labels, teacher preds) adds a runtime `trusted=` type-allowlist maintenance liability (Item 13 §2 edge case) and a new core dep for low marginal benefit on artifacts that never load in `scan()`. R8 de-scopes the *campaign*, NOT skops itself. | **Item 13 §3.4 steps 130-134 SURVIVE**: "pin `skops>=0.10,<1` if/when used" — skops stays a targeted, import-guarded, parity-gated option for the **model-load** artifact (or R6's split-serialize fork). Arrays go to `np.savez(allow_pickle=False)` / `scipy.sparse.save_npz` (Item 13 `safe_arrays.py` / R4), not skops. |
| 5 | **Pickle-SCANNER as a CI gate** (modelscan/fickling, ROADMAP_V2.md:2760; Item 11 §3.2 steps 82-83) | **De-scope the SCANNER sliver of #11 ONLY** | modelscan/fickling scan `src/na0s/models/*.pkl` for opcode-level RCE primitives — but Na0S **builds those pickles itself** from a trusted training pipeline and gates them at *load* time via `KNOWN_HASHES` + `safe_load`; an opcode scanner of self-authored, hash-pinned artifacts is low-signal and the tools are unpinned/heavy. **CRITICAL: this de-scopes ONLY the scanner.** | **Item 11 SURVIVES in full minus the scanner**: bandit (§3.2.80), ruff-S (.81), CodeQL security-extended (§3.3), pip-audit dep-audit (.84), and the **no-raw-loader grep-test** `tests/integrity/test_no_raw_loaders.py` (§3.4) — the grep-test is the *real* in-perpetuity defense against a new ungated `pickle.load`/`torch.load`, and it is keyless + deterministic. |

---

## Applicable steps (governance-doc-shaped — verification + reconciliation, not net-new code)

### Step 1 — Explore current rules around R8 (reload `na0s-review-checklist` FIRST)
- **Map every "adopt"/aspirational supply-chain row and every existing de-scope statement** (line refs
  re-verified against THIS worktree; re-grep at execution — they drifted via #476):
  | Surface | THIS-worktree ref | Current framing | R8 target |
  |---|---|---|---|
  | model-signing / Sigstore | `ROADMAP_V2.md:2759`, source `:2782` | "adopt: Sigstore signing v1.0 / L11" | annotate "evaluated & de-scoped — see R8 (keyless-host + single-publisher)" |
  | modelscan | `ROADMAP_V2.md:2760` | "adopt: Pickle content scanning / L11" | annotate "scanner de-scoped as a CI gate — see R8; bandit/ruff-S/CodeQL/grep-test survive" |
  | safetensors | `ROADMAP_V2.md:2763` | "adopt: Secure model format / L11" | LEAVE as-is (safetensors is the array target in Item 13/R4 — NOT de-scoped); note it is the *array* path, not the estimator path |
  | SLSA v1.1 | `ROADMAP_V2.md:2801` (source) | listed source | annotate the M8 block (:1766) already covers the runner; R8 covers the model artifact |
  | existing recommended-against | `ROADMAP_V2.md:1763-1766` | TEEs/FROST/repro-build SKIP | cross-link R8's new block; do NOT duplicate |
  | Item 11 scanner sliver | `specs/hardening/11-ci-security-gate.md` §3.2 step 82-83; README row 11 | "no fickling/modelscan" gap | record scanner de-scoped; SURVIVORS named |
  | "pin skops if used" | `specs/hardening/13-format-migration-skops-safetensors.md:130-134` | surviving guidance | PRESERVE verbatim; cite as the skops-survives anchor |
- **No-hallucination check (na0s-review-checklist §1):** every symbol R8 names is verified to exist —
  `KNOWN_HASHES` (`src/na0s/models/__init__.py:26-31`), `safe_load`/`safe_dump`
  (`src/na0s/integrity/safe_pickle.py`), `tests/integrity/test_no_raw_loaders.py` (Item 11 §3.4 deliverable),
  `deploy_model.py::_write_sidecar` (F-AR6, ROADMAP_V2.md). **R8 introduces NO new API and NO threshold**,
  so na0s-review-checklist §7 (arbitrary security thresholds) is trivially satisfied — there is no
  similarity-cutoff or any number to calibrate (this is a decision record, not a detector).

### Step 2 — Roadmap / Taxonomy / README / Coverage / benchmark for the gap
- **Roadmap:** the gap is the *un-reconciled* adopt rows (:2759/:2760/:2763) + the absence of a single
  consolidated negative-decision section. R8's deliverable lives ENTIRELY in the roadmap + a short
  SECURITY.md note (Steps 8/9). **No new attack-class roadmap line.**
- **Taxonomy / Coverage Matrix:** N/A reference — no attack class (see Step 10 N/A).
- **README / Benchmark:** the only doc surface is "what supply-chain controls Na0S ships vs. consciously
  skips" — a SECURITY.md row, not a README feature claim and not a benchmark number (Step 9).

### Step 3 — Root-cause "implementation" plan = the DECISION RATIONALE + reconciliation (LOCAL)
R8 writes **no load-bearing code**. Its plan is a 4-point governance edit:
1. **Author the consolidated negative-decision section** — add a "Supply-chain controls — evaluated &
   de-scoped (R8)" subsection in ROADMAP_V2.md *adjacent to* the existing M8 recommended-against block
   (`:1763-1766`) OR as a new clearly-titled section, carrying the five-row table above verbatim, each
   row with decision + rationale + survivor. Cross-link the M8 block so the two are discoverable together.
2. **Reconcile the adopt rows** — annotate `:2759` (model-signing/Sigstore) and `:2760` (modelscan) with
   "evaluated & de-scoped — see R8 §<n>"; LEAVE `:2763` (safetensors) as a real adopt (it is the array
   format, owned by Item 13/R4 — explicitly NOT de-scoped). Keep the rows as *research sources*.
3. **Pin the #11 scope split in writing** — in both ROADMAP_V2.md and (optionally) a one-line note at the
   top of `specs/hardening/11-ci-security-gate.md`, record: "Per R8, the modelscan/fickling pickle-SCANNER
   step is de-scoped as a CI gate; bandit + ruff-S + CodeQL security-extended + pip-audit + the
   no-raw-loader grep-test REMAIN the gate." (Do NOT edit #11's code/steps — just annotate the decision.)
4. **Preserve the skops survivor** — reference Item 13:130-134 ("pin `skops>=0.10,<1` if/when used") in
   R8's row #4 so the blanket-skops SKIP can never be misread as "delete skops" — skops stays a targeted,
   parity-gated option. **No threshold introduced** anywhere in R8 (na0s-review-checklist §7 N/A by
   construction).

### Step 7 — Cleanup / refactor
- R8 adds **no files to clean up** (governance doc; all edits are to existing docs). The one hygiene item:
  ensure the new de-scope section and the M8 block (`:1763-1766`) **cross-reference each other** so future
  readers find both negative-decision lists in one hop (avoid a third, divergent "recommended-against"
  list drifting into existence). Confirm `safetensors` is NOT accidentally swept into the SKIP set during
  the edit (it survives as the array target).

### Step 8 — Roadmap update (LOCAL)
- Add the consolidated "evaluated & de-scoped (R8)" section (Step 3.1 table); annotate the three Key-Library
  rows (Step 3.2); record the #11 scope split (Step 3.3). Check off R8 as DONE with the local commit SHA if
  one lands (per the roadmap-todo-sync convention). **Do NOT** mark Item 11 or Item 13 done — R8 only
  annotates their *scope decisions*; their code remains open and owned by their own specs.

### Step 9 — README / SECURITY / Benchmark
- **SECURITY.md:** add ONE concise row to the supply-chain section — "Na0S evaluated and consciously
  de-scoped Sigstore-keyless signing, SLSA/OCI artifact provenance, and blanket-skops migration for the
  bundled model (see ROADMAP_V2 R8); the shipped controls are SHA-256 `KNOWN_HASHES` + `safe_load` digest
  gate + the CI no-raw-loader grep-test + bandit/ruff-S/CodeQL." Plain-English, honest, no over-claim.
- **README:** no change — R8 is not a feature; do not advertise a de-scope as a capability.
- **Benchmark:** **No benchmark/recall change** — R8 is governance, touches no detector, changes no
  `predict_proba`, alters no TPR/FPR. (Same posture as Item 11 §9 "governance, not detection".)

---

## N/A steps (honest justifications)

- **Step 4 — Implement-now & wire into predict.py / cascade.py.**
  N/A — R8 is a decision record; there is NO capability to wire. `predict.py`/`cascade.py` load the
  model via `safe_load` (the survivor control) and reference no Sigstore/SLSA/OCI/skops-blanket surface.
  Wiring a "decision" into the scan path is nonsensical (and would violate na0s-review-checklist §11
  "smoke-first-wire-second" — there is nothing to smoke).
- **Step 5 — Harvester / dataset audit.**
  N/A — R-items are eval-integrity/supply-chain/governance infra, not attack classes; R8 specifically
  records governance verdicts about signing/provenance/serialization controls. No HuggingFace/arXiv/GitHub
  harvest, no F14 scenario, no decontam dataset applies. (Scope note: R1/R2/R3 touch the decontam pipeline;
  R8 does not.)
- **Step 6 — New tests for shipping behavior.**
  N/A — R8 ships no shipping code, so there is no behavior to test. The relevant *surviving* test
  (`tests/integrity/test_no_raw_loaders.py`) is authored and owned by **Item 11 §3.4**, not by R8; R8 only
  records that it survives. A doc-only change needs no pytest. (If the executor edits a `.md` that the
  `check_facts_drift.py` / `check_readme_drift.py` CI gate watches, the smoke step below covers it.)
- **Step 10 — Taxonomy + Coverage Matrix + per-feature thresholds.**
  N/A — a de-scope decision introduces no detectable attack class, no taxonomy code, no scored threshold,
  and no similarity cutoff. R8 records verdicts; it never scores anything. (na0s-review-checklist §7 —
  arbitrary thresholds — N/A by construction: R8 has zero numbers.)
- **Step 11 — PR / GitHub.**
  N/A — LOCAL ONLY per the directive; R8's edits land in ROADMAP_V2.md + SECURITY.md on the working
  branch; no standalone PR. No push / PR / GitHub at any execution step.

---

## Q&A self-check
- **Q1 — Can Na0S handle R8 (run scan/suite)?** N/A as a detector — R8 is a governance record, not a
  detection capability; the only "suite" relevance is the docs-drift gate (`check_facts_drift.py` /
  `check_readme_drift.py`) — run it after the SECURITY.md edit (smoke step below).
- **Q2 — Cleanup done / clutter?** Yes — R8 adds NO new files; it consolidates de-scope decisions into one
  cross-linked section and prevents a third divergent "recommended-against" list from forming.
- **Q3 — Pipeline wiring correct?** N/A — no capability to wire; `predict.py`/`cascade.py` are untouched.
- **Q4 — Tested for code AND use-case?** N/A for code (none ships). Use-case = "a future reader finds the
  de-scope rationale and the survivors in one place" — validated by the cross-link + the consolidated table,
  not by a test.
- **Q5 — Harvester audit.** N/A — governance verdict, not harvested intel.
- **Q6 — Taxonomy + coverage (no dups)?** N/A — no attack class; AND the dedup audit confirms NO duplicate
  R8 doc exists (the M8 block at :1763-1766 is a DIFFERENT, non-overlapping set; R8 cross-links it).
- **Q7 — Does the scorer score R8 right?** N/A — R8 introduces no scorer and no threshold.
- **Q8 — Do predict.py / cascade.py reference R8?** N/A — neither references Sigstore/SLSA/OCI/skops-blanket;
  both load the model via the SURVIVING `safe_load` digest gate, which R8 explicitly preserves (grep
  confirms no `sigstore`/`slsa`/`model_signing` import in `predict.py`/`cascade.py`).
- **Q9 — Harvester agent harvest this type?** N/A — not a harvestable intel type.
- **Q10 — Other correctness check.** (i) Re-grep ALL cited line numbers at execution — :2759/:2760/:2763
  and :1763-1766 drifted via #476; anchor on literal strings. (ii) SCOPE GUARD: de-scope ONLY the
  pickle-scanner of #11 and ONLY blanket-skops of #13 — bandit/ruff-S/CodeQL/grep-test and "pin skops if
  used" MUST be named as survivors in writing. (iii) safetensors (:2763) is NOT de-scoped (array target).
  (iv) Zero thresholds/numbers introduced — confirm na0s-review-checklist §7 is vacuously satisfied.
  (v) Cross-link the new block to the M8 block so the two negative-decision lists are co-discoverable.

---

## Execution preconditions / dependencies
- **Depends-on Item 11** — R8 de-scopes the scanner sliver; land the decision AFTER (or alongside) #11 so
  the survivors (bandit/ruff-S/CodeQL/grep-test) are concrete to point at. Do NOT edit #11's code.
- **Depends-on Item 13 §3.4:130-134** — the "pin `skops>=0.10,<1` if/when used" line MUST survive R8's
  blanket-skops de-scope; cite it verbatim. R8 must not contradict Item 13 / R6's targeted skops option.
- **Depends-on ROADMAP_V2.md:1763-1766** (M8 / R5 recommended-against block) — mirror its format; cross-link;
  never duplicate the TEEs/FROST/repro-build rows.
- **Re-grep gate (BEFORE any edit):** confirm the live line numbers for `model-signing | Sigstore` (was 2759),
  `modelscan | Pickle content` (2760), `safetensors | Secure model format` (2763), and `RECOMMENDED AGAINST
  adding` (1763) — they drifted via #476 and WILL drift again. Anchor edits on literal strings, not line numbers.
- **Env:** `skops`/`safetensors`/`sigstore`/`model-signing`/`modelscan` are NOT installed and (except the
  aspirational rows) NOT in `pyproject.toml` — R8 adds NONE of them (it is a doc; it removes the *pressure*
  to add them). Verify against MAIN (`PYTHONPATH=<worktree>/src`) if any symbol claim is checked, not the
  stale editable install. Work in a git worktree; never branch-switch the primary checkout.
- **Keyless:** the whole R8 rationale rests on the subscription-only / no-raw-key host — keep that the
  central justification; introduce no key requirement.
- **LOCAL ONLY** — no push / PR / GitHub at any step.

## Smoke / suite step (governance-doc-appropriate)
1. After the SECURITY.md / ROADMAP_V2.md edits, run the docs-drift gate so the change does not red-fail CI:
   `python3 scripts/check_facts_drift.py` and `python3 scripts/check_readme_drift.py` (whichever watch the
   edited files) — these are keyless and deterministic.
2. Confirm no source/test changed: `git status --porcelain` shows ONLY `ROADMAP_V2.md` + `SECURITY.md`
   (+ this spec). If any `.py` under `src/`/`tests/` appears, the edit overstepped — revert it.
3. (Optional) `python3 -m pytest tests/ -q --tb=line -k "facts or readme or roadmap"` to catch any doc-gate
   test. No full-suite run is required for a doc-only change, but the docs-gate must pass.

## Definition of done
- [ ] Live line numbers re-grepped (drift from the brief's :2692-2734 / :1698-1701 to :2759/:2760/:2763 /
      :1763-1766 confirmed; edits anchored on literal strings).
- [ ] Consolidated "Supply-chain controls — evaluated & de-scoped (R8)" section added to ROADMAP_V2.md,
      carrying the five-row decision table (Sigstore-keyless · SLSA-artifact · OCI/in-toto · blanket-skops ·
      pickle-scanner), cross-linked to the M8 recommended-against block (:1763-1766).
- [ ] Key-Library rows reconciled: `:2759` (Sigstore) + `:2760` (modelscan) annotated "evaluated &
      de-scoped — see R8"; `:2763` (safetensors) LEFT as a real adopt (array target, NOT de-scoped).
- [ ] #11 scope split recorded in writing: pickle-SCANNER (modelscan/fickling) de-scoped as a CI gate;
      bandit + ruff-S + CodeQL security-extended + pip-audit + `test_no_raw_loaders.py` grep-test named as
      SURVIVORS. No edit to #11's code/steps.
- [ ] Blanket-skops de-scoped but skops PRESERVED as a targeted, parity-gated option — Item 13:130-134
      ("pin skops if used") cited verbatim and NOT deleted.
- [ ] SECURITY.md one-line de-scope row added (honest, no over-claim); README unchanged; benchmark unchanged.
- [ ] Docs-drift gate green (`check_facts_drift.py` / `check_readme_drift.py`); `git status` shows ONLY
      ROADMAP_V2.md + SECURITY.md (+ this spec) touched — zero source/test changes.
- [ ] Zero thresholds/numbers introduced (na0s-review-checklist §7 vacuously satisfied).
- [ ] LOCAL-only throughout — no GitHub at any step; merge-to-main confirmed with the user first.

## Skills to reload at execution (Step 1)
- `na0s-review-checklist` — inject §1 (hallucinated APIs — verify `KNOWN_HASHES`, `safe_load`,
  `test_no_raw_loaders.py`, `deploy_model._write_sidecar` exist), §7 (arbitrary thresholds — confirm R8
  has NONE), §11 (smoke-first-wire-second — N/A, nothing to wire) into any subagent.
- N/A skills: `data-harvesting`, `eval-scenario-curation`, `incident-to-scenario`, `detector-authoring`,
  `eval-harness`, `cron-scheduling` (no harvest / scenario / detector / benchmark / cron surface — R8 is a
  governance doc).

## Agents to assign
- `layer-9-11-auditor` — L11 supply-chain integrity is R8's domain (it owns the model-signing / provenance /
  serialization decision surface); confirm the survivors (`KNOWN_HASHES`, `safe_load`, grep-test) are
  accurately described.
- `security-research-auditor` — validate the de-scope rationale against the actual threat model (single-
  publisher, keyless host, pip-distributed SDK) and confirm no real defense is being dropped (esp. that the
  no-raw-loader grep-test and bandit/ruff-S/CodeQL survivors are preserved).
- (No `github-pr-*` agent — LOCAL only; R8 ships no branch of its own beyond the doc edits.)
