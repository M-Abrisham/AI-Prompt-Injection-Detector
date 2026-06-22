---
item: 18
title: "L11 Open-Security-Hardening roadmap subsection + fix 24/24 count + purge stale specs/memory refs"
priority_tier: P2 (DOC-only governance/accuracy; no runtime behavior, no attacker precondition — but blocks honest tracking of items 1-12)
class: DOC item — roadmap/spec/memory accuracy (NOT a prompt-injection attack class, NOT a supply-chain code change)
depends_on: []   # SOFT-depends on items 1-12 for FINAL check-off SHAs; the subsection + count-fix + stale-ref purge can land standalone with items 1-12 listed as open
applicable_steps: [1, 2, 3, 6, 7, 8, 9]   # 9 only if a README number actually changes (it does not today)
na_steps: [4, 5, 10, 11]                   # see per-step N/A justifications
# Note: orchestration "step 11 = open a PR" IS applicable (renumbered as Process below); the template "step 10/11" attack-taxonomy/predict-ref family is what is N/A.
---

# Item 18 — L11 "Open-Security-Hardening" roadmap subsection + fix the `24/24 (COMPLETE)` count + purge stale `specs/`/`MEMORY` refs

> **DOC-ONLY.** The single deliverable is edits to Markdown/YAML doc files
> (`ROADMAP_V2.md`, `specs/01-import-linter-architecture.md`, and the
> `MEMORY.md` / agent-memory notes). **No `src/` or `tests/` file is touched.**
> Therefore most "feature" steps (harvester, taxonomy, scorer, pipeline wiring,
> new tests) are N/A — but the doc claims MUST be verified against live source
> with `file:line` citations, exactly like a code review (na0s-review-checklist:
> no hallucinated APIs, confirm imports/wiring exist, no arbitrary numbers).

---

## 0. Root cause (confirmed against source — line numbers verified 2026-06-22)

Three independent doc-accuracy defects, all in the Layer-11 / supply-chain area:

1. **L11 is mislabeled `24/24 (COMPLETE)`** at `ROADMAP_V2.md:124` (progress
   table) and `ROADMAP_V2.md:1104` (`## Layer 11: Supply Chain Integrity —
   Tasks: 24/24 (COMPLETE)`). That status is contradicted by **twelve open
   supply-chain hardening specs** (`specs/hardening/01..12-*.md`), every one of
   which is an L11 concern (KNOWN_HASHES drop, bare-pickle fallback, HF revision
   pin, TOCTOU/allowlist unpickler, raw-loader→`safe_load` routing, fail-closed
   loaders, sidecar-resolution downgrade, L11 adversarial stress tests, sidecar
   key validation, env model-id allowlist, CI security gate, worm 3-tier
   integrity). L11's own TODO list already carries two of these as open bullets
   (`ROADMAP_V2.md:1183` misfiled modules, `:1184` `scripts/safe_yaml.py`
   library-in-scripts), so the "COMPLETE" tag is internally inconsistent even
   before the new specs. **Root cause: the `24/24 (COMPLETE)` count was frozen
   when the original 24 audit items closed and never reopened when the
   supply-chain hardening audit (items 1-12) surfaced new, real, open work.**

2. **`specs/01-import-linter-architecture.md:16` is stale.** It lists as a
   still-to-do "fold-in fix": *"dead `rag_poison_weight` fold
   (`predict.py:1201`, computed-then-discarded) + missing `cascade.py`
   `_HAS_RAG_POISON` parity."* Both halves are **already fixed**:
   - `predict.py:1086-1109` now computes `rag_poison_weight` via
     `get_rag_poison_weight(rag_result)` (`predict.py:1091`) AND folds it into
     the composite (`predict.py:1108-1109`: `composite = min(composite +
     rag_poison_weight, 1.0)`), with an explicit comment at `:1103-1107`
     documenting the prior "computed…then discarded" bug as resolved. The
     cited line `1201` no longer points at this code (the file is now 2049 LOC;
     the block moved to ~1086).
   - `cascade.py` HAS `_HAS_RAG_POISON` parity: import at `cascade.py:165-168`
     (`_HAS_RAG_POISON = True`), mirror compute at `cascade.py:590-600`.
   - Fixed by commit **19d730b** ("hardening(rag): wire rag_poison_weight,
     reconcile taxonomy IDs, add exfil+paraphrase patterns"). **Root cause: the
     spec was authored before 19d730b landed and was never reconciled.**

3. **Auto-memory carries the same stale "computed-then-discarded" claim as
   still-current.** `project_rag_poison_hardening.md:12` ("…computed at
   `predict.py:1091` then **discarded** — never added to composite") and the
   index line `MEMORY.md:15` describe the bug in the present tense without a
   "FIXED by 19d730b" marker. The security-research-auditor agent-memory
   (`/Users/mehrnoosh/Na0S/.claude/agent-memory/security-research-auditor/MEMORY.md`)
   has **no** rag_poison/1201/safe_yaml line (grep clean) — so the agent-memory
   correction is a no-op there; only the user auto-memory needs the FIXED tag.
   **Root cause: memory written at fix-time framed the bug as the headline; the
   "still open" framing was never flipped to "shipped" after 19d730b.**

Net: the roadmap under-reports open L11 work (overclaims COMPLETE) while the
spec + memory over-report a closed bug as open. Item 18 makes all three honest.

---

## 1. Step 1 — Explore current state vs. what it SHOULD be (gaps + edge cases)

**Current (verified):**
- `ROADMAP_V2.md:124` — `| **L11**| ... | **24/24** | COMPLETE |`
- `ROADMAP_V2.md:1104` — `## Layer 11: Supply Chain Integrity — Tasks: 24/24 (COMPLETE)`
- `ROADMAP_V2.md:1170` — `### Completed (24 items)`
- `ROADMAP_V2.md:1174-1185` — L11 `### TODO List` with 5 open bullets across
  Polish / Test-coverage / Bugs subsections.
- `ROADMAP_V2.md:135` — `| **Hardening** | ... | **10/14** | 71% |` (a SEPARATE
  hardening rollup — items 1-12 are NOT yet represented here either; flag but
  do not silently renumber — see §8 edge case).
- `specs/hardening/01..12-*.md` — 12 specs exist on disk; each header declares
  L11 / supply-chain scope (verified via `head` grep of every file).
- `specs/01-import-linter-architecture.md:16` — stale rag_poison fold-in claim.
- `project_rag_poison_hardening.md:12`, `MEMORY.md:15` — stale present-tense bug.

**Should be:**
- L11 row + header reflect the reopened scope. Two honest representations are
  acceptable; pick ONE and apply consistently (see §3 decision):
  - **(A) Re-count:** `24/24` → `24/36 (12 open hardening items)` and status
    `COMPLETE` → `HARDENING IN PROGRESS`, OR
  - **(B) Keep core 24/24 but append a clearly-labeled `### Open-Security-Hardening
    (items 1-12)` subsection** under L11 and change the *table* status cell to
    `CORE COMPLETE · +12 hardening open`.
- A new `### Open-Security-Hardening` subsection under L11 (after the existing
  `### TODO List`, before the `---` at `:1187`) enumerating items 1-12 as a
  checklist with one-line summaries + a link to each `specs/hardening/NN-*.md`,
  priority tier, and depends-on. The 2 existing L11 TODO bullets (`:1183`,
  `:1184`) should be **cross-referenced** (they overlap item families) — not
  duplicated.
- `specs/01:16` rewritten to past tense: "Fixed in 19d730b; verified
  `predict.py:1086-1109` + `cascade.py:165-168,590-600`."
- Auto-memory lines tagged `FIXED 19d730b`.

**Gaps / edge cases:**
- **Count arithmetic must be defensible** (na0s-review-checklist: no arbitrary
  numbers). "24" core items + "12" hardening specs = a stated, traceable 36.
  Do NOT invent a fractional progress bar without showing the denominator.
- **Item 18 is itself in this set** — but it is a DOC item about the others;
  decide whether to list it in the L11 subsection (recommend: yes, as a
  meta/tracking row, marked DOC) or keep it only in whatever master list
  drives these specs. Avoid a self-referential loop where checking off 18
  changes the count 18 reports.
- **Two roadmap counters** (L11 row at `:124` and Hardening rollup at `:135`).
  Clarify which one owns items 1-12 so the same work isn't double-counted.
- **Dependency edges between specs** (e.g. item 8 depends-on 4; 11 depends-on
  5,6; 7 soft-depends 6; 10 depends-on 3) must be reflected so the subsection
  isn't a flat list that hides ordering.

---

## 2. Step 2 — Explore roadmap / Taxonomy / README / Coverage Matrix / source for full picture

- **Roadmap:** §0-§1 above already map the lines. Also read the per-spec
  headers (`specs/hardening/*.md` lines 1-12) for canonical title + tier +
  depends-on to copy verbatim into the subsection (no paraphrase drift).
- **Taxonomy / Coverage Matrix:** **N/A for content** — items 1-12 are
  supply-chain integrity (pickle/HMAC/loader/CI), not attack classes with
  taxonomy codes. The ONE taxonomy-adjacent fact to KEEP accurate is that
  19d730b "reconcile[d] taxonomy IDs" for rag_poison (I1/IM); item 18 does not
  re-touch taxonomy.yaml — only verify the spec/memory text doesn't claim a
  taxonomy gap that 19d730b already closed.
- **README:** scan for any "L11 COMPLETE" / "24/24" / supply-chain "complete"
  claims and a rag_poison "discarded" mention. If the README quotes the L11
  count or the rag_poison bug as open, it inherits the same staleness → fix.
  If README does not reference these numbers, README change is N/A (Step 9).
- **Source (ground every claim):** `predict.py:159-162` (`_HAS_RAG_POISON`
  import), `:1086-1109` (compute+fold); `cascade.py:165-170,590-600` (parity);
  `scripts/safe_yaml.py` still present + `src/na0s/integrity/safe_yaml.py`
  absent (confirms `:1184` TODO genuinely open); `src/na0s/integrity/` listing
  shows `safe_content.py` + `validation_allowlist.py` still misfiled (confirms
  `:1183` genuinely open). These two open TODOs are EVIDENCE the COMPLETE tag
  is wrong — cite them in the PR body.

---

## 3. Step 3 — Root-cause implementation plan (numbered, doc-only)

**Decision: adopt representation (B)** — keep the historically-meaningful "core
24/24" but make the open work visible. It avoids rewriting the meaning of the
original 24 audit items and is the least lossy edit.

1. **`ROADMAP_V2.md:124`** — change the L11 table row status cell from
   `COMPLETE` to `CORE 24/24 · +12 hardening open` (or, if the table column is
   width-constrained, `24/24 core · 12 open`). Keep the progress bar full for
   the core 24 but DO NOT leave the word `COMPLETE` standing alone.
2. **`ROADMAP_V2.md:1104`** — change header from
   `## Layer 11: Supply Chain Integrity — Tasks: 24/24 (COMPLETE)` to
   `## Layer 11: Supply Chain Integrity — Tasks: 24/24 core complete · 12 open hardening items`.
3. **`ROADMAP_V2.md` — insert `### Open-Security-Hardening (supply-chain audit, items 1-12)`**
   immediately after the existing L11 `### TODO List` block (after `:1185`,
   before the `---` at `:1187`). Content = a 12-row checklist, each row:
   `- [ ] **Item N — <title from spec header>** (<tier>) — <one-line root
   cause>. depends-on: <list>. Spec: \`specs/hardening/NN-*.md\`.` Pull
   title/tier/depends-on verbatim from each spec's YAML-ish header.
   Cross-reference the existing `:1183` (misfiled modules) and `:1184`
   (`scripts/safe_yaml.py`) bullets with a note that they overlap the
   hardening family but are tracked separately under the original audit.
4. **`ROADMAP_V2.md:135`** — add a one-line clarifying note next to the
   `Hardening 10/14` rollup stating that that counter tracks the EARLIER
   hardening pass and that the NEW supply-chain items 1-12 are tracked under
   L11's Open-Security-Hardening subsection (prevents double-counting). Do not
   change the 10/14 numerator/denominator unless an item proves it belongs
   there — flag, don't guess.
5. **`specs/01-import-linter-architecture.md:16`** — rewrite the stale bullet to
   past tense with the verified citations and the 19d730b SHA. Keep it as a
   "fixed; gate prevents regression" note so the import-linter contract's intent
   (catch future dead folds) still reads coherently.
6. **`project_rag_poison_hardening.md:12`** + **`MEMORY.md:15`** — prepend
   `FIXED 19d730b — ` (or append `(shipped 19d730b; verified predict.py:1086-1109,
   cascade.py:590-600)`) so the bug is no longer present-tense. Do NOT delete
   the line (history is useful); just flip the tense/status.
7. **README** — only if Step 9 finds a stale claim there.
8. **Self-check the arithmetic** and the depends-on graph in the new subsection
   against each spec header (no orphan/contradictory edges).

---

## 4. Step 4 — Implement from root cause + pipeline wiring (predict.py/cascade.py parity)

**N/A — DOC-only item; no capability is added, nothing is wired into the
pipeline.** The pipeline wiring this item *references* (rag_poison fold +
cascade parity) is the SUBJECT of the doc correction, already shipped in
19d730b at `predict.py:1086-1109` and `cascade.py:165-170,590-600` — item 18
only updates prose to match that reality, it does not modify those files.

---

## 5. Step 5 — Harvester audit / harvested datasets (Q5, Q9)

**N/A — this is supply-chain/doc hardening, not a prompt-injection attack class.
There is no harvested dataset for "roadmap accuracy."** The weekly harvester,
F14 scenario library, and taxonomy tagging are untouched. (Per SCOPE RULES,
dataset/harvester steps are N/A unless the item genuinely touches them — it
does not.)

---

## 6. Step 6 — Tests for Code + Use-Case (reframed: end-to-end doc correctness)

There is no Python code to unit-test. "Tests" here = **mechanical
doc-consistency checks** an agent runs and pastes into the PR (na0s-review-
checklist: a doc PR still needs a verifiable "did it actually do what it claims"
gate; no hollow assertions). Run ALL of the following and require each to pass:

**Code-equivalent checks (claims grounded in source):**
- `grep -n "rag_poison_weight" src/na0s/predict.py` shows the compute at
  `:1091` AND the fold at `:1108-1109` (NOT discarded). Assert both present.
- `grep -n "_HAS_RAG_POISON" src/na0s/cascade.py` shows the True branch at
  `:168` and the mirror compute around `:590-600`. Assert present.
- `test -f scripts/safe_yaml.py && ! test -f src/na0s/integrity/safe_yaml.py` →
  confirms the `:1184` TODO is genuinely still open (so it stays open, not
  silently checked).
- `ls src/na0s/integrity/ | grep -E "safe_content|validation_allowlist"` →
  confirms `:1183` still open.

**Use-case / behavior checks (doc internally consistent):**
- `ls specs/hardening/*.md | wc -l` == the count claimed in the new subsection
  (12 today). Fail the PR if the subsection lists a different number than files
  on disk.
- Every `specs/hardening/NN-*.md` referenced in the subsection EXISTS (no dead
  spec links): loop `for n in 01..12; do test -f specs/hardening/$n-*.md; done`.
- The L11 count line at `:124`, the header at `:1104`, and the subsection's
  enumerated count agree (no `24/24 COMPLETE` left anywhere for L11).
- `grep -rn "predict.py:1201\|computed-then-discarded\|then.*discarded" specs/ \
   ROADMAP_V2.md` returns ONLY past-tense/fixed mentions (no live "open" claim).
- The depends-on edges in the subsection match each spec header's `depends_on:`
  (8→4, 11→{5,6}, 7→6 soft, 10→3).

**CLI / suite smoke (na0s-review-checklist mandatory):** even for a doc PR,
run `python3 -m pytest tests/ -q --tb=line` to confirm the doc edits introduced
ZERO regressions (they cannot, but the checklist requires the green-suite
confirmation before reporting done). Also `python3 -c "import na0s; print(na0s.__version__)"`
as a CLI import smoke. No threshold/number is invented anywhere in the edits.

---

## 7. Step 7 — File/dir cleanup & refactor per conventions

- **Scope discipline:** edits limited to `ROADMAP_V2.md`,
  `specs/01-import-linter-architecture.md`, `project_rag_poison_hardening.md`,
  `MEMORY.md` (user auto-memory). Do NOT touch `src/`, `tests/`, the 12
  hardening specs' bodies (only LINK to them), or the agent-memory
  `security-research-auditor/MEMORY.md` (grep-clean — no stale ref to fix).
- **No new files.** The `specs/hardening/18-roadmap-doc-updates.md` spec itself
  (this file) is the only addition and is the planning deliverable.
- **De-clutter check (Q2):** the repo root has stray scratch files
  (`_skeptic_test_out.txt`, `_xfail_run.txt`, `pyt_out.txt`, `logs/`) per
  `git status` — OUT OF SCOPE for item 18; note for a separate housekeeping
  pass (do not bundle into a doc-accuracy PR).
- Keep commit messages one-liner short per CLAUDE.md; `cd <worktree> && git
  checkout -b docs/l11-open-hardening-roadmap` first (branch prefix `docs/`).

---

## 8. Step 8 — Update the roadmap (cite commit SHA) — THIS IS THE CORE WORK

This is the primary applicable step (per item scope: "Mostly steps 8/9").
Concrete edits enumerated in §3 (items 1-4 = roadmap). On commit, the new
subsection's item rows stay UNCHECKED (items 1-12 are open); only flip a row to
`[x]` with its merge SHA when that specific hardening item actually lands. When
items 1-12 merge, return here and check them off citing each SHA (per the
Roadmap-Todo Sync memory rule). The doc-correction commit for item 18 itself
should be cited back into whatever master tracking list drives items 1-18.

**Edge case to resolve during implementation:** confirm whether the
`Hardening 10/14` rollup at `:135` is the same bucket as items 1-12 or a
distinct earlier pass. Read the rows it counts before editing; if it is
distinct, add the clarifying note (§3.4) rather than merging the counts.

---

## 9. Step 9 — README / Benchmark updates

**Conditionally applicable — only if numbers change.** Action: `grep -in
"24/24\|L11.*complete\|supply.chain.*complete\|rag_poison.*discard" README.md`.
- If a hit exists → apply the same honesty fix (L11 not fully complete; bug
  fixed) and update.
- If no hit → **N/A** (README does not quote these numbers; nothing to change).
**Benchmark:** N/A — no detector recall/FPR number changes; item 18 ships no
code, so `technique_analysis.json` / `BENCHMARK_RESULTS.md` are untouched.

---

## 10. Step 10 — Taxonomy + Coverage Matrix + per-feature thresholds

**N/A — supply-chain/doc item, no attack-class taxonomy code and no scorer
threshold is involved.** The only taxonomy-adjacent fact (19d730b reconciled
rag_poison I1/IM IDs) is already shipped and is merely *cited* by this item, not
modified. No COVERAGE_MATRIX row maps to "roadmap accuracy." No magic number is
introduced (the only numbers — 24, 12, 36 — are file-counts with a shown
denominator, verified by `ls specs/hardening/*.md | wc -l`).

---

## 11. Process — Open a PR; require passing held-out tests before merge

PR on branch `docs/l11-open-hardening-roadmap`. Body must include: the three
root-cause defects, the §6 grounding-grep outputs (proving the rag_poison fold
+ cascade parity exist and the safe_yaml/misfiled TODOs are genuinely open),
the green `pytest -q` confirmation, and the count arithmetic (24 core + 12 specs
= 36, denominator shown). Gate: full suite green (cannot regress on doc edits,
but confirm per checklist). No merge-to-main / force-push without explicit
confirm (per No-Git-Commit memory rule).

---

## Q&A self-check

- **Q1 (can Na0S handle the target + suite green):** The "target" is doc
  accuracy. After edits, every roadmap/spec/memory claim is grounded in a
  verified `file:line`; full `pytest -q` stays green (no code touched). PASS
  once §6 checks + suite run clean.
- **Q2 (cleanup done):** Doc files de-cluttered (stale claims removed/flipped).
  Root-level scratch files + extra `logs/` noted as out-of-scope for a separate
  housekeeping pass — flagged, not silently bundled.
- **Q3 (pipeline wiring):** N/A — DOC item. The wiring it *describes*
  (rag_poison fold + cascade `_HAS_RAG_POISON` parity) is verified present at
  `predict.py:1086-1109` / `cascade.py:165-170,590-600` (19d730b).
- **Q4 (tested code AND use-case):** §6 — grounding-greps (code) + doc-self-
  consistency checks (use-case) + suite smoke. No hollow assertions.
- **Q5 (harvester audit):** N/A — no harvested dataset for roadmap accuracy.
- **Q6 (taxonomy + coverage matrix):** N/A — supply-chain/doc, not an attack
  class; only verify the spec/memory don't claim a taxonomy gap 19d730b closed.
- **Q7 (scorer scores it right):** N/A — no scorer; no threshold added.
- **Q8 (predict.py / cascade.py references):** YES, as the *subject* of the
  correction. `predict.py:159-162,1086-1109`; `cascade.py:165-170,590-600`.
  These are READ to verify the fix; NOT modified.
- **Q9 (harvester agent harvests this type):** N/A — not a harvestable data
  type.
- **Q10 (other correctness checks):** (a) resolve the L11-vs-Hardening-rollup
  double-count question; (b) ensure item 18's own row doesn't create a
  self-referential count loop; (c) confirm agent-memory
  `security-research-auditor/MEMORY.md` is grep-clean (it is) so no edit is
  forced there.

---

## Agent / skill team (inject na0s-review-checklist into every agent prompt)

| Step | Owner agent / skill | Why |
|---|---|---|
| 1-2 explore + ground claims | **security-research-auditor** + skill **na0s-review-checklist** | audits L11/supply-chain area, verifies every `file:line`, no hallucinated APIs |
| 1-2 spec/source cross-read | **layer-9-11-auditor** | owns L9-L11 scope; confirms the 12 specs are all L11 and the COMPLETE tag is wrong |
| 3 plan review | **Plan** (planning) | sanity-checks representation (B) + count arithmetic |
| 5 verify nothing wired/discarded | **silent-failure-hunter** | confirms rag_poison fold is live (not a silently-dropped weight) so the spec/memory "discarded" claim is truly stale |
| 6 doc-consistency + suite smoke | skill **na0s-debugging** + skill **eval-harness** | runs `pytest -q` green-gate + confirms no benchmark numbers move |
| 8-9 roadmap/README edits | **security-research-auditor** (executor) | applies the honest count/subsection edits |
| 11 PR | skill **github-pr-prep** then **pr-review-toolkit:review-pr** / **github-pr-review** | prep PR body with grounding-greps; precision-first review before merge |
| (not used) | data-harvesting, cron-scheduling, github-ci-fix | N/A — no harvest, no cron, no CI red to fix for a doc PR |

---

## Execution preconditions / dependencies

- **Hard deps: NONE.** The count-fix + stale-ref purge + subsection skeleton can
  land standalone with items 1-12 listed as OPEN.
- **Soft dep on items 1-12 for final check-off:** each row in the new
  Open-Security-Hardening subsection flips to `[x]` only when that item merges,
  citing its SHA. Item 18 therefore *outlives* a single PR — it is the tracking
  home that items 1-12 check into.
- **Confirmed-already-landed dep:** commit **19d730b** (rag_poison wiring +
  cascade parity + taxonomy reconcile) — the precondition that makes the
  `specs/01:16` and memory corrections valid. Verified present on the current
  branch's history.
- **Must verify against MAIN, not the stale editable-install env** (per
  na0s-debugging memory): re-confirm `predict.py`/`cascade.py` line numbers on
  the branch the PR targets before writing them into the spec, since the
  editable install may point at a d8 checkout (line numbers differ there).

---

## Definition of done

- [ ] `ROADMAP_V2.md:124` L11 row no longer reads bare `COMPLETE`; shows core
      24/24 + 12 open (denominator visible).
- [ ] `ROADMAP_V2.md:1104` header updated to "24/24 core complete · 12 open
      hardening items".
- [ ] New `### Open-Security-Hardening (items 1-12)` subsection added under L11
      with 12 unchecked rows, each citing title/tier/depends-on (verbatim from
      spec headers) + a working `specs/hardening/NN-*.md` link.
- [ ] `Hardening 10/14` rollup (`:135`) clarified vs. items 1-12 (no
      double-count); numerator/denominator unchanged unless proven otherwise.
- [ ] `specs/01-import-linter-architecture.md:16` rewritten to past tense citing
      19d730b + verified lines (`predict.py:1086-1109`, `cascade.py:590-600`).
- [ ] `project_rag_poison_hardening.md:12` + `MEMORY.md:15` tagged
      `FIXED 19d730b` (tense flipped, line preserved).
- [ ] README checked; updated only if it quoted the stale numbers (else N/A).
- [ ] §6 grounding-greps + doc-consistency checks all pass; `ls
      specs/hardening/*.md | wc -l` matches the subsection count.
- [ ] `python3 -m pytest tests/ -q --tb=line` green (0 regressions); `python3
      -c "import na0s"` smoke OK.
- [ ] No `src/` or `tests/` file modified (`git diff --name-only` shows only
      docs).
- [ ] PR opened on `docs/l11-open-hardening-roadmap`, reviewed precision-first,
      merge gated on green suite + explicit confirm.
