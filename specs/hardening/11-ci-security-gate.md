---
item: 11
title: CI security gate — bandit + fickling/modelscan + ruff S + CodeQL security-extended + gate security-review.yml
priority_tier: P1 (CI/governance — makes the supply-chain hardening of #1–#6 enforceable, not just present)
depends_on: [5, 6]        # #5 (route raw loaders through safe_load) MUST land first so bandit/ruff-S pass; #6 (fail-closed optional loaders) removes the remaining raw torch.load
applicable_steps: [1, 2, 3, 4, 6, 7, 8, 11]   # template steps + Q1, Q2, Q3, Q8, Q10
na_steps: [5, 9, 10]                            # + Q4(reframed-applicable), Q5, Q6, Q7, Q9 (see N/A justifications)
classification: CI/Governance gate over supply-chain SAST — NOT a prompt-injection attack class
---

# Item 11 — CI security gate (bandit + fickling/modelscan + ruff S + CodeQL security-extended + gate security-review.yml)

## 0. Root cause (one sentence)
Na0S has the SAST *tools* present but none of them **gate the build**: bandit runs only as a local
pre-commit hook (never in CI), ruff's flake8-bandit `S` rules are not selected anywhere, CodeQL runs
with the *default* query suite (not `security-extended`), there is no pickle/model-artifact content
scanner (fickling/modelscan) despite ROADMAP_V2 listing modelscan for L11, and `security-review.yml`
is a soft inline-comment action with no pass/fail gate — so a future raw `pickle.load`/`torch.load`
or a known-vuln dep can land on `main` with green checks.

---

## 1. KEY REFS — confirmed current line numbers (verified against the real files, corrections noted)

| Ref | File | What's there today | Lines (ref said → actual) | Gap |
|-----|------|--------------------|---------------------------|-----|
| A | `.pre-commit-config.yaml` | bandit hook `-c pyproject.toml`, `exclude: tests/` | **26–32** (ref ✅) | LOCAL ONLY — never runs in CI; and **`pyproject.toml` has NO `[tool.bandit]` section** (verified: only `setuptools`/`pytest`/`coverage` tables exist) so `-c pyproject.toml` gives bandit an empty config and zero exclusions/skips |
| B | `.github/workflows/ci.yml` | flake8 `--select=E9,F63,F7,F82` (line 54), warn-only flake8 (56), py_compile, coverage gate `--fail-under=50` (79), recall-gate `continue-on-error` (110–113) | lint block **51–113** (ref ✅) | NO bandit, NO ruff `S`, NO model/pickle scan, NO dep audit gate. ruff is in pre-commit (`.pre-commit-config.yaml:14–18`) but **not in CI at all** |
| C | `.github/workflows/pr-check.yml` | py_compile + blocking flake8 (E9/F63/F7/F82) + full coverage pytest | whole file **1–68** (ref gave no lines) | same gaps as ci.yml; this is the PR-blocking workflow, so the gate could live here OR in ci.yml |
| D | `.github/workflows/codeql.yml` | `init@v4 languages: python` then `analyze@v4`, weekly cron + PR + push | **20–28** (ref said 22–25 → drifted by the `@v4` bump; `init` is now 22–25, `analyze` 27–28) | NO `queries: security-extended` / `security-and-quality` on the `init` step → runs the smaller default suite |
| E | `.github/workflows/security-review.yml` | `anthropics/claude-code-security-review@main` with `comment-pr: true`, `claude-api-key` | action step **28–31** (ref ✅) | soft: posts inline comments, **never fails the check**; also requires a `CLAUDE_API_KEY` secret the user does NOT have (memory: subscription-only, no API key) → this job will hard-fail on a missing secret for every PR until either the secret is added or the job is made non-blocking / removed |
| F | `pyproject.toml` | dep floors+caps, optional extras, `[tool.setuptools/pytest/coverage]` | **whole file 1–128** | NO `[tool.bandit]`, NO `[tool.ruff]` / `[tool.ruff.lint]`; bandit, modelscan, fickling, ruff are NOT declared deps anywhere |

Supporting facts (verified, load-bearing):
- **Only ONE** `# noqa: S301` in the whole tree — `src/na0s/ml/stacking_classifier.py:130` — added speculatively; ruff `S`/flake8-bandit is **not** actually enabled, so it suppresses nothing today. Item #5 §3.3.8 removes it when it routes that loader through `safe_load`.
- Raw loaders that a bandit/ruff-S/fickling gate would flag TODAY (these are exactly what #5/#6 fix):
  `ml/faiss_classifier.py:219` (`pickle.load`), `ml/stacking_classifier.py:130` (`pickle.load`),
  `ml/embedding_adapter.py:437` (`torch.load`, no `weights_only`), `worm/detector.py:647` (`joblib.load`, already sidecar-gated).
  ⇒ **This item MUST land after #5 (and #6 for the torch.load) or the new gate red-fails on day one.**
- `integrity/dep_scanner.py:29` already shells `subprocess.run([...pip-audit...])` at *runtime* — pip-audit is a known dependency-audit surface but is NOT a CI gate. Reuse the same tool name for the CI dep-audit step (consistency, no new tool).
- ROADMAP_V2.md:2507 lists **`| modelscan | Pickle content scanning | L11 |`** — modelscan is already an intended L11 control; wiring it into CI satisfies a tracked roadmap row, not a new invention.
- CodeQL history is in CHANGELOG.md:56,88 ("weekly Python code scanning", default queries) — upgrading to `security-extended` is an additive change to an existing, working workflow.
- No `.flake8`/`setup.cfg`/`tox.ini`/`ruff.toml` exists — all lint config is inline in CI or in `.pre-commit-config.yaml`; the ruff `S` select must therefore be added to a NEW `[tool.ruff.lint]` table in `pyproject.toml` (single source) and referenced by both CI and pre-commit.
- `SECURITY.md` exists (10.5 KB) — Step 9 may add a one-line "CI SAST gate" row, optional.

---

## 2. Gap vs. ideal

| Control | Current | Ideal (target) |
|---------|---------|----------------|
| bandit (Python SAST) | pre-commit only, empty `[tool.bandit]`, `exclude tests/` | runs in CI as a **blocking** step over `src/`; real `[tool.bandit]` in pyproject (skips justified, `exclude_dirs=["tests"]`); fails on Medium+/High severity-confidence |
| ruff `S` (flake8-bandit) | not selected | `[tool.ruff.lint] select=["S",...]` with a per-file ignore for `tests/` and any *justified* `S` (each `noqa: S###` carries an inline reason); CI runs `ruff check` blocking |
| pickle/model content scan | none (only sidecar/HMAC at *load* time) | **modelscan** (and/or **fickling**) scans `src/na0s/models/*.pkl` in CI; any opcode-level RCE primitive (`GLOBAL`/`REDUCE` to os/subprocess/builtins) fails the build |
| dependency audit | runtime `dep_scanner` (pip-audit) only | a CI **dep-audit** step (pip-audit) reporting known CVEs in the locked deps; advisory→gate ratchet (start non-blocking, then block on High) |
| CodeQL | default query suite | `queries: security-extended` (or `security-and-quality`) on `init` |
| `security-review.yml` | soft, comments only, needs absent `CLAUDE_API_KEY` | make it **non-blocking / fork-safe** given the no-key reality: gate on the *deterministic* SAST jobs (bandit/ruff-S/modelscan/CodeQL), and leave the agentic Claude review as advisory `continue-on-error` (or guard the step with `if: secrets.CLAUDE_API_KEY != ''`) so a missing key never red-fails every PR |
| grep-test for new raw loaders | none | a **pytest** that greps `src/na0s/` for new un-gated `pickle.load`/`joblib.load`/`torch.load`/`pickle.loads` and **fails** unless the call site is the canonical `integrity.safe_pickle` path or carries a justified `# noqa: S301 — <reason>` → defends the #5/#6 fix in perpetuity |

Edge cases / FP-safety to design for:
- **Bandit false positives**: `assert` in tests (B101), `subprocess` in `dep_scanner.py` (B404/B603) are legitimate. Handle via `[tool.bandit] exclude_dirs` + per-line `# nosec B###  <reason>` with an inline justification — never a blanket skip-all. (Mirror this in ruff with `per-file-ignores`.)
- **Keyless reality**: the deterministic gate (bandit/ruff-S/modelscan/pip-audit/CodeQL) needs NO API key — it is the real gate. The Claude security-review action is *advisory only* and must not block when the secret is unset (memory: subscription-only host).
- **modelscan on absent models**: `src/na0s/models/` may ship `.pkl` via package-data (`pyproject.toml:120`); the scan step must no-op cleanly (exit 0) when no model file is present, not error.
- **Tool-version drift**: pin bandit/ruff/modelscan/pip-audit versions (a `[project.optional-dependencies] cisec = [...]` extra) so the gate is reproducible run-to-run and matches the pre-commit `rev:` pins (`.pre-commit-config.yaml:15,21,27`).
- **grep-test must be FP-safe**: it must allow the *gated* call sites (post-#5 those route through `safe_pickle`, so the literal `pickle.load` lives only inside `integrity/safe_pickle.py`); allow-list that one canonical module + any `# noqa: S301 — <reason>` line, fail on everything else. No magic threshold — it is a binary "new ungated loader present?" check.

---

## 3. Root-cause implementation plan (numbered, by file)

### 3.1 Tool config — single source of truth in `pyproject.toml` (Ref F)
1. Add `[tool.bandit]` table: `exclude_dirs = ["tests", "build", "dist"]`; explicit `skips` only for rules with a written justification (none blanket). This makes the EXISTING pre-commit `-c pyproject.toml` (Ref A) actually configured AND lets CI reuse the same config.
2. Add `[tool.ruff.lint]` table: extend `select` to include `"S"` (flake8-bandit) alongside the default `E,F`; add `[tool.ruff.lint.per-file-ignores]` `"tests/**" = ["S101","S105","S106","S301"]` (asserts + test fixtures), and any single justified src ignore inline via `# noqa`. Set `[tool.ruff] line-length = 120` to match the CI flake8 `--max-line-length=120` (ci.yml:56) — no behavior conflict.
3. Add a `cisec` optional-deps extra: `bandit[toml]>=1.7.8`, `ruff>=0.4.4`, `modelscan>=0.8`, `pip-audit>=2.7`, `fickling>=0.1` (pin floors matching pre-commit `rev:` so local==CI). Justification for the floors: match the already-pinned pre-commit revs (`.pre-commit-config.yaml:15 ruff v0.4.4`, `:27 bandit 1.7.8`); modelscan/pip-audit/fickling floors are the lowest releases with the JSON/`--format` output the gate parses.

### 3.2 CI gate workflow (Ref B / C)
4. Add a **`security` job** — preferred location: a new top-level `.github/workflows/security-gate.yml` (keeps the SAST gate independent of the test matrix so a flaky test never masks a SAST regression, and vice-versa). Triggers: `pull_request: [main]` + `push` on the same branch globs as ci.yml:4–13. `permissions: contents: read, security-events: write` (the last only if uploading SARIF).
   Steps (each BLOCKING unless marked):
   - `pip install -e ".[cisec]"`.
   - **bandit**: `bandit -c pyproject.toml -r src/na0s -ll` (`-ll` = report Medium+ severity; `-c` reuses §3.1.1). Blocking.
   - **ruff S**: `ruff check src/ tests/` (the `S` select from §3.1.2 now active). Blocking.
   - **modelscan**: `modelscan -p src/na0s/models -r json` (or iterate `*.pkl`); fail on any flagged operator. No-op-clean if dir empty (§2 edge case). Blocking once green on the shipped models.
   - **fickling** (defense-in-depth, optional/advisory at first): `fickling --check-safety src/na0s/models/*.pkl` → `continue-on-error: true` initially, ratchet to blocking after a clean baseline.
   - **dep-audit**: `pip-audit` (reuse the tool `dep_scanner.py:29` already calls) → start `continue-on-error: true` (advisory), ratchet to blocking on High once the baseline is clean. *Justification for the advisory→blocking ratchet*: identical pattern to the existing `recall-gate` rollout (ci.yml:106–113 comment) — do not block on a backlog of pre-existing findings; block only on NET-NEW.
   - **grep-test for new raw loaders**: `python3 -m pytest tests/integrity/test_no_raw_loaders.py -q` (the test authored in §3.4). Blocking.
5. Do NOT duplicate the gate in both ci.yml and pr-check.yml — put it once in `security-gate.yml` and make it a required check in branch protection (Step 11). Note in the workflow header which checks are required.

### 3.3 CodeQL upgrade (Ref D)
6. In `codeql.yml` `init` step (lines 22–25) add:
   ```yaml
   with:
     languages: python
     queries: security-extended
   ```
   (or `security-and-quality` if the quality lint is also wanted — pick `security-extended` to stay security-scoped and keep noise down). This is additive to the existing weekly+PR run; no new job.

### 3.4 grep-test (the item-specific deliverable) — `tests/integrity/test_no_raw_loaders.py`
7. New pytest (mirrors source layout: `tests/integrity/`). It walks `src/na0s/**/*.py`, regex-scans for `\b(pickle\.loads?|joblib\.load|torch\.load)\s*\(`, and asserts the match set is a subset of an ALLOW-LIST:
   - `src/na0s/integrity/safe_pickle.py` (the canonical, digest-gated `pickle.load` lives here by design),
   - any line carrying `# noqa: S301` **with a trailing reason** (regex requires non-empty justification text),
   - (transitional) the specific #5/#6-gated call sites IF #5/#6 land in the same train — but the intent is that post-#5 those route through `safe_pickle`, so the steady-state allow-list is just `safe_pickle.py`.
   The test FAILS with a precise message listing any new ungated `file:line` so a reviewer sees exactly what to route through `safe_load`. No threshold/magic-number — binary presence check. Assertion-rich: asserts the exact offending paths, not just "len==0".
8. Companion assertion: the test also confirms `# noqa: S301` count has not silently grown (snapshot the justified set) — prevents the gate being defeated by sprinkling `noqa`.

### 3.5 security-review.yml hardening (Ref E)
9. Guard the agentic Claude review so the absent `CLAUDE_API_KEY` never red-fails PRs:
   - add `if: ${{ secrets.CLAUDE_API_KEY != '' }}` to the `security` job (skip cleanly when no key), OR set the action step `continue-on-error: true`.
   - Keep the deterministic `security-gate.yml` (§3.2) as the REAL required gate. Document in both headers: "deterministic SAST is the gate; the Claude agentic review is advisory."
   - Leave the existing "Require approval for external contributors" fork-safety note (security-review.yml:9–10) intact.

### 3.6 Makefile parity (local==CI)
10. Add a `make security` target: runs `bandit -c pyproject.toml -r src/na0s -ll && ruff check src/ tests/ && modelscan -p src/na0s/models && pip-audit && pytest tests/integrity/test_no_raw_loaders.py`. Mirrors the CI job so a dev can reproduce the gate locally (the existing Makefile already mirrors CI: `lint`/`recall-gate` targets, Makefile:16–34). Add to `.PHONY` line (Makefile:1).

---

## 4. Step-by-step template instantiation

**Step 1 — Explore current rules.** DONE (§1–2): all five CI surfaces + tool-config absence + the four raw loaders + the lone speculative `noqa: S301` + the modelscan roadmap row confirmed at exact lines (with the codeql line-drift and the empty-`[tool.bandit]` correction noted).

**Step 2 — Roadmap/Taxonomy/README/Coverage for the gaps.** ROADMAP_V2 already claims CI is "fully wired … bandit … CodeQL" (lines 1661, 1708) and lists modelscan for L11 (2507). The CLAIM is **partially inaccurate**: bandit is pre-commit-only (not CI) and modelscan is listed but unwired. Step 8 reconciles the over-claim and checks off the new gate. Taxonomy/Coverage Matrix = N/A (this is a governance control, not an attack class — see Step 10).

**Step 3 — Root-cause plan.** DONE (§3).

**Step 4 — Implement + wire (parity).** Per §3.1–3.6. "Wiring" here = the gate is wired into the CI graph + branch protection, and the grep-test is wired into the pytest suite so it runs on every `make security` and every CI run. predict.py/cascade.py are NOT touched (Q8: this item adds no detector; the only runtime code is the grep-test, which lives under `tests/`). No `_HAS_*` flag — CI workflows + one test + tool config only.

**Step 5 — Harvester audit.** **N/A — CI/governance gate; no harvested dataset. The only "data" is the existing shipped `src/na0s/models/*.pkl` that modelscan inspects, not harvested intel.**

**Step 6 — Tests (Code + Use-Case).** See §5. Code = the grep-test asserts on synthetic in-tree fixtures; Use-Case (reframed per scope: end-to-end behavior of the integrity gate) = a tampered/new raw loader is REJECTED by the gate while the legit gated `safe_pickle` call site still PASSES, and the full pytest suite stays green.

**Step 7 — Cleanup/refactor.** (a) Once #5 removes the `# noqa: S301` (stacking:130), the grep-test allow-list shrinks to just `safe_pickle.py` — keep the two in sync. (b) Move all lint/SAST config into `pyproject.toml` (§3.1) so there is ONE source (no scattered `.flake8`/`.bandit`). (c) Optional: fold the warn-only flake8 (ci.yml:56) toward ruff to avoid two linters — out of scope, note as follow-up. (d) Ensure `make security` is added to `.PHONY` and `help`.

**Step 8 — Roadmap update.** Check off the item in ROADMAP_V2 (Infra/L14/L11 sections); cite the landing commit SHA; correct the "bandit … CI fully wired" over-claim (1661/1708) to reflect that bandit/ruff-S/modelscan/CodeQL-extended are NOW genuinely gating; mark the modelscan-L11 row (2507) as wired.

**Step 9 — README/Benchmark.** README badge: add a "security gate" / CodeQL badge (optional). SECURITY.md: add a one-line "CI runs bandit + ruff-S + modelscan + CodeQL security-extended on every PR" row. **No benchmark/recall change** — this is governance, not detection (recall/FPR unchanged).

**Step 10 — Taxonomy + Coverage Matrix + per-feature thresholds.** **N/A — a CI SAST gate is not a prompt-injection technique with a taxonomy code, a COVERAGE_MATRIX row, or a scored detection threshold.** The only "thresholds" are bandit severity `-ll` (Medium+, a documented bandit level, not arbitrary) and the advisory→blocking ratchet for pip-audit/fickling (justified by the existing recall-gate rollout pattern, ci.yml:106–113). No magic numbers introduced.

**Step 11 — PR + held-out test gate.** Branch `ci/ci-security-gate` off `main` (CLAUDE.md branch naming: `ci/`). Require full `pytest tests/ -q --tb=line` green + the new grep-test green + the new `security-gate.yml` green before merge; add `security-gate` to branch-protection required checks. Use `github-pr-prep` → `github-pr-review`; drive CI green with `github-ci-fix`. Do not merge to main without user confirmation (memory).

---

## 5. Test plan (Code + Use-Case) — `tests/integrity/`

Mirrors source layout (`tests/integrity/`, the existing integrity test dir). No hollow tests — every assertion checks a concrete path/exception/exit, not just "ran".

**Code-level (the grep-test, §3.4):**
1. `test_no_raw_loaders_in_src` — scan `src/na0s/`, assert the ungated-loader match set equals the allow-list; on failure the message lists offending `file:line`. Positive control: temporarily inject (in a `tmp_path` copy of a fake module, NOT the real tree) a `pickle.load(f)` line → assert the scanner FLAGS it. Negative control: a `# noqa: S301 — verified-by-safe_pickle` line → assert the scanner ALLOWS it. This proves the regex + allow-list logic, not just the current-tree happy path.
2. `test_noqa_s301_count_is_pinned` — assert the set of justified `# noqa: S301` sites matches the documented snapshot (post-#5 this should be empty or just `safe_pickle.py`); a NEW unjustified `S301` fails. Asserts the exact set, not a count alone.
3. `test_safe_pickle_is_only_canonical_loader` — assert `integrity/safe_pickle.py` is the sole src file containing a bare `pickle.load` (post-#5 steady state). Concrete path assertion.

**Use-Case / behavior (gate end-to-end, reframed per scope):**
4. **Tooling config loads**: assert `pyproject.toml` parses and contains `[tool.bandit]` + `[tool.ruff.lint]` with `"S"` in select (parse with `tomllib`; assert keys present) — catches a malformed table that would silently disable the gate.
5. **Legit gated path still passes**: with the #5/#6 fix in place, run the grep-test against the real tree → PASS (proves the gate does not false-positive on the canonical `safe_pickle` call site).
6. **New raw loader is rejected**: simulate a reviewer adding `torch.load(p)` to a copied fake module → the scanner returns it as offending (the gate would fail the build). Asserts the exact offending line.
7. **modelscan no-op on empty dir** (if modelscan importable; else `importorskip`): point modelscan at an empty `tmp_path` → exit 0, no error (proves §2 edge case). Skip cleanly when modelscan absent in dev env.

**Smoke step (CLI/suite, per na0s-review-checklist):**
8. Workflow lint: `python3 -c "import yaml,sys; [yaml.safe_load(open(f)) for f in ['.github/workflows/security-gate.yml','.github/workflows/codeql.yml','.github/workflows/security-review.yml']]; print('yaml-ok')"` — catches the YAML-syntax / hallucinated-key failure mode in the new workflow before it ever runs in Actions.
9. `make security` runs locally end-to-end (or each tool individually if a tool is absent in dev) — proves local==CI parity (no mocked-CLI gap).
10. `python3 -m pytest tests/integrity/ -q --tb=line` then full `python3 -m pytest tests/ -q --tb=line` — zero regressions (CLAUDE.md). The grep-test must pass ONLY after #5/#6 land (precondition §8).

---

## 6. Q&A self-check

- **Q1 — Can Na0S handle the target (gate the threat + suite stays green)?** Not today — bandit is pre-commit-only, ruff-S unselected, no model scan, CodeQL default-suite, security-review soft. After the fix: a NET-NEW raw loader / Medium+ bandit finding / known-vuln dep / unsafe pickle opcode fails the PR; the grep-test + full suite stay green once #5/#6 have routed the existing loaders. Verified by tests 1–10.
- **Q2 — Cleanup done?** Yes — all SAST config consolidated into `pyproject.toml` (no scattered configs); the lone speculative `# noqa: S301` reconciled with #5; over-claims in ROADMAP_V2 (1661/1708) corrected; `make security` added for parity.
- **Q3 — Pipeline wiring correct?** "Wiring" = CI graph + branch-protection required check + the grep-test in the pytest suite + `make security`. No detector/`predict.py`/`cascade.py` wiring (none needed — see Q8).
- **Q4 — Tested for code AND use-case?** Yes (reframed): code = grep-test logic with positive/negative controls (1–3); use-case = tampered/new loader rejected + legit gated path passes + modelscan no-ops + full suite green (5–10).
- **Q5 — Harvester audit.** N/A — CI governance gate; no harvested dataset (only the shipped `*.pkl` that modelscan inspects).
- **Q6 — Taxonomy + Coverage Matrix.** N/A — a SAST gate is not a taxonomy-coded attack class.
- **Q7 — Scorer.** N/A — no per-attack detection threshold; the only levels (bandit `-ll` Medium+, advisory→blocking ratchet) are documented tool levels mirroring the existing recall-gate rollout, not arbitrary magic numbers.
- **Q8 — predict.py / cascade.py references?** **No** — this item adds no detector and no runtime signal; it touches only CI workflows, `pyproject.toml` tool config, the Makefile, and one `tests/integrity/` grep-test. predict.py/cascade.py are untouched.
- **Q9 — Harvester agent harvest this type?** N/A — not a harvestable intel type.
- **Q10 — Other correctness checks.** (i) The gate MUST land after #5 (raw loaders) and #6 (fail-closed torch.load) or it red-fails on existing code — enforced as a depends-on. (ii) Keyless host: the deterministic gate needs no API key; the Claude agentic review is guarded `if secrets != ''` / `continue-on-error` so a missing `CLAUDE_API_KEY` never blocks. (iii) bandit/ruff FP handling via justified `[tool.bandit] skips` + ruff `per-file-ignores` (e.g. test asserts, `dep_scanner` subprocess) — never blanket-skip. (iv) Pin tool versions (`cisec` extra) matching pre-commit revs for reproducibility. (v) modelscan must no-op cleanly on an empty/absent models dir.

---

## 7. Agent / skill assignment (inject na0s-review-checklist into every subagent prompt)

| Step | Owner | Why |
|------|-------|-----|
| 1–2 explore CI + roadmap | `security-research-auditor` + skill `na0s-debugging` | map the 5 CI surfaces + tool-config gaps against MAIN (PYTHONPATH=<worktree>/src); confirm line drift |
| 3.1 tool config (pyproject) | `layer-9-11-auditor` + skill `security-review` | bandit/ruff-S config + `cisec` extra are L11 supply-chain governance |
| 3.2 security-gate.yml + 3.5 security-review guard | `layer-9-11-auditor` + skill `github-ci-fix` | author the gating workflow; ratchet advisory→blocking like the recall-gate |
| 3.3 CodeQL security-extended | `security-research-auditor` + skill `security-review` | additive query-suite upgrade |
| 3.4 grep-test (no-raw-loaders) | `silent-failure-hunter` + skill `na0s-debugging` | the test must NOT silently pass when a new ungated loader appears (its core job) — assertion-rich, positive+negative controls |
| 3.6 Makefile parity | `l3-l5-code-auditor` | mirror CI locally (`make security`) |
| 6 tests | `silent-failure-hunter` (no hollow tests; controls) | grep-test + tooling-config + modelscan-noop |
| 8 roadmap | `Plan` | check off + cite SHA; correct the bandit/modelscan over-claims |
| 11 PR + branch protection | `github-pr-prep` → `github-pr-review` (`pr-review-toolkit:*`) + skill `github-ci-fix` | PR prep/review; mark `security-gate` a required check; drive CI green |

N/A skills for this item: `eval-harness`, `data-harvesting`, `eval-scenario-curation`, `incident-to-scenario`, `cron-scheduling`, `detector-authoring`, `llm-judge` — no eval/harvest/scenario/cron/detector/judge surface in a CI SAST gate. (`cron-scheduling` is borderline — CodeQL keeps its existing weekly cron unchanged; if the dep-audit ratchet wants a weekly schedule later, invoke `cron-scheduling` then, not now.)

---

## 8. Execution preconditions / dependencies
- **Depends-on: #5 (route raw loaders through `safe_load`) and #6 (fail-closed optional loaders / torch.load).** These remove the four raw `pickle.load`/`torch.load`/`joblib.load` sites (faiss:219, stacking:130, embedding_adapter:437, worm:647) so bandit/ruff-S/the grep-test pass on day one. Landing #11 before them makes the gate red-fail on pre-existing code (anti-pattern; the gate must come up green).
- Work in a dedicated git worktree on branch `ci/ci-security-gate` off `main` (worktree discipline; never branch-switch the primary checkout).
- **Keyless host** (memory: subscription-only, no API key): the deterministic SAST gate is the real gate; the Claude agentic `security-review.yml` is advisory and must be guarded so a missing `CLAUDE_API_KEY` never blocks a PR.
- Pin the `cisec` tool versions to match the pre-commit `rev:` pins so local pre-commit == CI.
- Verify against MAIN, not the stale editable install.

## 9. Definition of done
- [ ] `[tool.bandit]` + `[tool.ruff.lint]` (with `"S"` select + justified `per-file-ignores`) added to `pyproject.toml`; `cisec` extra declared with pinned tool versions.
- [ ] `.github/workflows/security-gate.yml` added: blocking bandit (`-c pyproject.toml -r src/na0s -ll`) + ruff `S` + modelscan(src models) + grep-test; pip-audit & fickling start advisory (`continue-on-error`) with a documented ratchet.
- [ ] `codeql.yml` `init` step gains `queries: security-extended`.
- [ ] `security-review.yml` guarded (`if: secrets.CLAUDE_API_KEY != ''` or `continue-on-error`) so the missing key never red-fails PRs; deterministic gate documented as the real gate.
- [ ] `tests/integrity/test_no_raw_loaders.py` added: FP-safe allow-list (canonical `safe_pickle` + justified `# noqa: S301`), assertion-rich with positive+negative controls; `noqa: S301` set pinned.
- [ ] `make security` target added (`.PHONY` + `help`) mirroring the CI gate locally.
- [ ] YAML smoke-parse of all three touched workflows passes; gate comes up GREEN (because #5/#6 landed first).
- [ ] `pytest tests/integrity` green, then full `pytest tests/ -q --tb=line` zero regressions.
- [ ] ROADMAP_V2 item checked off with commit SHA; bandit/modelscan "CI fully wired" over-claims (1661/1708/2507) reconciled.
- [ ] PR opened; `security-gate` added to branch-protection required checks; merge gated on green full suite; main-merge confirmed with user.
