# Na0S v1.0.0 Refactor Playbook

> The method, the security gate, and the current progress map for the 13-phase
> repo reorg (flat `src/na0s/*.py` → semantic sub-packages → `v1.0.0` tag).
> Compiled 2026-06-02 from two research passes + the fusion-layer pilot that
> proved the loop on 5 real bugs. Read this before resuming the refactor.

---

## 0. The core principle

You refactor **layer by layer** not just to move files, but because each layer is
the moment to **review the code, fix bugs, and improve it** — then relocate the
*improved* code. A pure file-move would relocate hidden bugs into a cleaner tree
that only *looks* professional. The fusion pilot proved this: a review-before-move
found **5 real security bugs** (incl. fail-open-on-NaN) that green tests never caught.

**Order per layer:** `research → review/fix → SECURITY GATE → move`. Never move first.

---

## 1. The per-layer loop (security-grade)

Run this once per sub-package. Relocation and logic-changes are **separate commits**.

### Step 0 — Freeze the baseline (BEFORE touching anything)
```bash
python scripts/canary_eval.py            # record TPR / FPR / F1  (the gate)
python -m pytest tests/<layer>/ -q       # record green test count
# optional: scan-latency baseline (ReDoS/perf gate)
```
This snapshot is the **gate**: no change may make these worse.

### Step 1 — Review (research agent + review/patch agent, read-only first)
Hold each file to the 5 questions:
1. 💀 **Dead?** anything import/call it, or orphaned?
2. 🐞 **Bugs / security issues?**
3. 🥱 **Too simple/naive?** does it cut corners?
4. ✅ **Does it do what it claims?** (the fusion `evidence_grading` did NOT — it was a no-op)
5. ⬆️ **How could it be better?** (cite prior art)

### Step 2 — Verify findings independently
Reproduce each claimed bug with a tiny script. **Never patch on an agent's word** —
in the pilot, an alarming "BYPASS" finding turned out to be a *pre-existing* gap
(confirmed by stash-testing before/after), while 5 *other* bugs reproduced exactly.

### Step 3 — Patch (each fix = its own commit + new regression test)
Fix the real ones. Security-critical logic = human-approved before merge.

### Step 4 — THE SECURITY GATES (the senior-eng layer — see §2)
After ANY change, re-run the gate. If detection dropped → reject the change,
even if unit tests pass.

### Step 5 — Move (only now, on the cleaned code)
```bash
git mv src/na0s/foo.py src/na0s/<subpkg>/foo.py    # commit 1: PURE move, no edits
# add __init__.py in the SAME commit if new package (avoid implicit namespace pkg)
python -m libcst.tool codemod rename.RenameCommand \
    --old_name=na0s.foo --new_name=na0s.<subpkg>.foo src/ tests/   # commit 2: rewrite imports
ruff check --fix --select I src/ tests/             # sort imports (review __init__.py)
```
Leave a **PEP 562 lazy `__getattr__` shim** at the old path (warns only on access).
Fix internal callers to import the canonical path.

### Step 6 — Final gate + commit + one PR per layer
`pytest --collect-only -q` (fast import check) → full `pytest tests/` must match
baseline → push `refactor/<layer>-reorg`.

---

## 2. The 6 security gates (non-negotiable for a detector)

1. **Detection-efficacy gate** — "tests pass" ≠ "still catches attacks". Re-run
   `canary_eval` before/after every change; TPR can't drop, FPR can't rise.
   *This is the #1 failure mode that hides behind green tests.*
2. **Move ≠ fix in one commit** — a pure relocation has ZERO behavior change.
   Any bug-fix is a separate commit + new test. (Security: lets a reviewer see
   exactly what changed — that's how a subtle bypass gets caught.)
3. **Human gate on logic changes** — mechanical moves can be agent-driven; any
   *logic* change to a detector needs human review before merge.
4. **Adversarial verification** — prove a fix still BLOCKS the known attack AND
   doesn't newly false-positive on benign traffic (use real samples).
5. **ReDoS / latency gate** — scanner is in the request path; a "cleaner" regex
   can be catastrophically slow. Check perf didn't regress.
6. **Coverage delta + decision log** — did test coverage drop on moved code?
   Record *why* any threshold/logic changed (auditable trail).

**Fail CLOSED, not open** is the cardinal rule: a detector must default to
"suspicious" on malformed input, never silently become a pass-through.

---

## 3. The toolchain (validated, cited)

| Job | Tool | Note |
|-----|------|------|
| Relocate file | `git mv` (pure-move commit) | preserves blame/history |
| Rewrite all imports | **LibCST `RenameCommand`** | rewrites imports + dotted refs across the tree; run in diff mode first |
| Sort imports | `ruff check --fix --select I` | review order-sensitive `__init__.py` (`# isort: skip`) |
| Verify no cycles | `import-linter` / `grimp` | gate each phase; re-grouping creates circular imports |
| Fast import check | `pytest --collect-only -q` | catches every broken import in seconds |
| Interactive fallback | `rope` | misses dynamic `try/except ImportError` — eyeball the diff |

**Do NOT use Bowler** (archived; Meta recommends LibCST). **Shim pattern:** PEP 562
module-level `__getattr__` + `warnings.warn(DeprecationWarning, stacklevel=2)` —
lazy (warns on access, not import). Delete all shims only at the **major bump**
(`v1.0.0`), after shipping `docs/MIGRATION_v1.md`.

### Top pitfalls
- **Circular imports** from bundling flat modules under one `__init__.py` — keep
  `__init__` thin, prefer absolute intra-package imports, gate with import-linter.
- **Codemods mangling `try/except ImportError`** (Na0S registers layers this way) —
  run codemod in diff mode, manually review every dynamic/string import.
- **Editable-install staleness** — `pip install -e .` after any phase that
  renames/adds modules, or you get phantom `ModuleNotFoundError`.
- **Missing `__init__.py`** → accidental implicit namespace package (PEP 420);
  add it in the same commit as the move (enable ruff `INP001`).
- **Mixing move + edit** defeats git's rename detection → lost blame.

---

## 4. Progress map — where the reorg actually stands

**~30% done. Top level = 86 files (target 9), but 67 are harmless shims; only
~10 real modules need moving + the structural renames.**

| # | Phase | Status | What's left |
|---|-------|--------|-------------|
| 1 | validation/ | ✅ DONE | (deviation: `multi_turn` went to `detectors/`, not `conversation/`) |
| 2 | output/ | ✅ DONE | — |
| 3 | structural/ | ✅ DONE | — |
| 4 | dataset/ | 🟡 PARTIAL | absorb `scripts/{trust_score,quarantine,near_duplicate,social_scraper,weekly_harvest}.py` lib code |
| 5 | eval/ | 🟡 PARTIAL | absorb `scripts/{evaluate_*,benchmark*,shadow_evaluate}.py` lib code |
| 6 | probes/ | ⬜ NOT STARTED | create, absorb `scripts/taxonomy/` (30+ modules) |
| 7 | agents/ | ⚠️ NAME COLLISION | `agents/` already built for the deploy-automation system; roadmap wanted it for `mcp_tool`+L19 stubs — **reconcile intent first** |
| 8 | taxonomy/ | ⬜ NOT STARTED | create, absorb `scripts/sync_taxonomy.py` |
| 9 | promote-fusion | ⬜ NEXT | move `_voting,signal_boost,evidence_grading,groundedness` → `fusion/` (4 files, dest exists, no collision — the warm-up; security-fixed in PR #404) |
| 10 | promote-rules | ⬜ NOT STARTED | ⚠️ wants `rules/` dir — **collides with phase 12** (`layer1→rules/`); sequence carefully |
| 11 | promote-detectors | ⬜ NOT STARTED | move `intent_guard,multilingual_handler,multilingual_intent` → `detectors/` (3 files) |
| 12 | rename layerN | ⬜ NOT STARTED | `layer0→input, layer1→rules, layer2→obfuscation, layer15→threat_intel, layer16→conversation` (the big visible win) |
| 13 | predict→_pipeline | ⬜ NOT STARTED | rename + caller updates |
| 14 | delete-shims | ⬜ NOT STARTED | purge 67 shims, ship `docs/MIGRATION_v1.md`, tag **v1.0.0** |

### The 3 traps to remember
1. **Phase 7 name collision** — `agents/` is taken; decide rename-vs-merge before touching it.
2. **Phase 10 ⇄ 12** — both want `rules/`; land the `layer1→rules` rename first OR coordinate.
3. **Phase 1 stray** — `multi_turn` needs a second hop to `conversation/` when phase 12 creates it.

### Recommended resume order
**Phase 9 (fusion)** is the ideal re-entry — 4 self-contained modules, destination
`fusion/` already exists, no collisions, and the code is already security-hardened
(PR #404). Then 11 (detectors), then the partials (4,5), then the bigger structural
moves (6,8,10,12), then 13, then 14 → v1.0.0.

---

## 5. Proof the loop works — the fusion pilot (PR #404)

Review-before-move on the 4 fusion modules found **5 real bugs**, all reproduced
by direct execution, all fixed with the gate proving detection never dropped:

| Bug | Was | Fix |
|-----|-----|-----|
| NaN `ml_prob` | → SAFE (fail OPEN) | sanitize to finite [0,1] at entry |
| `DECISION_THRESHOLD=nan` | universal fail-open | `_valid_threshold()` rejects nan/inf/out-of-(0,1] |
| score escaped [0,1] | negative → −1.8 | clamp every return path |
| duplicate hits | double-counted | dedup before accumulation |
| `evidence_grading` | **verified no-op** (graded rule NAME not span) | span-aware rebuild + 6 anti-evasion rules |

Detection efficacy identical before/after (TPR 0.8348, FPR 0.0000, F1 0.9100);
anti-evasion verified independently; 33 new tests + 278-test regression green.
A pre-existing RCE-in-fence gap was caught + noted in ROADMAP_V2.md (Layer 1 TODO).

---

## References
- Both source research reports (process + progress) are the basis for §§1-4.
- `ROADMAP_V2.md` lines 134-151 (migration sequence), 58-90 (target tree).
- LibCST `RenameCommand`, PEP 562, Django deprecation policy, import-linter.
