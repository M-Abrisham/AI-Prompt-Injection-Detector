# Attack Spec: Character-Split Evasion (D7.5)

| Field | Value |
|-------|-------|
| **Attack** | Character-split / token-smuggling evasion |
| **Taxonomy** | D7.5 (payload delivery / token smuggling); related A1.1 |
| **Matrix row** | D7 payload-delivery (no standalone INJ row; folds into D7) |
| **Skills** | `detector-authoring`, `na0s-review-checklist`, `eval-harness` |
| **Branch** | `hardening/char-split-evasion` |
| **Status** | ✅ **DONE** (2026-06-22) — exemplar for the spec format |

Attackers split a word into single characters so the ML vocabulary is
destroyed and keyword rules miss it: `i.g.n.o.r.e`, `i_g_n_o_r_e`,
`i,g,n,o,r,e`, `i=g=n`, em-dash, double-dash, vertical newline stacks.

---

## To-do list (GENERAL Prompt, per item)

- [x] **1. Explore current rules + find gaps/edge-cases.** `_reassemble_char_splits`
  handled only space + dot; 5/8 (then 14) variants evaded — dot-glued,
  underscore, comma, interpunct, vertical, em/en-dash, `= # @ \ ^ )`,
  double-dash. Edge cases: list-FP (`a, b, c`), URL paths, snake_case,
  `\n` escape runs, ReDoS.
- [x] **2. Read roadmap/taxonomy/matrix/source.** Roadmap L0 §3035-3037
  (stale: listed reassembly as TODO though space/dot existed); matrix D7
  delivery 64%; `normalization.py`, `predict.py`, `cascade.py`.
- [x] **3. Root-cause plan.** (a) generalize reassembly to any *consistent
  pure-punctuation* separator + vertical pass; (b) the real recall fix —
  wire the (previously inert) `char_level_reassembly` flag into composite
  scoring, since glued reassembly matches no word-boundary rule.
- [x] **4. Implement + wire (parity).** Pass 3 backref regex + Pass 4
  vertical in `normalization.py` (returns `max_run` magnitude → `char_level_
  reassembly` + `_heavy` flags); scoring block in BOTH `predict.py` and
  `cascade.py` (+0.20 / +0.45+floor when heavy).
- [x] **5. Datasets → isolated tests.** No new corpus needed; covered by
  synthetic variant battery (reassembly is deterministic normalization).
- [x] **6. Test cases (code + use-case).** `tests/input/test_char_split_
  reassembly.py` (21 cases incl. FP guards) + `TestD7_5_ExtendedCharSplit`.
- [x] **7. File/dir cleanup + refactor.** Replaced enumerated separator list
  with one generalized regex; no clutter.
- [x] **8. Update roadmap.** §3035/3036 checked off, 3037 superseded, residuals filed.
- [ ] **9. README/benchmark.** N/A — input-normalization hardening, not a new INJ row / measured number.
- [x] **10. Taxonomy + matrix + threshold.** No new taxonomy code; D7 row note. Threshold: +0.20/+0.45 capped, heavy floors to decision threshold.
- [ ] **11. Open PR (held-out gate).** PENDING full-suite green.

---

## FP-safety (empirical, the binding constraint)

- Separator excludes **whitespace** (protects `a, b, c` lists) and **`/ | < > \`**
  (protects URL paths, regex `a|b|c`, CSS `a > b > c`, `\n` escapes) — each
  exclusion driven by an *observed* FP on a 30k real-world corpus, not guessed.
- Net benign fire rate: **2/30000 = baseline; both hits are actual attacks → 0 added FPs.**
- ReDoS-safe: `<9ms` on 50k chars (bounded `{1,3}` + backref).

## Documented residuals (left uncovered to preserve FP-safety)

Whitespace-bearing separators (`i . g`, tab), `/ | < > \`, per-character-mixed separators (`i.g_n-o,r`).

## Q&A verification

1. **Can Na0S catch it?** Yes — 3/8 → all realistic variants BLOCK (verified via `scan()`).
2. **Cleanup done?** Yes — generalized regex, no duplicate L2 flag.
3. **Pipeline wired?** Yes — `predict.py` + `cascade.py` parity.
4. **Tested (code + use-case)?** Yes — 21 unit + 6 scan tests; prior full suite 9682 passed.
5. **Harvester?** N/A — deterministic normalization, no training corpus.
6. **Taxonomy/matrix match, no dupes?** D7.5 existing; no new code.
7. **Scorer correct?** Bounded contribution + heavy floor; FP-safe.
8. **predict.py/cascade.py refs?** Yes (`char_split_obfuscation` hit).
9. **Harvester agent harvests it?** N/A.
