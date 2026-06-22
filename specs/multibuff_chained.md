# Attack Spec: Multi-Buff / Chained Obfuscation (MB, D4/D5 decode-and-rescan)

| Field | Value |
|-------|-------|
| **Attack** | Multi-buff / chained obfuscation — 2–3 stacked transforms (Base64, ROT13, Caesar, Leet, Reverse, Fullwidth, ZeroWidth, Homoglyph, CaseAlternating) |
| **Taxonomy** | MB (MB1.1–MB1.10 two-buff, MB2.1–MB2.5 three-buff); related D4 (encoding) + D5 (unicode) + CT (combo techniques) |
| **Matrix row** | Evasion-modifier axis "Multi-buff stacking" (`COVERAGE_MATRIX.md` line 92) — **33% measured, OPEN GAP**. No standalone INJ row. |
| **Skills** | `detector-authoring`, `na0s-review-checklist`, `eval-harness`, `detector-failure-analysis` |
| **Branch** | TBD |
| **Status** | 🔴 **OPEN** — root cause confirmed empirically; fix already scoped in ROADMAP_V2.md §3043 "L2 Chained Obfuscation Decoder" (not started) |

Attackers chain ≥2 obfuscation transforms so that no single decoder's
keyword gate ever fires. The recursive unwrapper (`obfuscation_scan`,
`max_depth=4`) is *structurally* correct — it re-scans every decoded view —
but each cipher decoder (`_is_rot13_candidate`, `_caesar_brute_force`,
`_is_reversed_candidate`, `_is_leetspeak_candidate`, Pig Latin, Morse,
numeric) only **emits a decoded view if the decode of *that one layer*
already contains attack keywords**. When the inner transform is itself an
obfuscation (no plaintext keywords yet), the outer gate sees noise, refuses
to peel, and the chain is never unwrapped.

---

## To-do list (GENERAL Prompt, per item)

- [ ] **1. Explore current rules + find gaps/edge-cases.** `obfuscation_scan`
  recurses correctly (`_recurse`, depth-4, cycle-detect, 10× expansion cap),
  but the gate is in `_scan_single_layer`: ROT13/Caesar/Reverse/Leet/PigLatin/
  Morse/numeric each call `_has_attack_keywords()` on **their own single-layer
  output** before emitting a decoded view. A multi-buff chain whose outer layer
  is one of these self-gated transforms never peels, so the recursion never
  reaches the inner payload. **Empirically confirmed misses (probe order =
  `_chain(text, INNER, OUTER)` → OUTER applied last):** MB1.7 `Leet(ROT13)`
  risk 0.005; MB1.10 `ROT13(Reverse)` risk 0.005; MB1.1 `ROT13(Base64)`
  risk 0.364; reversed-base64 risk 0.364 — all BELOW 0.55. Edge cases: buff
  *order* matters (base64-OUTER chains ARE caught — see step 2); english-
  plausibility gate needed to avoid a decode explosion; ReDoS / perf budget on
  cartesian re-decode; FP on benign nested encodings (e.g. base64 of a JSON
  config).
- [ ] **2. Read roadmap/taxonomy/matrix/source.** Roadmap §3043 "L2 — Chained
  Obfuscation Decoder (4d)" already scopes the fix (cross-decoder loop, English
  plausibility scorer, perf budget, promote `test_d4_rot13_plus_leet` /
  `test_d4_leet_plus_pig_latin` xfails) — **not started**. Roadmap §1720
  flags "D4 chained multi-buff (Base64(ROT13))" as a genuine gap (P2 doc-drift).
  Taxonomy: MB block present (`taxonomy.yaml` lines 725–752), 10 two-buff +
  5 three-buff, no duplicates. Matrix line 92/96 documents the 33%/0% split and
  the root cause accurately. **Stale-path flag:** matrix lines 83/96 still point
  at `src/na0s/layer2/obfuscation.py`; the file is at
  `src/na0s/obfuscation/obfuscation.py` post-rename — fix when next syncing.
- [ ] **3. Root-cause plan.** Two distinct fixes, both needed:
  **(a) Decode-emission gate is too strict for chains.** Allow a cipher
  decoder to emit its decoded view for *recursion* even when that view has no
  keywords yet, **provided** the view passes an English-plausibility OR
  high-decode-confidence check (so the recursion can try the next transform).
  Keep the *flag-raising* keyword gate where it is, only loosen the
  *recurse-into* path. This is exactly Roadmap §3043's "English plausibility
  scorer" (`_is_plausible_english()`: KL < 0.8 OR dict-hit-rate > 0.4).
  **(b) Wire the dead chain-boost.** `obfuscation_scan` already computes
  `combined_boost`/`combined_reasons` via `_analyze_encoding_chain` (depth +
  diversity, capped 0.20) and `max_depth_reached`/`encoding_chains` — but
  **predict.py (line 855) and cascade.py (line 527) read ONLY
  `obs["evasion_flags"]` and discard `combined_boost`.** Confirmed dead by grep:
  `combined_boost`/`combined_reasons`/`encoding_chains` have ZERO consumers in
  `src/na0s/` outside the obfuscation module. This is the same class of bug as
  the historic computed-then-discarded `rag_poison_weight`.
- [ ] **4. Implement + wire (parity).** Loosen the recurse-into emission gate in
  `_scan_single_layer` behind `_is_plausible_english()`; add
  `NA0S_MAX_CHAIN_DECODES` (50) + `NA0S_CHAIN_DECODE_TIMEOUT_MS` (200) budgets;
  consume `combined_boost` in BOTH `predict.py` and `cascade.py` (capped,
  additive, parity — mirror the existing obfuscation-flag handling). **Files**:
  `src/na0s/obfuscation/obfuscation.py`, `src/na0s/predict.py`,
  `src/na0s/cascade.py`.
- [ ] **5. Datasets → isolated tests.** Probe generator
  `scripts/taxonomy/multi_buff_samples.py` already builds MB1.1–MB2.5 (91
  samples) via `_chain(text, *buffs)` using `_buffs.py`. **No F14 eval scenarios
  for MB exist on disk** (no `attack_category: MB` YAML). The harvester does NOT
  harvest MB. Add: (i) deterministic synthetic battery (chain compose is
  reproducible) + (ii) a handful of F14 MB scenarios with benign nested-encoding
  siblings for FPR. Run isolated to avoid `FingerprintStore` contamination
  (`L0_FINGERPRINT_STORE=:memory:`, `SCAN_TIMEOUT_SEC=0`).
- [ ] **6. Test cases (code + use-case).** Promote `@expectedFailure` on
  `test_d4_rot13_plus_leet` and `test_d4_leet_plus_pig_latin`; add
  `tests/.../test_obfuscation_chaining.py` covering MB1.1/1.7/1.10 +
  Leet(ROT13)/Reverse(B64)/ROT13(Base64) with paired benign nested-encoding
  guards. Measure recall via the two-sided harness (`scripts/technique_analysis.py`).
- [ ] **7. File/dir cleanup + refactor.** No new module — extend existing
  `obfuscation.py`. Extract the english-plausibility check from the inline
  KL/composite logic into one reusable `_is_plausible_english()`.
- [ ] **8. Update roadmap.** Check off §3043 items as landed; update matrix
  line 92 from 33%→measured-after; fix the two `layer2/obfuscation.py` stale
  paths (lines 83, 96); resolve §1720 doc-drift note.
- [ ] **9. README/benchmark.** Re-measure the "Multi-buff stacking" matrix row
  via `na0s.scan` (the doc states it was measured this way, not via the harness);
  record the post-fix number with its basis. Update any README evasion-coverage
  claim only after measurement, not estimate.
- [ ] **10. Taxonomy + matrix + threshold.** MB taxonomy is complete and
  duplicate-free — **no taxonomy change needed**. Matrix: keep MB on the
  evasion axis (no fabricated INJ row). Threshold discipline: the chain-boost
  cap (0.20) and the new decode budgets (50 / 200ms) are arbitrary — gate them
  against measured FPR, do not hardcode-and-forget (na0s-review-checklist:
  arbitrary-threshold flag).
- [ ] **11. Open PR (held-out gate).** PENDING — branch TBD; requires full-suite
  green + benign-FPR non-regression on the F14 promotion gate before opening.

---

## FP-safety (the binding constraint)

The fix loosens a gate, so FP risk is the central concern:

- **Loosening must be recursion-only.** A decoded view is allowed to *recurse*
  on plausibility alone, but a *flag/score* is still only raised when the final
  unwrapped view satisfies the keyword gate. Benign nested encodings
  (`base64(json_config)`, a base64'd recipe, a reversed string of prose) decode
  to plausible-but-keyword-free text → they recurse, find no inner attack, and
  **emit no flag**. This mirrors the existing hex/`0x` keyword gate that keeps
  `tests/test_false_positives.py` at 71/0.
- **The `combined_boost` is additive and capped at 0.20** — it cannot alone push
  a benign input over 0.55 (it only compounds when real attack flags also fire).
  Validate this empirically; do not assume.
- **Decode-explosion / perf is a DoS-FP risk:** without `NA0S_MAX_CHAIN_DECODES`
  (50) + timeout (200ms) the loosened gate could fan out combinatorially on
  adversarial input. Add a `<500ms on 500-char input` perf regression test.
- **Historic lesson (memory):** an *ungated* decoder once surfaced benign
  decoded views that flipped a sibling scan via the `FingerprintStore` floor.
  Keep the relanded decoder **stateless** and never auto-register a
  plausibility-only (keyword-free) decoded view.

## Documented residuals (left uncovered to preserve FP-safety)

- Unicode-only chains where BOTH buffs are non-decoding cosmetic transforms
  (Fullwidth+ZeroWidth, Homoglyph+CaseAlternating) — already ~100% via NFKC +
  homoglyph fold per the matrix, not via this decoder path.
- Chains whose intermediate view is *plausible English that is itself benign*
  but wraps a third encoding beyond `max_depth=4` / the 50-decode budget.
- Caesar/PigLatin chains gated by the 370k-word dictionary load failing
  silently (degrades to keyword-only path) — orthogonal, pre-existing.

## Q&A verification

1. **Can Na0S catch it?** **Partial.** Empirically: base64-OUTER chains are
   caught incidentally (full-blob base64 decode is *not* keyword-gated, so the
   intermediate surfaces and the inner ROT13 gate then fires — e.g. plain
   `base64(rot13(...))` blob → risk 1.0). But the MB **probe order** puts the
   self-gated cipher OUTERMOST: MB1.1 ROT13(Base64)=0.364, MB1.7 Leet(ROT13)=0.005,
   MB1.10 ROT13(Reverse)=0.005, reverse(base64)=0.364 — all MISS. Matrix's
   "33% overall / 0% on keyword-gated inner-transform chains" is accurate.
2. **Cleanup done?** N/A (open) — plan is to extend `obfuscation.py`, no new module.
3. **Pipeline wired?** **Partially, and a key signal is dead.** `obfuscation_scan`
   is called by predict.py (L854) + cascade.py (L526), but both consume only
   `evasion_flags`; `combined_boost`/`encoding_chains` (the multi-buff signal)
   are computed-then-discarded. Wiring them is part of the fix.
4. **Tested (code + use-case)?** No green tests — two relevant cases sit as
   `@expectedFailure` (`test_d4_rot13_plus_leet`, `test_d4_leet_plus_pig_latin`).
5. **Harvester?** **No.** No MB references in `src/na0s/eval/harvest/` or
   `weekly_harvest.py`. Only the synthetic probe generator
   `scripts/taxonomy/multi_buff_samples.py` (91 samples) + taxonomy entries.
   No `attack_category: MB` F14 scenarios on disk → datasets are synthetic-only.
6. **Taxonomy/matrix match, no dupes?** Yes — MB1.1–MB2.5 present and unique in
   `taxonomy.yaml`; matrix carries one MB evasion-axis row (no duplicate, no
   fabricated INJ row). **Stale path** `layer2/obfuscation.py` in matrix lines
   83/96 should be `obfuscation/obfuscation.py`.
7. **Scorer correct?** The scorer (`_analyze_encoding_chain`) is correct and
   capped (0.20) but **not consumed** — it must be wired into the composite in
   predict.py + cascade.py with parity.
8. **predict.py/cascade.py refs?** Yes — `from .obfuscation import obfuscation_scan`
   in both (predict.py L79/L854, cascade.py L35/L526), reading `evasion_flags` only.
9. **Harvester agent harvests it?** No — MB is out of the harvester's tagging
   vocabulary; add MB to discovery/tagging + author F14 MB scenarios with benign
   nested-encoding siblings.
