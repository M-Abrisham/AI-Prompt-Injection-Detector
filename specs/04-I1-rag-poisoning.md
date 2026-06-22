# Spec: I1 — Data Source / RAG Poisoning (attack-type coverage, item 04)

## Goal
Make Na0S correctly **score and detect** I1 (Data Source Poisoning: instruction/authority/exfil payloads embedded in retrieved RAG context, documents, email, vector/knowledge DBs). Root-cause fix the dead RAG-poison scorer and close the predict/cascade parity gap, then wire in the harvested I1 eval coverage. Absorbs the long-pending **rag_poison fold-in** (formerly spec-01 Item B). Fully local, no API key.

## Root cause (verified on main `208db9e`)
1. **Dead scorer in `predict.py`.** `rag_poison_weight = get_rag_poison_weight(rag_result)` is computed at `predict.py:1201` but **never added to `composite`** — unlike its siblings `inter_model_weight` (`predict.py:1242` `composite = min(composite + inter_model_weight, 1.0)`) and `tool_abuse_weight`. The `rag_poison:*` hits fire, but the calibrated weight is discarded → I1 is systematically under-scored.
2. **No cascade parity.** `cascade.py` wires `inter_model` (`:185-186,639-648`) and `tool_abuse` (`:192-195,661-669`) behind `_HAS_*` flags, but has **zero** rag_poison references → the RAG-poison detector never runs in the cascade path.

## Applicable template steps (attack-detection feature — most apply)
1–4 (explore I1/rag surface + scorer, gaps, plan, implement+WIRE into predict.py AND cascade.py behind `_HAS_RAG_POISON` with capped weight), 5 (harvester audit — I1 drafts already harvested in **PR #454**: move a vetted subset into an isolated I1 test scenario file), 6 (tests: recall on I1 scenarios + FP-safe paired-benign), 7 (cleanup), 8 (roadmap), 9 (coverage/benchmark), 10 (taxonomy I1.x + COVERAGE_MATRIX + scorer threshold/cap), 11 (PR). Q&A #1 (can Na0S catch I1 — full-scan after fix), #3 (pipeline wiring parity), #10 (scorer), #11 (predict/cascade refs), #12 (harvester) all central.

## Scope
- **predict.py**: add the missing "wire `rag_poison_weight` into composite scoring" block, mirroring the inter_model block exactly (cap already enforced inside `get_rag_poison_weight`; a lone hit stays a soft signal).
- **cascade.py**: add `_HAS_RAG_POISON` optional-import + a wiring block mirroring `tool_abuse`/`inter_model` (parity).
- Reconcile the top-level `rag_poison_detector.py` vs `rag/poison_detector.py` (confirm which is canonical; the other must be a shim, not a divergent copy).
- Verify `get_rag_poison_weight` cap (≤0.30, sibling parity) — no arbitrary magic; in config if needed.
- Tests: a `tests/` I1 recall test (the harvested I1 scenarios should now flag) + FP-safe (paired benign siblings stay below threshold). Mirror source tree.
- Coverage matrix + taxonomy: confirm I1.x rows reflect the now-wired scorer; no duplicate/non-canonical I1 codes.

## Definition of done
RAG-poison weight applied in BOTH predict.py and cascade.py behind `_HAS_RAG_POISON`; I1 recall test green + FP-safe; coverage matrix updated; full suite green; PR open. No API key. No merge without approval.

## Constraints
Hot files (`predict.py`/`cascade.py`) — edit only in this worktree, scoped commits. FP-safe (benign RAG/summarization siblings must NOT regress). Local/no-API. detector-authoring discipline (parity + capped weight + paired bounds).
