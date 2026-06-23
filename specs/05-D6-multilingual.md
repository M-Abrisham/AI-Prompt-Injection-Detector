# Spec: D6 — Multilingual Injection (attack-type coverage, item 05)

## Goal
Make D6 (Multilingual Injection) **measurable** and close its real recall gaps FP-safely. The detector already exists and is wired (`multilingual_handler.py`, `multilingual_intent.py`, `_HAS_MULTILINGUAL`, the D6 block at `predict.py:916`, `_MULTILINGUAL_FORCE_HITS`) and catches common direct overrides (FR/DE/ZH "ignore all previous instructions" all flag) — but there are **zero D6 eval scenarios**, so recall is unmeasured, and the harder cases are likely uncovered. Local, no API key.

## Current state (verified on main `b8cffcc`)
- Detector present + wired in `predict.py` (force-hits override/extraction latin+cjk; `get_multilingual_rule_weight`). Common FR/DE/ZH direct overrides flag (risk 0.74–1.0).
- **No D6 scenarios** in `data/eval/scenarios/` → recall unmeasured; the audit's "weakest detector / zero acquired data" is a *measurement* gap, not (only) a detector gap.
- **Possible cascade parity gap**: `grep multilingual cascade.py` = NONE (confirm whether D6 reaches the cascade path via the rules/L0 fold or is genuinely absent).

## Applicable template steps
1–4 (explore detector + measure ACROSS difficulty, find the REAL gaps, plan, harden ONLY genuine gaps FP-safely + ensure cascade parity), 5 (harvest/synthesize D6 scenarios — translate/paraphrase canonical injection techniques into multiple languages incl. low-resource + transliteration + code-switching; decontam by construction; paired benign multilingual siblings), 6 (recall + FP tests), 7 (cleanup), 8 (roadmap), 9 (coverage/benchmark), 10 (taxonomy D6.x + COVERAGE_MATRIX + scorer), 11 (PR). Q&A #1 (can Na0S catch D6 — across languages), #3 (cascade parity), #10 (scorer), #11 (predict/cascade refs), #12 (harvester).

## Scope
- **Probe broadly** to find the ACTUAL recall gaps: low-resource languages (not just FR/DE/ZH), **transliteration/romanization** (e.g. Hindi/Arabic/Russian in Latin script), **code-switching** (EN + L2 mid-sentence), and subtle phrasings beyond "ignore all previous instructions". Do NOT assume the detector is broken — measure.
- **Synthesize D6 scenarios** across ≥6 languages × difficulty, paired with benign multilingual siblings (legit non-English requests that share vocabulary). Decontaminate vs v0.1 + training.
- **Harden FP-safely** only the gaps found (transliteration normalization / additional language anchors) — capped weight, no arbitrary thresholds, benign multilingual must NOT regress (FP-safe is the hard constraint — non-English ≠ malicious).
- **Cascade parity**: if D6 is genuinely absent from the cascade path, wire it behind `_HAS_MULTILINGUAL` mirroring the sibling detectors.
- Recall + FP test; honest `xfail` for any residual gap (no weakening, no cap-raising).

## Definition of done
D6 recall is measured against new scenarios; genuine gaps hardened FP-safely; cascade parity confirmed/added; recall+FP test green (honest xfails allowed); coverage matrix updated; full suite green; PR open. No API key. No merge without approval.

## Constraints
FP-safe is paramount — benign multilingual content must stay below threshold. Hot files (`predict.py`/`cascade.py`) edited only in this worktree, scoped commits. Local/no-API. detector-authoring + intel-harvest/eval-scenario-curation discipline.
