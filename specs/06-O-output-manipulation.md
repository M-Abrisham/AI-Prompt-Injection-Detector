# Spec: O — Output Manipulation (attack-type coverage, item 06)

## Goal
Make O (Output Manipulation) **measurable** and close its real recall gaps FP-safely. O is **output-side**: it is detected via `scan_output()` / `OutputScanner` (the `output/` package), NOT the input `predict.py`/`cascade.py` path — so "predict/cascade refs of O" is correctly N/A (do not force a parity edit). The scanner is mature and wired (`scan_output` is a public API), but there are **zero O eval scenarios**, so recall is unmeasured. Local, no API key.

## Current state (verified on main `b8cffcc`)
- Output scanner present + wired: `src/na0s/output/{scanner,propagation,attribution,segment_grader,streaming,dual}.py`, public `na0s.scan_output()` (`__init__.py:82`), `OutputScanResult`.
- O taxonomy has 11 techniques (O1.1–O1.5, O2.1–O2.6).
- **No O scenarios** in `data/eval/scenarios/` → recall unmeasured.
- Probe via `scan_output` (uncalibrated 0.55): markdown-image exfil → **1.0 (caught)**; benign code w/ legit URL → **0.0 (clean, good FP-safety)**; XSS `<img onerror=…>` → **0.5 (borderline MISS)**; hidden-instruction HTML-comment in output → **0.0 (MISS)**. So there are genuine gaps (output XSS, hidden instructions in output).

## Applicable template steps
1–4 (explore the output scanner + measure across O1.x/O2.x, find REAL gaps, plan, harden ONLY genuine gaps FP-safely), 5 (synthesize O scenarios = malicious LLM OUTPUTS: XSS/script, markdown-image data-exfil, hidden HTML-comment instructions, data leakage, unsafe links/code — paired with benign LLM outputs that share surface: legit code, legit markdown images, legit links), 6 (recall + FP test via `scan_output`), 7 (cleanup), 8 (roadmap), 9 (coverage/benchmark), 10 (taxonomy O.x + COVERAGE_MATRIX + scorer/sensitivity), 11 (PR). Q&A #1 (can Na0S catch O — via scan_output), #10 (scorer/sensitivity), #12 (harvester). **Q&A #11 (predict/cascade refs) is N/A — O is output-side by design; state that, don't add a bogus parity edit.**

## Scope
- **Measure** `scan_output` recall across the 11 O techniques (find the right `OutputScanResult` field for the verdict first). Probe XSS variants, markdown/HTML exfil, hidden-instruction-in-output, data leakage, unsafe-link/code emission.
- **Harden FP-safely** only the gaps found (e.g. output XSS event-handler / `<script>`; hidden HTML-comment AI-directed instructions in output) in the `output/` package — capped, no arbitrary thresholds, and **benign LLM outputs must NOT regress** (legit code blocks, legit markdown images to trusted hosts, legit links). FP-safe is the hard constraint.
- **Synthesize O scenarios** (malicious outputs + paired benign outputs), decontaminate, tag canonical O1.x/O2.x.
- Recall + FP test driven by `scan_output`; honest `xfail` for residual gaps (no weakening, no sensitivity-cranking).

## Definition of done
O recall measured via `scan_output`; genuine gaps hardened FP-safely; recall+FP test green (honest xfails allowed); coverage matrix updated; full suite green; PR open. No API key. No merge without approval.

## Constraints
FP-safe paramount — benign LLM output (code, markdown, links) must stay clean. Output-side only (no predict/cascade parity). Scoped commits. Local/no-API. detector-authoring + eval-scenario-curation discipline.
