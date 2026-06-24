# Spec: I2 — HTML/Markup Injection (attack-type coverage, item 07)

## Goal
Make I2 (HTML/Markup Injection: instructions hidden in `<!-- comments -->`, `display:none`/`hidden` divs, invisible-text CSS within INPUT) **measurable** and confirm/close any FP-safe gaps. The detector exists and is wired (`src/na0s/input/html_extractor.py`; flags `hidden_html_content`→I2.1, `suspicious_html_comment`→I2.2, … mapped at `predict.py:1650`) and already catches the obvious cases, but there are **zero I2 eval scenarios**, so recall is unmeasured. Local, no API key.

## Current state (verified on main `b8cffcc`)
- Detector present + wired: `input/html_extractor.py` → predict.py:1650 hit→I2.x map. Probe: hidden-div / HTML-comment / `<span hidden>` injections all flag (1.0 / 0.70 / 1.0).
- Benign HTML/markup in input stays clean (legit `<div>` w/ comment 0.0001, CSS `.hidden{display:none}` 0.12, `<span style>` 0.03) — good FP baseline.
- I2 taxonomy = I2.1 Hidden-div (high), I2.2 HTML-comment (medium), I2.3 Invisible-text-CSS (high).
- **No I2 scenarios** → recall unmeasured.

## Applicable template steps
1–4 (explore html_extractor + measure across I2.1/.2/.3, find any REAL gaps, plan, harden ONLY genuine gaps FP-safely — likely minimal since the detector works), 5 (synthesize I2 scenarios = inputs with hidden-div/HTML-comment/invisible-CSS injections + paired benign HTML/markup siblings: legit code snippets, real comments, styled content), 6 (recall + FP test), 7 (cleanup), 8 (roadmap), 9 (coverage/benchmark), 10 (taxonomy I2.x + COVERAGE_MATRIX + scorer), 11 (PR). Q&A #1 (catch I2), #3 (cascade parity — confirm html_extractor reaches cascade or note if predict-only), #10 (scorer), #11 (predict/cascade refs), #12 (harvester).

## Scope
- **Measure** html_extractor recall across I2.1/I2.2/I2.3 incl. harder variants (zero-width/CSS `font-size:0`/`opacity:0`/`color:#fff`/off-screen positioning, comment-nested instructions, attribute-smuggled text).
- **Synthesize I2 scenarios** (hidden HTML instructions + paired benign HTML siblings), decontaminate, tag canonical I2.1/I2.2/I2.3.
- **Harden FP-safely** only genuine gaps found (capped, no arbitrary thresholds) — benign HTML/markup (code blocks, legit comments, normal CSS) MUST stay clean. FP-safe is the hard constraint.
- Confirm cascade parity (does html_extractor run in the cascade path? if predict-only, note or add parity).
- Recall + FP test; honest `xfail` for residual gaps (no weakening).

## Definition of done
I2 recall measured; any genuine gaps hardened FP-safely; recall+FP test green (honest xfails allowed); coverage matrix updated; full suite green; PR open. No API key. No merge without approval.

## Constraints
FP-safe paramount — benign HTML/markup must stay clean. Scoped commits (hot file predict.py only in this worktree if touched). Local/no-API. detector-authoring + eval-scenario-curation discipline.
