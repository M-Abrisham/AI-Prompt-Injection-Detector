# Draft scenarios — review staging area

This directory is the **hand-off point** between threat-intel harvesting and the
live F14 evaluation library. It is **not** part of the gated/promoted set.

## Flow
```
threat intel (advisory / dataset / na0s.layer15 sync output)
  └─ threat-intel-harvester agent (or scripts/extract_intel_scenarios.py, when built)
       └─ DRAFT scenario YAML lands HERE  (data/eval/scenarios/_drafts/)
            └─ validate:  python scripts/validate_draft_scenarios.py
                 └─ human review + decontamination + (future) f14 promotion gate
                      └─ promote into  data/eval/scenarios/v0.1/   (via PR)
```

## Rules
- Drafts here are **candidates only**. Nothing in `_drafts/` is loaded by the
  promotion gate or training pipeline.
- Every draft must carry provenance: `source: harvest_pipeline` and the origin
  URL + retrieval date in `description` (see the `eval-scenario-curation` skill).
- Promotion to `v0.1/` is a **human-reviewed PR**, never automatic. Decontamination
  (SHA-256 + semantic) is the gate's job, not the drafter's.
- File naming: `<YYYY-MM-DD>-<source-slug>.yaml` (e.g. `2026-06-17-atlas-aml-t0051.yaml`).

## Validate before review
```
python scripts/validate_draft_scenarios.py            # checks this dir
python scripts/validate_draft_scenarios.py <path>     # checks any scenarios dir
```
Exits non-zero if any draft fails the F14 schema, so it can guard a PR check.
