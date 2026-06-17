#!/usr/bin/env python3
"""Validate draft eval scenarios against the F14 schema.

The drafting boundary for threat-intel harvesting: scenarios proposed by the
``threat-intel-harvester`` agent (or a future ``extract_intel_scenarios.py``)
land in ``data/eval/scenarios/_drafts/`` and must load cleanly through
``ScenarioLoader`` before a human reviews and promotes them into ``v0.1/``.

This is a read-only validator — it never writes, promotes, or mutates anything.
It exits non-zero if any draft fails the schema, so it can guard a PR check.

Usage:
    python scripts/validate_draft_scenarios.py            # default _drafts dir
    python scripts/validate_draft_scenarios.py <dir>      # any scenarios dir
"""

from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_DIR = Path("data/eval/scenarios/_drafts")


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else DEFAULT_DIR

    if not target.exists():
        print(f"[validate-drafts] {target} does not exist yet — nothing to validate.")
        return 0

    yaml_files = sorted(p for p in target.glob("*.yaml") if p.name != "README.md")
    if not yaml_files:
        print(f"[validate-drafts] no draft scenarios in {target} — nothing to validate.")
        return 0

    # Imported lazily so the "empty dir" fast paths don't require the package.
    from na0s.eval.scenarios import load_scenarios_dir

    try:
        scenarios = load_scenarios_dir(str(target))
    except Exception as exc:  # noqa: BLE001 — surface the schema error verbatim
        print(f"[validate-drafts] FAIL — {target} did not load: {exc}")
        return 1

    print(f"[validate-drafts] OK — {len(scenarios)} draft scenario(s) in {target}:")
    for s in scenarios:
        provenance = "ok" if s.source == "harvest_pipeline" else f"source={s.source!r}"
        sid = (s.stable_id or "")[:8]
        print(f"  - {s.name}  [{s.attack_category}/{s.type.value}]  {provenance}  id={sid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
