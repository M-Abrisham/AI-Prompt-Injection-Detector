#!/usr/bin/env python3
"""Generate JSONL samples from all registered taxonomy probes.

Usage:
    python scripts/generate_taxonomy.py --output data/staging/taxonomy_samples.jsonl
    python scripts/generate_taxonomy.py --category D1 D5 --output out.jsonl
    python scripts/generate_taxonomy.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Path setup — make scripts/ and project root importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _script_dir)
sys.path.insert(0, os.path.join(_project_root, "src"))

from taxonomy import ALL_PROBES
from na0s.dataset.schema import DataLabel, Na0SSample


def _generate_samples(probe_classes, categories=None):
    """Yield Na0SSample instances from each probe's generate() output.

    Parameters
    ----------
    probe_classes : list
        List of Probe subclasses (from ALL_PROBES).
    categories : set[str] or None
        If given, only include probes whose category_id is in this set.
    """
    for ProbeClass in probe_classes:
        if categories and ProbeClass.category_id not in categories:
            continue

        probe = ProbeClass()
        raw_samples = probe.generate()

        for item in raw_samples:
            if len(item) == 3:
                text, technique_id, meta = item
            else:
                text, technique_id = item[:2]
                meta = {}

            sample = Na0SSample(
                text=text,
                label=DataLabel.INJECTION,
                augmentation_type=meta.get("augmentation_type"),
                technique_id=technique_id,
                source="taxonomy_probe",
                source_id=probe.category_id,
                language=meta.get("language", "en"),
                difficulty=meta.get("difficulty"),
                license=meta.get("license"),
                license_url=meta.get("license_url"),
                attribution=meta.get("attribution"),
            )
            yield sample


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate JSONL samples from taxonomy probes"
    )
    parser.add_argument(
        "--output",
        default="data/staging/taxonomy_samples.jsonl",
        help="Output JSONL file path (default: data/staging/taxonomy_samples.jsonl)",
    )
    parser.add_argument(
        "--category",
        nargs="*",
        default=None,
        help="Filter to specific probe category IDs (e.g. D1 D5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count samples without writing output",
    )
    args = parser.parse_args(argv)

    categories = set(args.category) if args.category else None

    count = 0
    if args.dry_run:
        for _ in _generate_samples(ALL_PROBES, categories):
            count += 1
        print("Dry run: {} samples would be generated".format(count))
        return count

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for sample in _generate_samples(ALL_PROBES, categories):
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
            count += 1

    print("Wrote {} samples to {}".format(count, args.output))
    return count


if __name__ == "__main__":
    main()
