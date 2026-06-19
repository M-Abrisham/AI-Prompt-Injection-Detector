#!/usr/bin/env python3
"""Extract benchmark facts from the committed harness artifacts into docs/benchmarks.yaml.

The README ## Benchmarks section and BENCHMARK_RESULTS.md source their numbers
from this file rather than hand-typing them — the same discipline scripts/
extract_facts.py applies to structural facts. Every value here is derived from
benchmarks/results/technique_analysis.json (the two-sided recall harness) and
benchmarks/results/bm1/*.json (external datasets), keyed by the artifact's
git SHA so a stale doc is detectable.

Percentages are rounded to one decimal so the displayed forms (e.g. "57.1%")
are exact, drift-checkable tokens. Regenerate with `make benchmark-facts`;
`scripts/check_readme_drift.py` then fails CI if the README cites a benchmark
number that is not in this file.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "benchmarks" / "results"
OUTPUT_PATH = REPO_ROOT / "docs" / "benchmarks.yaml"


def _load(rel: str) -> dict:
    return json.loads((RESULTS / rel).read_text())


def pct(x) -> float:
    return round(x * 100, 1)


def pct_pair(pair):
    lo, hi = pair
    return [pct(lo), pct(hi)]


def build_benchmarks() -> dict:
    ta = _load("technique_analysis.json")
    s = ta["summary"]

    headline = {
        "malicious_recall_pct": pct(s["overall_malicious_recall"]),
        "malicious_recall_ci_pct": pct_pair(s["overall_malicious_recall_ci"]),
        "malicious_detected": s["malicious_detected"],
        "malicious_total": s["malicious_total"],
        "benign_fpr_pct": pct(s["overall_benign_fpr"]),
        "benign_fpr_ci_pct": pct_pair(s["overall_benign_fpr_ci"]),
        "benign_false_positives": s["benign_false_positives"],
        "benign_total": s["benign_total"],
        "evasion_detection_pct": pct(s["overall_evasion_detection_rate"]),
        "evasion_detected": s["evasion_detected"],
        "evasion_total": s["evasion_total"],
        "metrics_note": s.get("metrics_note", ""),
    }

    per_category = {
        cid: {"recall_pct": pct(v["recall"]), "detected": v["detected"], "total": v["total"]}
        for cid, v in sorted(ta["per_category"].items(), key=lambda kv: kv[1]["recall"])
    }
    per_evasion = {
        name: {"detection_pct": pct(v["detection_rate"]), "detected": v["detected"], "total": v["total"]}
        for name, v in sorted(ta["per_evasion_type"].items(), key=lambda kv: kv[1]["detection_rate"], reverse=True)
    }
    per_benign = {
        name: {"fpr_pct": pct(v["false_positive_rate"]), "false_positives": v["false_positives"], "total": v["total"]}
        for name, v in ta["per_benign"].items()
    }

    external = {}
    for name, fname in (("deepset", "bench_deepset.json"), ("alpaca", "bench_alpaca.json"), ("dolly", "bench_dolly.json")):
        b = _load(f"bm1/{fname}")
        row = {"n_samples": b["n_samples"], "fpr_pct": pct(b["fpr"]), "avg_latency_ms": round(b["avg_latency_ms"], 1)}
        if b.get("n_malicious"):  # labelled set (deepset) carries recall/precision
            row.update({
                "recall_pct": pct(b["recall"]),
                "precision_pct": pct(b["precision"]),
                "f1": round(b["f1"], 3),
                "auc_roc": round(b["auc_roc"], 2),
            })
        external[name] = row

    return {
        "_meta": {
            "generated_by": "scripts/extract_benchmarks.py",
            "edit_source_not_this_file": True,
            "artifact": "benchmarks/results/technique_analysis.json",
            "git_sha": ta.get("git_sha"),
            "measured": (ta.get("timestamp") or "").split("T")[0],
            "threshold": s["threshold"],
        },
        "headline": headline,
        "per_category": per_category,
        "per_evasion_type": per_evasion,
        "per_benign": per_benign,
        "external": external,
    }


def main() -> int:
    facts = build_benchmarks()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        yaml.safe_dump(facts, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
