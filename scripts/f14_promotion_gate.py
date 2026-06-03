#!/usr/bin/env python3
"""F14 promotion gate — CI regression guard for candidate model releases.

Runs the ~30 hand-curated scenarios from ``data/eval/scenarios/v0.1/*.yaml``
through the full Na0S pipeline against a candidate model, computes per-category
TPR/FPR + per-severity metrics, compares to a committed baseline, and exits 1
if any of the three regression rules trigger.

Usage:
    python scripts/f14_promotion_gate.py --candidate data/processed/
    python scripts/f14_promotion_gate.py --candidate data/processed/ -v
    python scripts/f14_promotion_gate.py --candidate data/processed/ --update-baseline

Regression rules (exit 1 if ANY trigger):
    (a) Overall TPR drops by more than OVERALL_TPR_DROP_LIMIT (2%).
    (b) A category that contains at least one critical-severity scenario has
        TPR delta < 0 AND at least one additional miss vs baseline.
    (c) Global critical-severity TPR drops at all (strictly less than baseline).

Outputs:
    models/f14_gate_results.json                           (always)
    data/eval/scenarios/_baselines/last_good.json          (first-run or --update-baseline)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from na0s.eval.scenarios.loader import load_scenarios_dir  # noqa: E402
from na0s.eval.scenarios.schema import ScenarioType  # noqa: E402

# _load_model_pair owns the (model, vectorizer, scaler) triple convention —
# don't reimplement, import it from shadow_evaluate.
from shadow_evaluate import _load_model_pair  # noqa: E402

# canary_eval owns compute_metrics — same TP/FP/TN/FN arithmetic as the
# canary gate so per-category slices don't silently diverge.
from canary_eval import compute_metrics  # noqa: E402


_DEFAULT_CANDIDATE = str(_PROJECT_ROOT / "data" / "processed")
_DEFAULT_SCENARIOS = str(_PROJECT_ROOT / "data" / "eval" / "scenarios" / "v0.1")
_DEFAULT_BASELINE = str(
    _PROJECT_ROOT / "data" / "eval" / "scenarios" / "_baselines" / "last_good.json"
)
_DEFAULT_RESULTS = str(_PROJECT_ROOT / "models" / "f14_gate_results.json")

OVERALL_TPR_DROP_LIMIT = 0.02
CRITICAL_TPR_DROP_LIMIT = 0.0
CATEGORY_MIN_EXTRA_MISSES = 1


# ── Result dataclasses ──────────────────────────────────────────────────────


@dataclass
class CategoryMetrics:
    tpr: float
    fpr: float
    tp: int
    fn: int
    tn: int
    fp: int
    n_attack: int
    n_benign: int
    critical_count: int


@dataclass
class SeverityMetrics:
    tpr: float
    n: int
    tp: int
    fn: int


@dataclass
class OverallMetrics:
    tpr: float
    fpr: float
    tnr: float
    f1: float
    tp: int
    fn: int
    tn: int
    fp: int
    n_blocked: int
    n_allowed: int


@dataclass
class Regression:
    category: str
    severity: str
    delta: float
    rule: str  # "a_overall_tpr" | "b_category_regress" | "c_critical_tpr"


@dataclass
class BaselineComparison:
    baseline_path: str
    baseline_timestamp: str
    overall_tpr_delta: float
    per_category_tpr_delta: dict[str, float] = field(default_factory=dict)
    regressions: list[Regression] = field(default_factory=list)


@dataclass
class GateReport:
    timestamp: str
    candidate_path: str
    scenarios_total: int
    overall: OverallMetrics
    per_category: dict[str, CategoryMetrics]
    per_severity: dict[str, SeverityMetrics]
    baseline_comparison: BaselineComparison | None
    verdict: str  # "PASS" | "FAIL"
    failures: list[str]


# ── Core flow ───────────────────────────────────────────────────────────────


def main() -> int:
    args = _parse_args()

    if args.update_baseline and args.no_baseline:
        print("ERROR: --update-baseline and --no-baseline are mutually exclusive.",
              file=sys.stderr)
        return 2

    model, vectorizer, scaler = _load_model_pair(args.candidate)

    # classify_prompt reads the scaler from a module-level cache
    # (``predict._cached_scaler``) loaded from production SCALER_PATH. For the
    # gate to evaluate the CANDIDATE scaler (not production), install it into
    # the cache here. v0.2 should add a scaler parameter to classify_prompt
    # and remove this monkey-patch.
    if scaler is not None:
        from na0s import predict as _predict_mod
        _predict_mod._cached_scaler = scaler

    scenarios = list(load_scenarios_dir(Path(args.scenarios_dir)))
    if not scenarios:
        print(f"ERROR: no scenarios found under {args.scenarios_dir!r}",
              file=sys.stderr)
        return 2

    records = _run_scenarios(scenarios, vectorizer, model, verbose=args.verbose)
    overall, per_category, per_severity = _compute_metrics_by_category(records)

    baseline = None if args.no_baseline else _load_baseline(args.baseline)
    comparison, failures = _compare_to_baseline(
        overall=overall,
        per_category=per_category,
        per_severity=per_severity,
        baseline=baseline,
        baseline_path=args.baseline,
    )

    verdict = "PASS" if not failures else "FAIL"
    report = GateReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        candidate_path=args.candidate,
        scenarios_total=len(scenarios),
        overall=overall,
        per_category=per_category,
        per_severity=per_severity,
        baseline_comparison=comparison,
        verdict=verdict,
        failures=failures,
    )

    _emit_json(report, args.results)
    _print_console_report(report)

    # Baseline write rules:
    #   - first-run (baseline missing, not --no-baseline): seed the baseline
    #   - --update-baseline + PASS: overwrite baseline with current run
    #   - --update-baseline + FAIL: refuse (exit 2), don't corrupt baseline
    if args.update_baseline and verdict == "FAIL":
        print("ERROR: refusing to update baseline from a FAIL run.", file=sys.stderr)
        return 2
    if baseline is None and not args.no_baseline:
        _write_baseline(report, args.baseline)
        print(f"NOTE: wrote initial baseline at {args.baseline}")
    elif args.update_baseline:
        _write_baseline(report, args.baseline)
        print(f"NOTE: updated baseline at {args.baseline}")

    return _exit_code_from_verdict(verdict)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Na0S F14 promotion gate against a candidate model",
    )
    parser.add_argument("--candidate", default=_DEFAULT_CANDIDATE,
                        help=f"Candidate model directory (default: {_DEFAULT_CANDIDATE})")
    parser.add_argument("--scenarios-dir", default=_DEFAULT_SCENARIOS,
                        help=f"Scenario YAML dir (default: {_DEFAULT_SCENARIOS})")
    parser.add_argument("--baseline", default=_DEFAULT_BASELINE,
                        help=f"Baseline JSON path (default: {_DEFAULT_BASELINE})")
    parser.add_argument("--results", default=_DEFAULT_RESULTS,
                        help=f"Results JSON path (default: {_DEFAULT_RESULTS})")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Overwrite baseline with current run (PASS only; local use)")
    parser.add_argument("--no-baseline", action="store_true",
                        help="Skip baseline comparison entirely (bootstrap/debug)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-scenario verdict while running")
    return parser.parse_args()


def _load_baseline(path: str) -> dict[str, Any] | None:
    """Return parsed baseline JSON or None if the file doesn't exist.

    A missing baseline is NOT a failure — the first run after F14 ships has
    nothing to compare against. main() seeds the baseline from the current
    report in that case.
    """
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_scenarios(scenarios, vectorizer, model, *, verbose: bool = False):
    """Execute every scenario and collect (y_true, y_pred, category, severity).

    Multi-turn convention: run classify_prompt per turn, mark the scenario as
    "attack detected" if ANY turn predicts MALICIOUS. Break on first hit — this
    matches how an SDK-embedded pipeline fires on the earliest malicious turn
    rather than waiting for the full transcript.
    """
    # Import here so a missing dep in the loader path doesn't prevent _parse_args
    # from completing (useful for --help).
    from na0s.predict import classify_prompt

    records: list[dict[str, Any]] = []
    for scn in scenarios:
        y_true = 1 if scn.expected_verdict == "blocked" else 0
        y_pred = 0
        if scn.type == ScenarioType.SINGLE_PROMPT:
            label, _prob, *_ = classify_prompt(scn.payload, vectorizer, model)
            y_pred = 1 if "MALICIOUS" in label else 0
        elif scn.type == ScenarioType.MULTI_TURN:
            for turn in scn.turns:
                label, _prob, *_ = classify_prompt(turn.text, vectorizer, model)
                if "MALICIOUS" in label:
                    y_pred = 1
                    break
        else:
            raise ValueError(f"Unknown scenario type: {scn.type!r} in {scn.name}")

        records.append({
            "name": scn.name,
            "attack_category": scn.attack_category,
            "severity": scn.severity,
            "y_true": y_true,
            "y_pred": y_pred,
        })
        if verbose:
            mark = "OK  " if y_pred == y_true else "MISS"
            print(f"  [{mark}] {scn.name:<40} cat={scn.attack_category:<8} sev={scn.severity}")
    return records


def _compute_metrics_by_category(records):
    """Slice records three ways and compute metrics per slice."""
    y_true_all = [r["y_true"] for r in records]
    y_pred_all = [r["y_pred"] for r in records]
    m = compute_metrics(y_true_all, y_pred_all)
    overall = OverallMetrics(
        tpr=m["tpr"], fpr=m["fpr"], tnr=m["tnr"], f1=m["f1"],
        tp=m["tp"], fn=m["fn"], tn=m["tn"], fp=m["fp"],
        n_blocked=m["tp"] + m["fn"], n_allowed=m["tn"] + m["fp"],
    )

    cat_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        cat_buckets[r["attack_category"]].append(r)
    per_category: dict[str, CategoryMetrics] = {}
    for cat, rows in cat_buckets.items():
        cm = compute_metrics([r["y_true"] for r in rows], [r["y_pred"] for r in rows])
        per_category[cat] = CategoryMetrics(
            tpr=cm["tpr"], fpr=cm["fpr"],
            tp=cm["tp"], fn=cm["fn"], tn=cm["tn"], fp=cm["fp"],
            n_attack=cm["tp"] + cm["fn"], n_benign=cm["tn"] + cm["fp"],
            critical_count=sum(1 for r in rows if r["severity"] == "critical"),
        )

    sev_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r["y_true"] == 1:
            sev_buckets[r["severity"]].append(r)
    per_severity: dict[str, SeverityMetrics] = {}
    for sev, rows in sev_buckets.items():
        sm = compute_metrics([r["y_true"] for r in rows], [r["y_pred"] for r in rows])
        per_severity[sev] = SeverityMetrics(
            tpr=sm["tpr"], n=len(rows), tp=sm["tp"], fn=sm["fn"],
        )

    return overall, per_category, per_severity


def _compare_to_baseline(*, overall, per_category, per_severity,
                          baseline, baseline_path):
    """Apply the three regression rules; return (comparison, failure_messages)."""
    if baseline is None:
        return None, []

    b_overall = baseline["overall"]
    overall_delta = overall.tpr - b_overall["tpr"]
    per_cat_delta: dict[str, float] = {}
    regressions: list[Regression] = []
    failures: list[str] = []

    # Rule (a): overall TPR drop > limit
    if overall_delta < -OVERALL_TPR_DROP_LIMIT:
        regressions.append(Regression(
            category="__overall__", severity="__any__",
            delta=overall_delta, rule="a_overall_tpr",
        ))
        failures.append(
            f"Rule (a): overall TPR dropped {overall_delta:+.3f} "
            f"(limit {-OVERALL_TPR_DROP_LIMIT:+.3f})"
        )

    # Rule (b): critical-containing category regressed with extra misses
    b_categories = baseline.get("per_category", {})
    for cat, cm in per_category.items():
        b_cm = b_categories.get(cat)
        if b_cm is None:
            per_cat_delta[cat] = 0.0
            continue
        delta = cm.tpr - b_cm["tpr"]
        per_cat_delta[cat] = delta
        # Use raw FN counts when present (new baseline schema); fall back to
        # rounded TPR-derived count for legacy baselines.
        baseline_fn = b_cm.get("fn")
        if baseline_fn is None:
            baseline_fn = b_cm["n_attack"] - int(round(b_cm["tpr"] * b_cm["n_attack"]))
        extra_misses = cm.fn - baseline_fn
        if (
            cm.critical_count > 0
            and delta < 0
            and extra_misses >= CATEGORY_MIN_EXTRA_MISSES
        ):
            regressions.append(Regression(
                category=cat, severity="critical-containing",
                delta=delta, rule="b_category_regress",
            ))
            failures.append(
                f"Rule (b): category {cat} TPR dropped {delta:+.3f} with "
                f"{extra_misses} additional miss(es)"
            )

    # Rule (c): global critical-severity TPR any-drop
    b_critical = baseline.get("per_severity", {}).get("critical")
    cur_critical = per_severity.get("critical")
    if b_critical is not None and cur_critical is not None:
        c_delta = cur_critical.tpr - b_critical["tpr"]
        if c_delta < -CRITICAL_TPR_DROP_LIMIT:
            regressions.append(Regression(
                category="__any__", severity="critical",
                delta=c_delta, rule="c_critical_tpr",
            ))
            failures.append(
                f"Rule (c): critical-severity TPR dropped {c_delta:+.3f}"
            )

    comparison = BaselineComparison(
        baseline_path=baseline_path,
        baseline_timestamp=baseline.get("timestamp", "unknown"),
        overall_tpr_delta=overall_delta,
        per_category_tpr_delta=per_cat_delta,
        regressions=regressions,
    )
    return comparison, failures


def _emit_json(report: GateReport, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, sort_keys=False)


def _write_baseline(report: GateReport, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = asdict(report)
    payload["baseline_comparison"] = None  # baselines never reference themselves
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def _print_console_report(report: GateReport) -> None:
    print("=" * 70)
    print("  Na0S F14 Promotion Gate")
    print("=" * 70)
    print(f"  Candidate:     {report.candidate_path}")
    print(f"  Scenarios:     {report.scenarios_total}")
    print(f"  Overall TPR:   {report.overall.tpr:.3f}  "
          f"FPR: {report.overall.fpr:.3f}  F1: {report.overall.f1:.3f}")
    print()
    print("  Per-category TPR:")
    for cat, cm in sorted(report.per_category.items()):
        print(f"    {cat:<8} tpr={cm.tpr:.3f} fpr={cm.fpr:.3f} "
              f"n_attack={cm.n_attack} n_benign={cm.n_benign} "
              f"crit={cm.critical_count}")
    print()
    if report.baseline_comparison:
        bc = report.baseline_comparison
        print(f"  Baseline:      {bc.baseline_path}")
        print(f"  Overall dTPR:  {bc.overall_tpr_delta:+.3f}")
        if bc.regressions:
            print("  Regressions:")
            for reg in bc.regressions:
                print(f"    [{reg.rule}] {reg.category} "
                      f"sev={reg.severity} delta={reg.delta:+.3f}")
    else:
        print("  Baseline:      (none — bootstrapping on this run)")
    print()
    print(f"  VERDICT: {report.verdict}")
    if report.failures:
        for msg in report.failures:
            print(f"    - {msg}")
    print("=" * 70)


def _exit_code_from_verdict(verdict: str) -> int:
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
