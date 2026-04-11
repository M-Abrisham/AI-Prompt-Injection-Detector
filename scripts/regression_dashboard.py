"""
Regression dashboard — track probe evaluation metrics over time.

Reads probe_results.json (written by evaluate_probes.py), appends a
snapshot to regression_history.jsonl, and compares successive runs to
flag regressions.

Usage:
    python scripts/regression_dashboard.py --run
    python scripts/regression_dashboard.py --compare
    python scripts/regression_dashboard.py --baseline
    python scripts/regression_dashboard.py --run --output json
"""

import argparse
import json
import os
import subprocess
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
_EVAL_DIR = os.path.join(_PROJECT_ROOT, "data", "evaluation")
_RESULTS_PATH = os.path.join(_EVAL_DIR, "probe_results.json")
_HISTORY_PATH = os.path.join(_EVAL_DIR, "regression_history.jsonl")

REGRESSION_THRESHOLD = 0.02  # 2 percentage-point drop


def _git_sha():
    """Return short git SHA for the current HEAD, or 'unknown'."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def load_probe_results(path=None):
    """Load probe_results.json and return the list under 'probe_results'."""
    path = path or _RESULTS_PATH
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("probe_results", [])


def build_snapshot(probe_results, is_baseline=False):
    """Build a regression-history entry from a list of probe result dicts."""
    per_probe = {}
    for r in probe_results:
        probe_id = r.get("probe", "unknown")
        per_probe[probe_id] = {
            "recall": r.get("recall", 0.0),
            "attribution_rate": r.get("attribution_rate", 0.0),
            "sample_count": r.get("total", 0),
        }

    recalls = [v["recall"] for v in per_probe.values()]
    attr_rates = [v["attribution_rate"] for v in per_probe.values()]
    n = len(recalls)

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "is_baseline": is_baseline,
        "per_probe": per_probe,
        "overall": {
            "mean_recall": sum(recalls) / n if n else 0.0,
            "mean_attribution_rate": sum(attr_rates) / n if n else 0.0,
        },
        "latency_ms": {"p50": None, "p95": None},
    }


def append_history(snapshot, path=None):
    """Append a snapshot as one JSON line to the history file."""
    path = path or _HISTORY_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(snapshot) + "\n")


def load_history(path=None):
    """Load all entries from regression_history.jsonl."""
    path = path or _HISTORY_PATH
    if not os.path.isfile(path):
        return []
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def compute_deltas(current, previous):
    """Compare two snapshots and return a list of delta dicts.

    Each dict: {probe, recall, prev_recall, delta, status}.
    Status is 'REGRESSION' if recall dropped > REGRESSION_THRESHOLD,
    'NEW' if the probe wasn't in the previous snapshot, else 'OK'.
    """
    deltas = []
    prev_probes = previous.get("per_probe", {})
    curr_probes = current.get("per_probe", {})

    all_probes = sorted(set(curr_probes) | set(prev_probes))
    for probe_id in all_probes:
        curr = curr_probes.get(probe_id)
        prev = prev_probes.get(probe_id)

        if curr is None:
            # Probe was removed — skip or mark
            continue

        recall = curr["recall"]
        if prev is None:
            deltas.append({
                "probe": probe_id,
                "recall": recall,
                "prev_recall": None,
                "delta": None,
                "status": "NEW",
            })
        else:
            prev_recall = prev["recall"]
            delta = recall - prev_recall
            if delta < -REGRESSION_THRESHOLD:
                status = "REGRESSION"
            else:
                status = "OK"
            deltas.append({
                "probe": probe_id,
                "recall": recall,
                "prev_recall": prev_recall,
                "delta": delta,
                "status": status,
            })
    return deltas


def format_table(deltas):
    """Render deltas as a human-readable table string."""
    lines = []
    header = "{:<8s} {:>8s} {:>8s} {:>8s} {:>12s}".format(
        "Probe", "Recall", "Prev", "Delta", "Status")
    lines.append(header)
    lines.append("-" * len(header))
    for d in deltas:
        recall_s = "{:.1f}%".format(d["recall"] * 100)
        prev_s = "{:.1f}%".format(d["prev_recall"] * 100) if d["prev_recall"] is not None else "—"
        delta_s = "{:+.1f}%".format(d["delta"] * 100) if d["delta"] is not None else "—"
        lines.append("{:<8s} {:>8s} {:>8s} {:>8s} {:>12s}".format(
            d["probe"], recall_s, prev_s, delta_s, d["status"]))
    return "\n".join(lines)


def format_snapshot_table(snapshot):
    """Render a single snapshot as a summary table string."""
    lines = []
    header = "{:<8s} {:>8s} {:>8s} {:>8s}".format(
        "Probe", "Recall", "Attr%", "Samples")
    lines.append(header)
    lines.append("-" * len(header))
    for probe_id in sorted(snapshot["per_probe"]):
        p = snapshot["per_probe"][probe_id]
        lines.append("{:<8s} {:>7.1f}% {:>7.1f}% {:>8d}".format(
            probe_id,
            p["recall"] * 100,
            p["attribution_rate"] * 100,
            p["sample_count"]))
    ov = snapshot["overall"]
    lines.append("-" * len(header))
    lines.append("{:<8s} {:>7.1f}% {:>7.1f}%".format(
        "MEAN", ov["mean_recall"] * 100, ov["mean_attribution_rate"] * 100))
    return "\n".join(lines)


def cmd_run(output_format="table", is_baseline=False, results_path=None,
            history_path=None):
    """Execute --run (or --baseline): read results, append history, print."""
    probe_results = load_probe_results(results_path)
    if probe_results is None:
        print("ERROR: probe_results.json not found. Run evaluate_probes.py first.",
              file=sys.stderr)
        return 1

    snapshot = build_snapshot(probe_results, is_baseline=is_baseline)
    append_history(snapshot, history_path)

    if output_format == "json":
        print(json.dumps(snapshot, indent=2))
    else:
        print("Regression dashboard — {} (git {})".format(
            snapshot["timestamp"], snapshot["git_sha"]))
        if is_baseline:
            print("** Marked as BASELINE **")
        print()
        print(format_snapshot_table(snapshot))

    return 0


def cmd_compare(output_format="table", history_path=None):
    """Execute --compare: load last 2 history entries, print delta."""
    entries = load_history(history_path)
    if len(entries) < 2:
        msg = "Not enough history entries for comparison (need >= 2, have {}).".format(
            len(entries))
        if output_format == "json":
            print(json.dumps({"error": msg}))
        else:
            print(msg)
        return 0

    current = entries[-1]
    previous = entries[-2]
    deltas = compute_deltas(current, previous)

    if output_format == "json":
        print(json.dumps({"current_sha": current["git_sha"],
                          "previous_sha": previous["git_sha"],
                          "deltas": deltas}, indent=2))
    else:
        print("Comparing {} ({}) vs {} ({})".format(
            current["git_sha"], current["timestamp"],
            previous["git_sha"], previous["timestamp"]))
        print()
        print(format_table(deltas))

    regressions = [d for d in deltas if d["status"] == "REGRESSION"]
    if regressions:
        return 1
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Regression dashboard for probe evaluation metrics")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true",
                       help="Read probe_results.json, append to history, print summary")
    group.add_argument("--compare", action="store_true",
                       help="Compare last 2 history entries and flag regressions")
    group.add_argument("--baseline", action="store_true",
                       help="Same as --run but mark entry as pinned baseline")
    parser.add_argument("--output", choices=["table", "json"], default="table",
                        help="Output format (default: table)")
    args = parser.parse_args(argv)

    if args.run:
        return cmd_run(output_format=args.output)
    elif args.baseline:
        return cmd_run(output_format=args.output, is_baseline=True)
    elif args.compare:
        return cmd_compare(output_format=args.output)


if __name__ == "__main__":
    sys.exit(main() or 0)
