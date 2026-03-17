"""
Evaluate all taxonomy probes against the live detector.

Runs every probe's samples through classify_prompt() and reports
per-probe recall, per-technique breakdowns, and multi-taxonomy views.

Usage:
    python scripts/evaluate_probes.py
    python scripts/evaluate_probes.py --taxonomy owasp
    python scripts/evaluate_probes.py --taxonomy avid
    python scripts/evaluate_probes.py --taxonomy lmrc
    python scripts/evaluate_probes.py --json
    python scripts/evaluate_probes.py --buffs
    python scripts/evaluate_probes.py --buffs --buff-samples 50
    python scripts/evaluate_probes.py --probes P001 P002
    python scripts/evaluate_probes.py --buffs --output report.json
"""

import argparse
import json
import os
import random
import sys
import time

# Path setup — scripts/taxonomy/ needs to be importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _script_dir)

from taxonomy import ALL_PROBES
from taxonomy._buffs import ALL_BUFFS
from taxonomy._tags import aggregate_by_taxonomy, summarize_groups


def _load_classifier():
    """Load the model and return a classify_fn compatible with Probe.evaluate()."""
    from na0s.predict import predict_prompt, classify_prompt

    vectorizer, model = predict_prompt()

    def classify_fn(text):
        return classify_prompt(text, vectorizer, model)

    return classify_fn


def _print_probe_report(results):
    """Print per-probe recall table."""
    print("\n{:<6s} {:<35s} {:>7s} {:>7s} {:>8s}".format(
        "Probe", "Name", "Det", "Total", "Recall"))
    print("-" * 68)

    weak = []
    for r in results:
        recall_pct = r["recall"] * 100
        marker = "  <-- WEAK" if recall_pct < 80 else ""
        print("{:<6s} {:<35s} {:>7d} {:>7d} {:>7.1f}%{}".format(
            r["probe"], r["name"], r["detected"], r["total"],
            recall_pct, marker))
        if recall_pct < 80:
            weak.append(r)

    total_det = sum(r["detected"] for r in results)
    total_all = sum(r["total"] for r in results)
    overall = total_det / total_all * 100 if total_all else 0
    print("-" * 68)
    print("{:<6s} {:<35s} {:>7d} {:>7d} {:>7.1f}%".format(
        "", "OVERALL", total_det, total_all, overall))

    if weak:
        print("\nWeak probes (recall < 80%):")
        for r in weak:
            print("  {} — {}: {:.1f}%".format(r["probe"], r["name"], r["recall"] * 100))
            for sample_text, tech_id, prob in r["missed_samples"][:3]:
                print("    MISSED {}: {}... (conf={:.2f})".format(
                    tech_id, sample_text[:80], prob))


def _print_taxonomy_report(results, namespace, label):
    """Print results grouped by an external taxonomy."""
    groups = aggregate_by_taxonomy(results, namespace)
    if not groups:
        print("\nNo tags found for namespace '{}'".format(namespace))
        return

    summary = summarize_groups(groups, namespace)

    print("\n{}:".format(label))
    print("{:<35s} {:>7s} {:>7s} {:>7s} {:>8s} {:>8s}  {}".format(
        "Tag", "Det", "Attr", "Total", "Recall", "Attr%", "Probes"))
    print("-" * 96)
    for tag in sorted(groups):
        g = groups[tag]
        print("{:<35s} {:>7d} {:>7d} {:>7d} {:>7.1f}% {:>7.1f}%  {}".format(
            tag.split(":")[-1][:35],
            g["detected"], g["attributed"], g["total"],
            g["recall"] * 100, g["attribution_rate"] * 100,
            ", ".join(g["probes"])))
    print("-" * 96)
    print("{:<35s} {:>7d} {:>7d} {:>7d} {:>7.1f}% {:>7.1f}%".format(
        "TOTAL", summary["detected"], summary["attributed"],
        summary["total"], summary["recall"] * 100,
        summary["attribution_rate"] * 100))


def _run_buff_matrix(classify_fn, probe_classes, seed, max_samples):
    """Run ALL selected probes x ALL buffs and return structured results.

    Returns a dict with:
      - "matrix": list of per-probe rows, each with per-buff recall
      - "per_buff_summary": aggregate detection rate per buff across all probes
      - "per_probe_summary": aggregate detection rate per probe across all buffs
    """
    buff_instances = [B() for B in ALL_BUFFS]
    buff_names = ["raw"] + [b.name for b in buff_instances]

    print("\nBuff mutation matrix ({} probes x {} buffs, {} samples/probe):".format(
        len(probe_classes), len(buff_instances) + 1, max_samples))

    # Header
    header = "{:<25s}".format("Probe")
    for name in buff_names:
        header += " {:>8s}".format(name)
    print(header)
    print("-" * (25 + 9 * len(buff_names)))

    # Accumulators for per-buff aggregate totals
    # Index 0 = raw, 1..N = each buff
    agg_detected = [0] * (len(buff_instances) + 1)
    agg_total = [0] * (len(buff_instances) + 1)

    matrix_rows = []

    for i, ProbeClass in enumerate(probe_classes):
        probe = ProbeClass()
        random.seed(seed + i)
        samples = probe.generate()[:max_samples]
        n_samples = len(samples)

        row_data = {
            "probe": probe.category_id,
            "name": probe.name,
            "n_samples": n_samples,
            "buffs": {},
        }
        row = "{:<25s}".format("{} {}".format(probe.category_id, probe.name[:20]))

        # Raw (no buff)
        det = 0
        for text, tech_id in samples:
            label, prob, hits, l0, *_ = classify_fn(text)
            if l0.rejected or ("SAFE" not in label.upper()):
                det += 1
        recall = det / n_samples * 100 if n_samples else 0
        row += " {:>7.1f}%".format(recall)
        row_data["buffs"]["raw"] = {"detected": det, "total": n_samples,
                                     "recall": det / n_samples if n_samples else 0}
        agg_detected[0] += det
        agg_total[0] += n_samples

        # Each buff
        for j, buff in enumerate(buff_instances):
            det = 0
            tested = 0
            missed = []
            for text, tech_id in samples:
                try:
                    mutated = buff.apply(text)
                except Exception:
                    continue
                tested += 1
                label, prob, hits, l0, *_ = classify_fn(mutated)
                if l0.rejected or ("SAFE" not in label.upper()):
                    det += 1
                else:
                    missed.append({"tech_id": tech_id, "confidence": prob,
                                   "original_snippet": text[:80]})
            recall = det / tested * 100 if tested else 0
            row += " {:>7.1f}%".format(recall)
            row_data["buffs"][buff.name] = {
                "detected": det, "total": tested,
                "recall": det / tested if tested else 0,
                "missed_count": len(missed),
            }
            agg_detected[j + 1] += det
            agg_total[j + 1] += tested

        print(row)
        matrix_rows.append(row_data)

    # Print aggregate per-buff summary row
    print("-" * (25 + 9 * len(buff_names)))
    agg_row = "{:<25s}".format("AGGREGATE")
    per_buff_summary = {}
    for k, name in enumerate(buff_names):
        pct = agg_detected[k] / agg_total[k] * 100 if agg_total[k] else 0
        agg_row += " {:>7.1f}%".format(pct)
        per_buff_summary[name] = {
            "detected": agg_detected[k],
            "total": agg_total[k],
            "recall": agg_detected[k] / agg_total[k] if agg_total[k] else 0,
        }
    print(agg_row)

    # Per-buff detection rate summary
    print("\nPer-buff detection rate summary:")
    print("{:<15s} {:>7s} {:>7s} {:>8s}".format("Buff", "Det", "Total", "Recall"))
    print("-" * 40)
    for name in buff_names:
        s = per_buff_summary[name]
        marker = "  <-- WEAK" if s["recall"] < 0.8 else ""
        print("{:<15s} {:>7d} {:>7d} {:>7.1f}%{}".format(
            name, s["detected"], s["total"], s["recall"] * 100, marker))

    return {
        "matrix": matrix_rows,
        "per_buff_summary": per_buff_summary,
    }


def _select_probes(probe_ids):
    """Filter ALL_PROBES by category_id list. Returns all if probe_ids is None."""
    if not probe_ids:
        return ALL_PROBES
    id_set = set(probe_ids)
    selected = [P for P in ALL_PROBES if P.category_id in id_set]
    missing = id_set - {P.category_id for P in selected}
    if missing:
        print("WARNING: unknown probe IDs: {}".format(", ".join(sorted(missing))))
    return selected


def main():
    parser = argparse.ArgumentParser(description="Evaluate taxonomy probes")
    parser.add_argument("--probes", nargs="*", default=None,
                        help="Probe category_ids to evaluate (default: all)")
    parser.add_argument("--taxonomy", choices=["owasp", "avid", "lmrc", "all"],
                        default=None, help="Show results grouped by external taxonomy")
    parser.add_argument("--json", action="store_true",
                        help="Write full results to data/evaluation/probe_results.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom JSON report output path")
    parser.add_argument("--buffs", action="store_true",
                        help="Run buff mutation sweep across ALL probes x ALL buffs")
    parser.add_argument("--buff-samples", type=int, default=20,
                        help="Max samples per probe in buff matrix (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading model...")
    classify_fn = _load_classifier()

    selected_probes = _select_probes(args.probes)

    print("Evaluating {} probes...\n".format(len(selected_probes)))
    all_results = []
    start = time.time()

    for i, ProbeClass in enumerate(selected_probes):
        probe = ProbeClass()
        random.seed(args.seed + i)
        result = probe.evaluate(classify_fn)
        all_results.append(result)
        print("  {:<6s} {:<35s} recall={:.1f}%  ({}/{})".format(
            result["probe"], result["name"],
            result["recall"] * 100,
            result["detected"], result["total"]))

    elapsed = time.time() - start
    print("\nDone in {:.1f}s".format(elapsed))

    # Internal taxonomy report
    _print_probe_report(all_results)

    # External taxonomy reports
    taxonomy_map = {
        "owasp": ("owasp-llm", "OWASP LLM Top 10 (2025)"),
        "avid": ("avid-effect", "AVID Security/Ethics Effects"),
        "lmrc": ("risk-cards", "LM Risk Cards"),
    }

    if args.taxonomy == "all":
        for ns, label in taxonomy_map.values():
            _print_taxonomy_report(all_results, ns, label)
    elif args.taxonomy:
        ns, label = taxonomy_map[args.taxonomy]
        _print_taxonomy_report(all_results, ns, label)

    # Buff mutation matrix — sweep ALL selected probes x ALL buffs
    buff_results = None
    if args.buffs:
        buff_results = _run_buff_matrix(
            classify_fn, selected_probes, args.seed, args.buff_samples)

    # JSON output
    json_path = args.output
    if args.json and not json_path:
        out_dir = os.path.join(_project_root, "data", "evaluation")
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, "probe_results.json")

    if json_path:
        # Strip missed_samples text for JSON (keep tech_id and prob only)
        for r in all_results:
            r["missed_samples"] = [
                {"tech_id": t, "confidence": p}
                for _, t, p in r["missed_samples"]
            ]

        report = {"probe_results": all_results}
        if buff_results is not None:
            report["buff_matrix"] = buff_results

        out_dir = os.path.dirname(os.path.abspath(json_path))
        os.makedirs(out_dir, exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2)
        print("\nResults written to: {}".format(json_path))


if __name__ == "__main__":
    main()
