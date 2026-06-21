#!/usr/bin/env python
"""Evaluate the LLM judge's false-positive rate on the combined dataset.

Runs the LLM judge (OpenAI or Groq backend) against a sample of the
training data and reports TP/FP/TN/FN, FPR, FNR, precision, recall,
and latency statistics.

Usage:
    PYTHONPATH=src:. python scripts/evaluate_llm_judge.py [--backend openai|groq] [--max N]
"""

import argparse
import csv
import os
import sys
import time

from na0s.judge.llm_judge import LLMJudge

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATASET_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "combined_data.csv"
)


def evaluate(backend="openai", model=None, max_samples=500):
    judge = LLMJudge(backend=backend, model=model)
    print("=== LLM Judge Evaluation ===")
    print("Backend: {}  Model: {}".format(backend, judge.model))
    print("Dataset: {}".format(DATASET_PATH))
    print("Max samples: {}\n".format(max_samples))

    counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "errors": 0}
    latencies = []
    fp_examples = []
    fn_examples = []
    total = 0

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if total >= max_samples:
                break

            text = row["text"]
            true_label = int(row["label"])  # 0=safe, 1=malicious

            verdict = judge.classify(text)
            latencies.append(verdict.latency_ms)
            total += 1

            if verdict.error:
                counts["errors"] += 1
                continue

            predicted_malicious = verdict.verdict == "MALICIOUS"

            if true_label == 1 and predicted_malicious:
                counts["TP"] += 1
            elif true_label == 0 and predicted_malicious:
                counts["FP"] += 1
                if len(fp_examples) < 10:
                    fp_examples.append(
                        (text[:120], verdict.confidence, verdict.reasoning)
                    )
            elif true_label == 0 and not predicted_malicious:
                counts["TN"] += 1
            elif true_label == 1 and not predicted_malicious:
                counts["FN"] += 1
                if len(fn_examples) < 10:
                    fn_examples.append(
                        (text[:120], verdict.confidence, verdict.reasoning)
                    )

            # Progress indicator every 50 samples
            if total % 50 == 0:
                elapsed = sum(latencies)
                print(
                    "  ... {}/{} samples ({:.0f}ms total)".format(
                        total, max_samples, elapsed
                    )
                )

    # --- Metrics ---
    total_safe = counts["TN"] + counts["FP"]
    total_malicious = counts["TP"] + counts["FN"]

    fpr = counts["FP"] / total_safe if total_safe > 0 else 0
    fnr = counts["FN"] / total_malicious if total_malicious > 0 else 0
    precision = (
        counts["TP"] / (counts["TP"] + counts["FP"])
        if (counts["TP"] + counts["FP"]) > 0
        else 0
    )
    recall = (
        counts["TP"] / (counts["TP"] + counts["FN"])
        if (counts["TP"] + counts["FN"]) > 0
        else 0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    sorted_lat = sorted(latencies) if latencies else [0]
    p50 = sorted_lat[len(sorted_lat) // 2]
    p95_idx = min(int(0.95 * len(sorted_lat)), len(sorted_lat) - 1)
    p95 = sorted_lat[p95_idx]

    print("\n--- Results ---")
    print("Samples evaluated: {}".format(total))
    print("API errors:        {}".format(counts["errors"]))
    print()
    print("TP: {:>5}   FP: {:>5}".format(counts["TP"], counts["FP"]))
    print("FN: {:>5}   TN: {:>5}".format(counts["FN"], counts["TN"]))
    print()
    print("FPR:       {:.2%}".format(fpr))
    print("FNR:       {:.2%}".format(fnr))
    print("Precision: {:.2%}".format(precision))
    print("Recall:    {:.2%}".format(recall))
    print("F1:        {:.2%}".format(f1))
    print()
    print("Latency  avg: {:.0f}ms  p50: {:.0f}ms  p95: {:.0f}ms".format(
        avg_latency, p50, p95
    ))

    if fp_examples:
        print("\n--- Top False Positives (safe inputs flagged MALICIOUS) ---")
        for text, conf, reason in fp_examples:
            print("  [{:.2f}] {} ...".format(conf, text))
            print("         reason: {}".format(reason))

    if fn_examples:
        print("\n--- Top False Negatives (malicious inputs missed) ---")
        for text, conf, reason in fn_examples:
            print("  [{:.2f}] {} ...".format(conf, text))
            print("         reason: {}".format(reason))

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM judge on training dataset"
    )
    parser.add_argument(
        "--backend",
        choices=["openai", "groq"],
        default="groq",
        help="LLM backend (default: groq)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model name (default: backend default)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=500,
        dest="max_samples",
        help="Maximum samples to evaluate (default: 500)",
    )
    args = parser.parse_args()
    evaluate(
        backend=args.backend,
        model=args.model,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
