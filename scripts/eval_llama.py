"""Evaluate a fine-tuned Llama LoRA adapter on prompt injection detection.

Usage:
    python scripts/eval_llama.py --model-dir data/models/llama-injection
    python scripts/eval_llama.py --model-dir data/models/llama-injection --max-samples 500

Requires the same dependencies as ``finetune_llama.py``:

    pip install torch transformers peft bitsandbytes datasets accelerate
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Re-use utilities from the fine-tuning script.
from finetune_llama import (
    LABEL_MAP,
    check_dependencies,
    dependency_guard,
    format_instruction_inference,
)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = "data/processed/combined_data.csv"

RESPONSE_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}  # SAFE->0, MALICIOUS->1


def load_eval_data(
    path: str,
    max_samples: int | None = None,
) -> tuple[list[str], list[int]]:
    """Load CSV and return (texts, labels)."""
    texts: list[str] = []
    labels: list[int] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
            if max_samples is not None and len(texts) >= max_samples:
                break
    print(f"Loaded {len(texts)} evaluation samples from {path}")
    return texts, labels


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_batch(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 8,
) -> list[int]:
    """Generate predictions for a list of texts."""
    import torch

    predictions: list[int] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        prompts = [format_instruction_inference(t) for t in batch_texts]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=1.0,
            )

        for j, output_ids in enumerate(outputs):
            # Decode only the generated tokens (after the input).
            input_len = inputs["input_ids"][j].shape[0]
            generated = tokenizer.decode(
                output_ids[input_len:], skip_special_tokens=True
            ).strip().upper()

            # Map to label — default to MALICIOUS (fail-safe).
            if "SAFE" in generated and "MALICIOUS" not in generated:
                predictions.append(0)
            else:
                predictions.append(1)

        done = min(i + batch_size, len(texts))
        print(f"  predicted {done}/{len(texts)}", end="\r")

    print()
    return predictions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def print_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """Print and return classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n===== Llama Fine-Tune Evaluation =====")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"\nConfusion matrix:\n{cm}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['SAFE', 'MALICIOUS'])}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def compare_baseline(metrics: dict) -> None:
    """If TF-IDF training metrics exist, print a comparison."""
    baseline_path = Path("data/processed/training_metrics.json")
    if not baseline_path.exists():
        print("(TF-IDF baseline metrics not found — skipping comparison)")
        return

    import json

    with open(baseline_path) as fh:
        baseline = json.load(fh)

    print("\n===== Comparison vs TF-IDF Baseline =====")
    for key in ("accuracy", "precision", "recall", "f1"):
        base_val = baseline.get(key) or baseline.get(f"test_{key}")
        if base_val is not None:
            delta = metrics[key] - float(base_val)
            sign = "+" if delta >= 0 else ""
            print(f"  {key:12s}  Llama={metrics[key]:.4f}  TF-IDF={float(base_val):.4f}  ({sign}{delta:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> dict:
    """Load model and run evaluation."""
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    texts, labels = load_eval_data(args.data, args.max_samples)

    print(f"Loading fine-tuned model from {args.model_dir}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Running inference...")
    t0 = time.time()
    predictions = predict_batch(model, tokenizer, texts, batch_size=args.batch_size)
    elapsed = time.time() - t0
    print(f"Inference completed in {elapsed:.1f}s ({len(texts)/elapsed:.1f} samples/s)")

    metrics = print_metrics(labels, predictions)
    compare_baseline(metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Llama prompt-injection detector.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the saved LoRA adapter directory",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help=f"Path to evaluation CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Inference batch size (default: 8)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit evaluation samples (default: use all)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    dependency_guard()
    parser = build_parser()
    args = parser.parse_args(argv)
    evaluate(args)


if __name__ == "__main__":
    main()
