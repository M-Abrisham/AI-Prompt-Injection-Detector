"""Fine-tune Llama 3.2 for prompt injection detection using QLoRA.

This is a Layer 4 scaffold script.  It is **not** part of the default
Na0S pipeline and requires additional dependencies:

    pip install torch transformers peft trl bitsandbytes datasets accelerate

Usage:
    python scripts/finetune_llama.py                        # defaults
    python scripts/finetune_llama.py --model meta-llama/Llama-3.2-3B \
        --epochs 5 --batch-size 2 --max-samples 10000
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency guard
# ---------------------------------------------------------------------------

REQUIRED_PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "peft": "peft",
    "trl": "trl",
    "bitsandbytes": "bitsandbytes",
    "datasets": "datasets",
    "accelerate": "accelerate",
}


def check_dependencies() -> list[str]:
    """Return list of missing package *import* names."""
    missing: list[str] = []
    for import_name in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(import_name)
    return missing


def dependency_guard() -> None:
    """Exit with a helpful message when dependencies are missing."""
    missing = check_dependencies()
    if missing:
        pip_names = " ".join(REQUIRED_PACKAGES[m] for m in missing)
        print(
            "ERROR: The following packages are required but not installed:\n"
            f"  {', '.join(missing)}\n\n"
            "Install them with:\n"
            f"  pip install {pip_names}\n",
            file=sys.stderr,
        )
        sys.exit(1)


def check_hf_token() -> None:
    """Warn (but do not exit) if HF_TOKEN is unset — needed for gated models."""
    if not os.environ.get("HF_TOKEN"):
        print(
            "WARNING: HF_TOKEN environment variable is not set.\n"
            "Llama models are gated — you may need to run:\n"
            "  export HF_TOKEN=<your-huggingface-token>\n"
            "and accept the licence at https://huggingface.co/meta-llama\n",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Instruction formatting
# ---------------------------------------------------------------------------

INSTRUCTION = (
    "Classify the following text as either SAFE or MALICIOUS "
    "(prompt injection)."
)

LABEL_MAP = {0: "SAFE", 1: "MALICIOUS"}


def format_instruction(text: str, label: int) -> str:
    """Return an instruction-tuning prompt for a single example."""
    return (
        f"### Instruction: {INSTRUCTION}\n"
        f"### Input: {text}\n"
        f"### Response: {LABEL_MAP[label]}"
    )


def format_instruction_inference(text: str) -> str:
    """Return the prompt used at inference time (no response)."""
    return (
        f"### Instruction: {INSTRUCTION}\n"
        f"### Input: {text}\n"
        f"### Response:"
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = "data/processed/combined_data.csv"


def load_training_data(
    path: str = DEFAULT_DATA_PATH,
    max_samples: int | None = None,
) -> list[dict]:
    """Load CSV and return list of dicts with ``text`` key for SFTTrainer."""
    records: list[dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "text" not in (reader.fieldnames or []) or "label" not in (reader.fieldnames or []):
            raise ValueError(
                f"CSV at {path} must have 'text' and 'label' columns; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            label = int(row["label"])
            records.append({"text": format_instruction(row["text"], label)})
            if max_samples is not None and len(records) >= max_samples:
                break
    if not records:
        raise ValueError(f"No records loaded from {path}")
    print(f"Loaded {len(records)} training samples from {path}")
    return records


# ---------------------------------------------------------------------------
# LoRA / QLoRA configuration helpers
# ---------------------------------------------------------------------------

def get_lora_config():
    """Return a LoraConfig for the fine-tune."""
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def get_quantization_config():
    """Return a BitsAndBytesConfig for 4-bit QLoRA."""
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Run the QLoRA fine-tune."""
    # Late imports so the dependency guard runs first.
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    check_hf_token()

    # ---- data ----
    records = load_training_data(args.data, args.max_samples)
    dataset = Dataset.from_list(records)

    # ---- model + tokenizer ----
    print(f"Loading model: {args.model}")
    quant_config = get_quantization_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=False,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- LoRA ----
    model = prepare_model_for_kbit_training(model)
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- training args ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        optim="paged_adamw_8bit",
        warmup_ratio=0.05,
        gradient_accumulation_steps=max(1, 16 // args.batch_size),
    )

    # ---- trainer ----
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
        max_seq_length=512,
    )

    print("Starting training...")
    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    # ---- metrics ----
    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"  Total loss : {result.training_loss:.4f}")
    for entry in trainer.state.log_history:
        if "loss" in entry:
            step = entry.get("step", "?")
            loss = entry["loss"]
            print(f"  step {step}: loss={loss:.4f}")

    # ---- save ----
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nAdapter + tokenizer saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 3.2 for prompt injection detection (QLoRA).",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model ID (default: meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help=f"Path to training CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device batch size (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/models/llama-injection",
        help="Directory for saved adapter (default: data/models/llama-injection)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit training samples (default: use all)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    dependency_guard()
    parser = build_parser()
    args = parser.parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
