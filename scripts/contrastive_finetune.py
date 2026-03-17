"""Contrastive learning fine-tuning for sentence-transformer embeddings.

Fine-tunes a sentence-transformer on injection/safe pairs using contrastive
loss so that injection prompts cluster tightly and separate from benign text.

Usage:
    PYTHONPATH=src:. python scripts/contrastive_finetune.py \
        --model all-MiniLM-L6-v2 \
        --data data/processed/combined_data.csv \
        --output data/processed/contrastive_model \
        --epochs 3 --batch-size 32
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency guard: sentence-transformers is optional
# ---------------------------------------------------------------------------
_HAS_SENTENCE_TRANSFORMERS = False
_import_error: Optional[str] = None

try:
    from sentence_transformers import SentenceTransformer, InputExample
    from sentence_transformers.losses import CosineSimilarityLoss
    from torch.utils.data import DataLoader

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError as exc:
    _import_error = str(exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_training_pairs(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    max_pairs: int = 50000,
    seed: int = 42,
) -> list:
    """Create positive/negative pairs from a labelled DataFrame.

    Positive pair (label 1.0): two texts with the SAME class label.
    Negative pair (label 0.0): two texts with DIFFERENT class labels.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *text_col* and *label_col* columns.
        label_col should be 0 (safe) or 1 (malicious).
    text_col : str
        Column name for text.
    label_col : str
        Column name for integer label (0 or 1).
    max_pairs : int
        Maximum number of pairs to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[InputExample]
        Training pairs with cosine similarity labels (1.0 or 0.0).
        If sentence-transformers is not installed, returns a list of
        plain dicts with keys ``texts`` and ``label``.
    """
    rng = random.Random(seed)

    safe_texts = df.loc[df[label_col] == 0, text_col].tolist()
    mal_texts = df.loc[df[label_col] == 1, text_col].tolist()

    if not safe_texts or not mal_texts:
        raise ValueError(
            "Need both safe (label=0) and malicious (label=1) samples. "
            "Got safe={0}, malicious={1}".format(len(safe_texts), len(mal_texts))
        )

    pairs: list = []
    half = max_pairs // 2

    # Positive pairs (same class)
    for _ in range(half):
        if rng.random() < 0.5:
            a, b = rng.sample(safe_texts, min(2, len(safe_texts)))[:2] if len(safe_texts) >= 2 else (safe_texts[0], safe_texts[0])
        else:
            a, b = rng.sample(mal_texts, min(2, len(mal_texts)))[:2] if len(mal_texts) >= 2 else (mal_texts[0], mal_texts[0])
        pair = _make_pair([a, b], 1.0)
        pairs.append(pair)

    # Negative pairs (different class)
    for _ in range(half):
        a = rng.choice(safe_texts)
        b = rng.choice(mal_texts)
        pair = _make_pair([a, b], 0.0)
        pairs.append(pair)

    rng.shuffle(pairs)
    return pairs


def _make_pair(texts: List[str], label: float):
    """Create a training pair — InputExample if available, else dict."""
    if _HAS_SENTENCE_TRANSFORMERS:
        return InputExample(texts=texts, label=label)
    return {"texts": texts, "label": label}


def finetune(
    model_name: str = "all-MiniLM-L6-v2",
    pairs: Optional[list] = None,
    output_path: str = "data/processed/contrastive_model",
    epochs: int = 3,
    batch_size: int = 32,
    warmup_ratio: float = 0.1,
) -> Optional[object]:
    """Fine-tune a sentence-transformer using cosine similarity loss.

    Parameters
    ----------
    model_name : str
        HuggingFace model name or local path.
    pairs : list[InputExample]
        Training pairs from :func:`create_training_pairs`.
    output_path : str
        Directory to save the fine-tuned model.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    warmup_ratio : float
        Fraction of total steps used for linear warmup.

    Returns
    -------
    SentenceTransformer or None
        The fine-tuned model, or None if dependencies are missing.
    """
    if not _HAS_SENTENCE_TRANSFORMERS:
        logger.warning(
            "sentence-transformers not installed — cannot fine-tune. "
            "Install with: pip install sentence-transformers"
        )
        return None

    if pairs is None or len(pairs) == 0:
        raise ValueError("No training pairs provided")

    model = SentenceTransformer(model_name)
    train_dataloader = DataLoader(pairs, shuffle=True, batch_size=batch_size)
    loss = CosineSimilarityLoss(model=model)

    warmup_steps = int(len(train_dataloader) * epochs * warmup_ratio)

    logger.info(
        "Fine-tuning '%s': %d pairs, %d epochs, batch_size=%d, warmup=%d steps",
        model_name, len(pairs), epochs, batch_size, warmup_steps,
    )

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True,
    )

    logger.info("Model saved to %s", output_path)
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune sentence-transformer with contrastive learning",
    )
    parser.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="Base sentence-transformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--data", default="data/processed/combined_data.csv",
        help="Path to CSV with 'text' and 'label' columns",
    )
    parser.add_argument(
        "--output", default="data/processed/contrastive_model",
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--max-pairs", type=int, default=50000,
        help="Maximum number of training pairs to generate (default: 50000)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    args = _parse_args(argv)

    if not _HAS_SENTENCE_TRANSFORMERS:
        print(
            "ERROR: sentence-transformers is required for contrastive fine-tuning.\n"
            "Install with: pip install sentence-transformers\n"
            "Import error: {0}".format(_import_error),
            file=sys.stderr,
        )
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("Loading data from {0}".format(args.data))
    df = pd.read_csv(args.data)
    df["text"] = df["text"].fillna("").astype(str)

    print("Creating training pairs (max {0})...".format(args.max_pairs))
    pairs = create_training_pairs(df, max_pairs=args.max_pairs)
    print("  Generated {0} pairs".format(len(pairs)))

    print("Fine-tuning {0} for {1} epochs...".format(args.model, args.epochs))
    finetune(
        model_name=args.model,
        pairs=pairs,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print("Done. Model saved to {0}".format(args.output))


if __name__ == "__main__":
    main()
