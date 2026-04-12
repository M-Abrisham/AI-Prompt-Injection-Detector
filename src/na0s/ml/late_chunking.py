"""Layer 5 enhancement: Late chunking for embeddings.

Late chunking embeds a full document first at the token level, then splits
the token-level embeddings into overlapping chunks.  Each chunk retains
full-document context, solving the "buried payload" problem where injections
are hidden deep inside long, otherwise-benign documents.

Traditional chunking:
    text -> split -> embed each chunk independently
    Problem: each chunk loses context from the rest of the document.

Late chunking:
    text -> embed full document (token-level) -> split embeddings -> pool
    Result: each chunk's embedding is informed by the entire document.

Usage::

    from na0s.late_chunking import late_chunk_classify

    result = late_chunk_classify(text, model, tokenizer, classifier)
    # result = {"max_score": 0.87, "max_chunk_idx": 3, "chunks": [...]}

Environment variable ``NA0S_LATE_CHUNKING=1`` enables late chunking in the
inference pipeline (default: disabled).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful import: sentence-transformers + torch are optional
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_LATE_CHUNKING_ENABLED = os.environ.get("NA0S_LATE_CHUNKING", "0") == "1"
_MIN_TOKENS_FOR_CHUNKING = int(
    os.environ.get("NA0S_LATE_CHUNK_MIN_TOKENS", "256")
)

# Default model — matches the rest of the Na0S embedding pipeline
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
_HUGGINGFACE_PREFIX = "sentence-transformers/"


# ---------------------------------------------------------------------------
# Core: late chunking embed
# ---------------------------------------------------------------------------

def late_chunk_embed(
    text: str,
    model: Any,
    tokenizer: Any,
    chunk_size: int = 128,
    stride: int = 64,
) -> Optional[List["np.ndarray"]]:
    """Embed a document using late chunking.

    Tokenizes the full document, obtains token-level embeddings from the
    transformer model, then splits the token embeddings into overlapping
    chunks and mean-pools each chunk.

    Parameters
    ----------
    text : str
        Full document text.
    model : transformers.PreTrainedModel
        A HuggingFace transformer model that outputs ``last_hidden_state``.
    tokenizer : transformers.PreTrainedTokenizer
        Matching tokenizer for the model.
    chunk_size : int
        Number of tokens per chunk (default 128).
    stride : int
        Step size between chunk starts (default 64, giving 50% overlap).

    Returns
    -------
    list[np.ndarray] or None
        List of chunk embeddings, each of shape ``(embedding_dim,)``.
        Returns *None* if transformers are not available or if the text
        is too short for chunking.
    """
    if not _HAS_TRANSFORMERS:
        return None

    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1, got {0}".format(chunk_size))
    if stride < 1:
        raise ValueError("stride must be >= 1, got {0}".format(stride))

    # Tokenize the full document
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,  # model max context
        padding=False,
    )

    input_ids = encoded["input_ids"]
    num_tokens = input_ids.shape[1]

    # Skip special tokens ([CLS] and [SEP]) for chunking boundaries
    # Token positions: 0=[CLS], 1..num_tokens-2=content, num_tokens-1=[SEP]
    content_start = 1
    content_end = num_tokens - 1
    content_length = content_end - content_start

    if content_length < _MIN_TOKENS_FOR_CHUNKING:
        return None

    # Forward pass: get token-level embeddings
    with torch.no_grad():
        outputs = model(**encoded)

    # last_hidden_state shape: (1, seq_len, hidden_dim)
    token_embeddings = outputs.last_hidden_state.squeeze(0).cpu().numpy()

    # Split into overlapping chunks and mean-pool each
    chunks = _split_and_pool(
        token_embeddings, content_start, content_end, chunk_size, stride
    )

    return chunks


def _split_and_pool(
    token_embeddings: "np.ndarray",
    content_start: int,
    content_end: int,
    chunk_size: int,
    stride: int,
) -> List["np.ndarray"]:
    """Split token embeddings into overlapping chunks and mean-pool each.

    Parameters
    ----------
    token_embeddings : np.ndarray
        Shape ``(seq_len, hidden_dim)`` — all token embeddings including
        special tokens.
    content_start : int
        Index of first content token (after [CLS]).
    content_end : int
        Index one past last content token (before [SEP]).
    chunk_size : int
        Tokens per chunk.
    stride : int
        Step between chunk starts.

    Returns
    -------
    list[np.ndarray]
        Mean-pooled chunk embeddings, each of shape ``(hidden_dim,)``.
    """
    chunks = []
    pos = content_start
    while pos < content_end:
        end = min(pos + chunk_size, content_end)
        chunk_tokens = token_embeddings[pos:end]
        # Mean pool across the token dimension
        chunk_embedding = chunk_tokens.mean(axis=0)
        # L2 normalize
        norm = np.linalg.norm(chunk_embedding)
        if norm > 0:
            chunk_embedding = chunk_embedding / norm
        chunks.append(chunk_embedding)
        pos += stride
        # Avoid duplicate tiny trailing chunk
        if end == content_end:
            break

    return chunks


# ---------------------------------------------------------------------------
# Core: late chunking classify
# ---------------------------------------------------------------------------

def late_chunk_classify(
    text: str,
    model: Any,
    tokenizer: Any,
    classifier: Any,
    chunk_size: int = 128,
    stride: int = 64,
) -> Optional[Dict[str, Any]]:
    """Embed via late chunking, classify each chunk, return max-risk score.

    Parameters
    ----------
    text : str
        Full document text.
    model : transformers.PreTrainedModel
        HuggingFace transformer model.
    tokenizer : transformers.PreTrainedTokenizer
        Matching tokenizer.
    classifier : object
        Any classifier with ``.classify(text)`` returning
        ``(score, technique_matches)`` — compatible with
        ``EmbeddingClassifier``.  Alternatively, any object with
        ``.predict_proba(embedding)`` for sklearn-style classifiers.
    chunk_size : int
        Tokens per chunk (default 128).
    stride : int
        Step between chunk starts (default 64).

    Returns
    -------
    dict or None
        ``{"max_score": float, "max_chunk_idx": int,
           "chunk_scores": list[float], "num_chunks": int}``
        Returns *None* if late chunking is not applicable (short text,
        missing dependencies, etc.).
    """
    if not _HAS_TRANSFORMERS:
        return None

    chunk_embeddings = late_chunk_embed(
        text, model, tokenizer, chunk_size=chunk_size, stride=stride
    )
    if chunk_embeddings is None or len(chunk_embeddings) == 0:
        return None

    # Classify each chunk embedding
    chunk_scores: List[float] = []

    for chunk_emb in chunk_embeddings:
        score = _classify_single_embedding(chunk_emb, classifier)
        chunk_scores.append(score)

    max_score = max(chunk_scores)
    max_idx = chunk_scores.index(max_score)

    return {
        "max_score": max_score,
        "max_chunk_idx": max_idx,
        "chunk_scores": chunk_scores,
        "num_chunks": len(chunk_scores),
    }


def _classify_single_embedding(
    embedding: "np.ndarray",
    classifier: Any,
) -> float:
    """Classify a single chunk embedding and return its risk score.

    Supports two classifier interfaces:
    1. Centroid-based (EmbeddingClassifier): uses cosine similarity between
       the chunk embedding and attack centroids.
    2. sklearn-style: has ``predict_proba`` method.

    Parameters
    ----------
    embedding : np.ndarray
        Shape ``(hidden_dim,)`` — a single chunk embedding.
    classifier : object
        Classifier instance.

    Returns
    -------
    float
        Risk score in [0.0, 1.0].
    """
    # Centroid-based classifier path
    if hasattr(classifier, "_centroids") and classifier._centroids is not None:
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        max_sim = 0.0
        for centroid in classifier._centroids.values():
            sim = float(np.dot(embedding, centroid))
            if sim > max_sim:
                max_sim = sim
        return max_sim

    # sklearn-style classifier path
    if hasattr(classifier, "predict_proba"):
        try:
            proba = classifier.predict_proba(embedding.reshape(1, -1))[0]
            # Assume class 1 = malicious
            return float(proba[1]) if len(proba) > 1 else float(proba[0])
        except Exception as exc:
            logger.warning("predict_proba failed on chunk: %s", exc)
            return 0.0

    return 0.0


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_late_chunking_model(
    model_name: str = DEFAULT_MODEL_NAME,
) -> Optional[Tuple[Any, Any]]:
    """Load a HuggingFace transformer model + tokenizer for late chunking.

    Parameters
    ----------
    model_name : str
        Model name.  If it does not contain a ``/``, the
        ``sentence-transformers/`` prefix is added automatically.

    Returns
    -------
    tuple[model, tokenizer] or None
        Returns *None* if transformers is not installed.
    """
    if not _HAS_TRANSFORMERS:
        return None

    if "/" not in model_name:
        model_name = _HUGGINGFACE_PREFIX + model_name

    logger.info("Loading late-chunking model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Integration helper for predict_embedding.py
# ---------------------------------------------------------------------------

def maybe_late_chunk_boost(
    text: str,
    full_text_score: float,
    classifier: Any,
    model: Any = None,
    tokenizer: Any = None,
    chunk_size: int = 128,
    stride: int = 64,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """Optionally boost a scan score using late chunking.

    Called after the main classification pass.  Only activates if:
    1. ``NA0S_LATE_CHUNKING=1`` is set
    2. The text is long enough (>= ``NA0S_LATE_CHUNK_MIN_TOKENS`` tokens)
    3. sentence-transformers / transformers is available

    If any late-chunked chunk scores higher than ``full_text_score``,
    the higher score is returned.

    Parameters
    ----------
    text : str
        Full document text (post-sanitization).
    full_text_score : float
        Score from the main (full-text) classification pass.
    classifier : object
        Classifier compatible with ``late_chunk_classify``.
    model : transformers.PreTrainedModel or None
        If None and late chunking is enabled, the model is loaded on demand.
    tokenizer : transformers.PreTrainedTokenizer or None
        If None and late chunking is enabled, loaded with the model.
    chunk_size : int
        Tokens per chunk.
    stride : int
        Stride between chunk starts.

    Returns
    -------
    tuple[float, dict or None]
        ``(final_score, chunk_details)`` — final_score is
        ``max(full_text_score, best_chunk_score)``.
        chunk_details is the dict from ``late_chunk_classify`` or None.
    """
    if not is_late_chunking_enabled():
        return full_text_score, None

    if not _HAS_TRANSFORMERS:
        logger.debug("Late chunking requested but transformers not available")
        return full_text_score, None

    # Lazy-load model if not provided
    if model is None or tokenizer is None:
        loaded = load_late_chunking_model()
        if loaded is None:
            return full_text_score, None
        model, tokenizer = loaded

    result = late_chunk_classify(
        text, model, tokenizer, classifier,
        chunk_size=chunk_size, stride=stride,
    )

    if result is None:
        return full_text_score, None

    best_chunk_score = result["max_score"]
    final_score = max(full_text_score, best_chunk_score)

    if best_chunk_score > full_text_score:
        logger.info(
            "Late chunking boosted score: %.3f -> %.3f (chunk %d/%d)",
            full_text_score, best_chunk_score,
            result["max_chunk_idx"], result["num_chunks"],
        )

    return final_score, result


def is_late_chunking_enabled() -> bool:
    """Check if late chunking is enabled via environment variable."""
    return os.environ.get("NA0S_LATE_CHUNKING", "0") == "1"
