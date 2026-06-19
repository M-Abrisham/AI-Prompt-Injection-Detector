"""Layer 5: Adapter layer on top of frozen sentence-transformer embeddings.

Adds a small trainable 2-layer MLP (adapter) on top of frozen embeddings
from a sentence-transformer. Only the adapter weights are trained, keeping
the embedding model frozen for efficiency.

Architecture:
    frozen sentence-transformer -> 384-dim embedding
        -> Linear(384, hidden_dim) -> ReLU -> Dropout
        -> Linear(hidden_dim, 2) -> output logits

Graceful degradation: if torch is not installed, exports placeholder classes
that raise informative errors on use.

Usage (programmatic):
    from na0s.embedding_adapter import AdapterClassifier, train_adapter
    clf = AdapterClassifier(model_name="all-MiniLM-L6-v2")
    clf.train(texts, labels, epochs=10)
    score = clf.predict_proba(text)
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency guard: torch is optional
# ---------------------------------------------------------------------------
_HAS_TORCH = False
_torch_import_error: Optional[str] = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    _HAS_TORCH = True
except ImportError as exc:
    _torch_import_error = str(exc)

# sentence-transformers is also optional (only needed for AdapterClassifier)
_HAS_SENTENCE_TRANSFORMERS = False
_st_import_error: Optional[str] = None

try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError as exc:
    _st_import_error = str(exc)

# Shared pinned loader: revision-pins all-MiniLM-L6-v2 (the default model) so
# the encoder snapshot is deterministic across runs.
from na0s.ml._st_loader import load_pinned_sentence_transformer


# ---------------------------------------------------------------------------
# EmbeddingAdapter — the trainable MLP head
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class EmbeddingAdapter(nn.Module):
        """Small 2-layer MLP adapter on top of frozen embeddings.

        Parameters
        ----------
        input_dim : int
            Dimensionality of input embeddings (384 for all-MiniLM-L6-v2).
        hidden_dim : int
            Hidden layer size.
        num_classes : int
            Number of output classes (2 for binary: safe/malicious).
        dropout : float
            Dropout probability between layers.
        """

        def __init__(
            self,
            input_dim: int = 384,
            hidden_dim: int = 128,
            num_classes: int = 2,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Parameters
            ----------
            x : torch.Tensor
                Input embeddings of shape ``(batch_size, input_dim)``.

            Returns
            -------
            torch.Tensor
                Logits of shape ``(batch_size, num_classes)``.
            """
            return self.net(x)

else:

    class EmbeddingAdapter:  # type: ignore[no-redef]
        """Placeholder when torch is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for EmbeddingAdapter. "
                "Install with: pip install torch\n"
                "Import error: {0}".format(_torch_import_error)
            )


# ---------------------------------------------------------------------------
# train_adapter — standalone training function
# ---------------------------------------------------------------------------

def train_adapter(
    embeddings: np.ndarray,
    labels: np.ndarray,
    input_dim: int = 384,
    hidden_dim: int = 128,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    dropout: float = 0.3,
    val_split: float = 0.1,
    device: Optional[str] = None,
) -> object:
    """Train an adapter MLP on pre-computed frozen embeddings.

    Parameters
    ----------
    embeddings : np.ndarray
        Pre-computed embeddings, shape ``(n_samples, input_dim)``.
    labels : np.ndarray
        Binary labels (0=safe, 1=malicious), shape ``(n_samples,)``.
    input_dim : int
        Embedding dimensionality (must match embeddings.shape[1]).
    hidden_dim : int
        Hidden layer size for the adapter.
    epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    learning_rate : float
        Learning rate for Adam optimizer.
    dropout : float
        Dropout probability.
    val_split : float
        Fraction of data to hold out for validation.
    device : str or None
        Device string ('cpu', 'cuda', 'mps'). Auto-detected if None.

    Returns
    -------
    EmbeddingAdapter
        Trained adapter model (in eval mode).

    Raises
    ------
    ImportError
        If torch is not installed.
    """
    if not _HAS_TORCH:
        raise ImportError(
            "PyTorch is required for adapter training. "
            "Install with: pip install torch\n"
            "Import error: {0}".format(_torch_import_error)
        )

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dev = torch.device(device)
    logger.info("Training adapter on device: %s", dev)

    # Validate shapes
    if embeddings.shape[1] != input_dim:
        input_dim = embeddings.shape[1]
        logger.info("Auto-detected input_dim=%d from embeddings", input_dim)

    # Train/val split
    n = len(labels)
    indices = np.random.RandomState(42).permutation(n)
    val_n = max(1, int(n * val_split))
    val_idx, train_idx = indices[:val_n], indices[val_n:]

    X_train = torch.tensor(embeddings[train_idx], dtype=torch.float32)
    y_train = torch.tensor(labels[train_idx], dtype=torch.long)
    X_val = torch.tensor(embeddings[val_idx], dtype=torch.float32)
    y_val = torch.tensor(labels[val_idx], dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Build adapter
    adapter = EmbeddingAdapter(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=2,
        dropout=dropout,
    ).to(dev)

    optimizer = optim.Adam(adapter.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        adapter.train()
        total_loss = 0.0
        n_batches = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(dev), batch_y.to(dev)
            optimizer.zero_grad()
            logits = adapter(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation
        adapter.eval()
        with torch.no_grad():
            val_logits = adapter(X_val.to(dev))
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = (val_preds == y_val.numpy()).mean()

        avg_loss = total_loss / max(n_batches, 1)
        logger.info(
            "Epoch %d/%d — loss: %.4f, val_acc: %.4f",
            epoch + 1, epochs, avg_loss, val_acc,
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}

    # Restore best checkpoint
    if best_state is not None:
        adapter.load_state_dict(best_state)

    adapter.eval()
    logger.info("Training complete. Best val accuracy: %.4f", best_val_acc)
    return adapter


# ---------------------------------------------------------------------------
# AdapterClassifier — wraps frozen encoder + trainable adapter
# ---------------------------------------------------------------------------

class AdapterClassifier:
    """Frozen sentence-transformer + trainable adapter for classification.

    The sentence-transformer produces embeddings (frozen, no gradient).
    A small MLP adapter is trained on top for binary classification.

    Parameters
    ----------
    model_name : str
        Sentence-transformer model name.
    hidden_dim : int
        Hidden dimension for the adapter MLP.
    dropout : float
        Dropout probability.
    device : str or None
        Device string. Auto-detected if None.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        hidden_dim: int = 128,
        dropout: float = 0.3,
        device: Optional[str] = None,
    ):
        if not _HAS_TORCH:
            raise ImportError(
                "PyTorch is required for AdapterClassifier. "
                "Install with: pip install torch\n"
                "Import error: {0}".format(_torch_import_error)
            )
        if not _HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers is required for AdapterClassifier. "
                "Install with: pip install sentence-transformers\n"
                "Import error: {0}".format(_st_import_error)
            )

        self._model_name = model_name
        self._hidden_dim = hidden_dim
        self._dropout = dropout
        self._device = device
        self._encoder: Optional[object] = None
        self._adapter: Optional[object] = None
        self._input_dim: Optional[int] = None

    def _ensure_encoder(self):
        """Lazy-load the sentence-transformer encoder."""
        if self._encoder is None:
            self._encoder = load_pinned_sentence_transformer(
                SentenceTransformer, self._model_name,
            )
            # Determine embedding dimension from model
            self._input_dim = self._encoder.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts using the frozen sentence-transformer.

        Parameters
        ----------
        texts : list[str]
            Input texts.

        Returns
        -------
        np.ndarray
            Embeddings of shape ``(len(texts), input_dim)``.
        """
        self._ensure_encoder()
        return self._encoder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=64,
        )

    def train(
        self,
        texts: List[str],
        labels: np.ndarray,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ) -> None:
        """Encode texts and train the adapter on top.

        Parameters
        ----------
        texts : list[str]
            Training texts.
        labels : np.ndarray
            Binary labels (0=safe, 1=malicious).
        epochs : int
            Training epochs.
        batch_size : int
            Batch size.
        learning_rate : float
            Learning rate.
        """
        self._ensure_encoder()
        embeddings = self.encode(texts)
        self._adapter = train_adapter(
            embeddings=embeddings,
            labels=labels,
            input_dim=self._input_dim,
            hidden_dim=self._hidden_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dropout=self._dropout,
            device=self._device,
        )

    def predict_proba(self, text: str) -> float:
        """Return P(malicious) for a single text.

        Parameters
        ----------
        text : str
            Input text to classify.

        Returns
        -------
        float
            Probability of being malicious (0.0 to 1.0).
        """
        if self._adapter is None:
            raise RuntimeError("Adapter not trained yet. Call train() first.")

        embedding = self.encode([text])
        x = torch.tensor(embedding, dtype=torch.float32)

        self._adapter.eval()
        with torch.no_grad():
            logits = self._adapter(x)
            probs = torch.softmax(logits, dim=1)
        return float(probs[0, 1])

    def save(self, path: str) -> None:
        """Save the adapter weights to disk.

        Parameters
        ----------
        path : str
            File path for the saved state dict.
        """
        if self._adapter is None:
            raise RuntimeError("No adapter to save. Call train() first.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._adapter.state_dict(), path)
        logger.info("Adapter saved to %s", path)

    def load(self, path: str, input_dim: int = 384) -> None:
        """Load adapter weights from disk.

        Parameters
        ----------
        path : str
            File path to the saved state dict.
        input_dim : int
            Embedding dimension (must match the saved adapter).
        """
        self._input_dim = input_dim
        self._adapter = EmbeddingAdapter(
            input_dim=input_dim,
            hidden_dim=self._hidden_dim,
            dropout=self._dropout,
        )
        self._adapter.load_state_dict(torch.load(path, map_location="cpu"))
        self._adapter.eval()
        logger.info("Adapter loaded from %s", path)
