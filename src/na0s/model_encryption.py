"""AES-256-GCM encryption for model files at rest (Layer 11).

Encrypts and decrypts model files using AES-256-GCM.
Gated behind ``NA0S_ENCRYPTION_KEY`` environment variable (64 hex chars = 32 bytes).

Requires the ``cryptography`` package.  The module is always importable, but
operations raise ``ImportError`` with a helpful message when the library is
missing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    _HAS_CRYPTO = True
except ImportError:
    _HAS_CRYPTO = False

_NONCE_LEN = 12
_TAG_LEN = 16


def _require_crypto() -> None:
    if not _HAS_CRYPTO:
        raise ImportError(
            "The 'cryptography' package is required for model encryption. "
            "Install it with:  pip install cryptography"
        )


class ModelEncryptor:
    """Encrypt / decrypt model artefacts with AES-256-GCM.

    Parameters
    ----------
    key : bytes | str | None
        A 32-byte key or 64 hex-character string.  When *None* the key is
        read from the ``NA0S_ENCRYPTION_KEY`` environment variable.
    """

    def __init__(self, key: Optional[bytes | str] = None) -> None:
        _require_crypto()

        if key is None:
            key = os.environ.get("NA0S_ENCRYPTION_KEY")
        if key is None:
            raise ValueError(
                "No encryption key provided. Set NA0S_ENCRYPTION_KEY env var "
                "or pass a key to ModelEncryptor()."
            )

        if isinstance(key, str):
            # Accept 64 hex chars → 32 bytes
            key = bytes.fromhex(key)

        if len(key) != 32:
            raise ValueError(
                f"Key must be exactly 32 bytes (got {len(key)}). "
                "Provide 64 hex characters or 32 raw bytes."
            )

        self._key: bytes = key
        self._aesgcm = AESGCM(self._key)

    # ------------------------------------------------------------------
    # Byte-level API
    # ------------------------------------------------------------------

    def encrypt_bytes(self, data: bytes) -> bytes:
        """Return *nonce* (12 B) + *tag* (16 B) + *ciphertext*.

        ``AESGCM.encrypt`` returns *ciphertext || tag*; we rearrange to
        nonce || tag || ciphertext for a fixed-offset layout.
        """
        nonce = os.urandom(_NONCE_LEN)
        ct_and_tag = self._aesgcm.encrypt(nonce, data, None)
        # cryptography appends the 16-byte tag at the end
        ciphertext = ct_and_tag[:-_TAG_LEN]
        tag = ct_and_tag[-_TAG_LEN:]
        return nonce + tag + ciphertext

    def decrypt_bytes(self, data: bytes) -> bytes:
        """Parse *nonce || tag || ciphertext* and return plaintext."""
        if len(data) < _NONCE_LEN + _TAG_LEN:
            raise ValueError("Data too short to contain nonce + tag.")
        nonce = data[:_NONCE_LEN]
        tag = data[_NONCE_LEN : _NONCE_LEN + _TAG_LEN]
        ciphertext = data[_NONCE_LEN + _TAG_LEN :]
        # cryptography expects ciphertext || tag
        return self._aesgcm.decrypt(nonce, ciphertext + tag, None)

    # ------------------------------------------------------------------
    # File-level API
    # ------------------------------------------------------------------

    def encrypt_file(
        self, src_path: str | Path, dst_path: Optional[str | Path] = None
    ) -> Path:
        """Encrypt *src_path* → *dst_path* (default ``{src}.enc``)."""
        src = Path(src_path)
        if dst_path is None:
            dst = src.with_suffix(src.suffix + ".enc")
        else:
            dst = Path(dst_path)

        plaintext = src.read_bytes()
        dst.write_bytes(self.encrypt_bytes(plaintext))
        return dst

    def decrypt_file(
        self, src_path: str | Path, dst_path: Optional[str | Path] = None
    ) -> Path:
        """Decrypt *src_path* → *dst_path* (default strips ``.enc``)."""
        src = Path(src_path)
        if dst_path is None:
            name = str(src)
            if name.endswith(".enc"):
                dst = Path(name[:-4])
            else:
                dst = src.with_suffix(".dec")
        else:
            dst = Path(dst_path)

        blob = src.read_bytes()
        dst.write_bytes(self.decrypt_bytes(blob))
        return dst
