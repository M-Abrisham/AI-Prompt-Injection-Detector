"""Model weight storage and path resolution.

Pre-trained model weights are bundled inside the package at install
time.  Use ``get_model_path(filename)`` to get an absolute path to a
bundled model file.

Hardcoded SHA-256 hashes
~~~~~~~~~~~~~~~~~~~~~~~~
The ``KNOWN_HASHES`` dict maps each bundled ``.pkl`` filename to its
expected SHA-256 hex digest.  Because these hashes live *inside* the
Python source (which is itself signed by pip's wheel signature), an
attacker who tampers with a ``.pkl`` file cannot update the expected
hash without also patching this module.  This eliminates the "security
theater" problem of shipping ``.sha256`` sidecar files next to the very
artefacts they are supposed to protect.

The sidecar files are kept for backward compatibility (e.g. user-trained
models), but ``safe_load()`` will always prefer the hardcoded hash when
one is available.
"""

import importlib.resources

# Authoritative SHA-256 hex digests for every bundled pickle file.
# Update this dict whenever a model is retrained.
KNOWN_HASHES = {
    "model.pkl": "db28b5c8b952c75830a59de63a564dfac70f72fb7fab7a089f90301595491319",
    "tfidf_vectorizer.pkl": "280c8705936527b5cbbaacb206d13a8f366feb503ba88a4c792515e61c0c0ac4",
    "model_embedding.pkl": "09209a059188b2417bb79687218b1e6ef10714025a31b1e7283de42dbec15c2e",
}


def get_model_path(filename):
    """Return the absolute path to a bundled model file as a string."""
    return str(importlib.resources.files(__package__) / filename)
