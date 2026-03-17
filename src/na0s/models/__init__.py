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
    "model.pkl": "057280f9fd4558796ce76df3c1eefbccbdb39d35097e5e1b4eba96e54b5b1594",
    "structural_scaler.pkl": "51f1e0791a9caff6e0f554950b8f83f79da321cc87e321449893369c5374c192",
    "model_embedding.pkl": "09209a059188b2417bb79687218b1e6ef10714025a31b1e7283de42dbec15c2e",
    "tfidf_vectorizer.pkl": "347b2b4ebdbf6a69674d18d9a9c242da2d415f274463a24af197eed2d68f90a4",
}


def get_model_path(filename):
    """Return the absolute path to a bundled model file as a string."""
    return str(importlib.resources.files(__package__) / filename)
