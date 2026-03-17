"""Error-handling tests for scripts/features.py and scripts/model.py.

Covers:
- features.py: missing input file, empty CSV, missing 'text' column,
  missing 'label' column, single-class labels.
- model.py: missing features file, invalid features format (wrong type,
  wrong tuple length, empty matrix, too few samples, single-class labels).

All ``sys.exit(1)`` calls are captured via ``SystemExit`` assertions so
the test suite never actually terminates.

Run with:
    python -m unittest tests.test_pipeline_error_handling -v
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

# Ensure the src package is importable when running from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path, content):
    """Write *content* (str) to *path*."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)


# ---------------------------------------------------------------------------
# features.py tests
# ---------------------------------------------------------------------------

class TestFeaturesErrorHandling(unittest.TestCase):
    """Guard-rail tests for scripts/features.py :: load_training_data()."""

    def _run(self, input_path):
        """Import and call load_training_data() with INPUT_PATH overridden."""
        import scripts.features as feat
        with patch.object(feat, "INPUT_PATH", input_path):
            feat.load_training_data()

    # ------------------------------------------------------------------
    # 1. Missing input file
    # ------------------------------------------------------------------
    def test_missing_input_file_exits_1(self):
        """load_training_data() exits 1 when the input CSV does not exist."""
        with self.assertRaises(SystemExit) as ctx:
            self._run("/nonexistent/path/combined_data.csv")
        self.assertEqual(ctx.exception.code, 1)

    # ------------------------------------------------------------------
    # 2. Empty CSV (zero data rows)
    # ------------------------------------------------------------------
    def test_empty_csv_exits_1(self):
        """load_training_data() exits 1 when the CSV has no data rows."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as fh:
            fh.write("text,label\n")
            tmp_path = fh.name
        try:
            with self.assertRaises(SystemExit) as ctx:
                self._run(tmp_path)
            self.assertEqual(ctx.exception.code, 1)
        finally:
            os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # 3. CSV missing the 'text' column
    # ------------------------------------------------------------------
    def test_missing_text_column_exits_1(self):
        """load_training_data() exits 1 when the 'text' column is absent."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as fh:
            fh.write("content,label\n")
            fh.write("ignore me,0\n")
            fh.write("ignore me too,1\n")
            tmp_path = fh.name
        try:
            with self.assertRaises(SystemExit) as ctx:
                self._run(tmp_path)
            self.assertEqual(ctx.exception.code, 1)
        finally:
            os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # 4. CSV missing the 'label' column
    # ------------------------------------------------------------------
    def test_missing_label_column_exits_1(self):
        """load_training_data() exits 1 when the 'label' column is absent."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as fh:
            fh.write("text,class\n")
            fh.write("hello,0\n")
            fh.write("world,1\n")
            tmp_path = fh.name
        try:
            with self.assertRaises(SystemExit) as ctx:
                self._run(tmp_path)
            self.assertEqual(ctx.exception.code, 1)
        finally:
            os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # 5. Single-class labels (only 0s)
    # ------------------------------------------------------------------
    def test_single_class_only_zeros_exits_1(self):
        """load_training_data() exits 1 when all labels are 0 (no positives)."""
        rows = "\n".join(
            [f"sample text {i},0" for i in range(20)]
        )
        csv_content = "text,label\n" + rows + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as fh:
            fh.write(csv_content)
            tmp_path = fh.name
        try:
            with self.assertRaises(SystemExit) as ctx:
                self._run(tmp_path)
            self.assertEqual(ctx.exception.code, 1)
        finally:
            os.unlink(tmp_path)

    # ------------------------------------------------------------------
    # 6. Single-class labels (only 1s)
    # ------------------------------------------------------------------
    def test_single_class_only_ones_exits_1(self):
        """load_training_data() exits 1 when all labels are 1 (no negatives)."""
        rows = "\n".join(
            [f"inject override {i},1" for i in range(20)]
        )
        csv_content = "text,label\n" + rows + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as fh:
            fh.write(csv_content)
            tmp_path = fh.name
        try:
            with self.assertRaises(SystemExit) as ctx:
                self._run(tmp_path)
            self.assertEqual(ctx.exception.code, 1)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# model.py tests
# ---------------------------------------------------------------------------

class TestModelErrorHandling(unittest.TestCase):
    """Guard-rail tests for scripts/model.py :: train_model()."""

    def _run(self, features_path):
        """Import and call train_model() with FEATURES_PATH overridden."""
        import scripts.model as mdl
        with patch.object(mdl, "FEATURES_PATH", features_path):
            mdl.train_model()

    # ------------------------------------------------------------------
    # 7. Missing features file
    # ------------------------------------------------------------------
    def test_missing_features_file_exits_1(self):
        """train_model() exits 1 when the features pickle does not exist."""
        with self.assertRaises(SystemExit) as ctx:
            self._run("/nonexistent/path/features.pkl")
        self.assertEqual(ctx.exception.code, 1)

    # ------------------------------------------------------------------
    # 8. Features file contains wrong type (not a tuple)
    # ------------------------------------------------------------------
    def test_invalid_features_not_tuple_exits_1(self):
        """train_model() exits 1 when safe_load returns a non-tuple value."""
        import scripts.model as mdl
        with patch("scripts.model.safe_load", return_value="not a tuple"):
            with patch.object(mdl, "FEATURES_PATH", "/fake/features.pkl"):
                with patch("os.path.isfile", return_value=True):
                    with self.assertRaises(SystemExit) as ctx:
                        mdl.train_model()
        self.assertEqual(ctx.exception.code, 1)

    # ------------------------------------------------------------------
    # 9. Features file contains a 1-tuple (too short)
    # ------------------------------------------------------------------
    def test_invalid_features_1tuple_exits_1(self):
        """train_model() exits 1 when safe_load returns a 1-element tuple."""
        import scripts.model as mdl
        with patch("scripts.model.safe_load", return_value=("only_X",)):
            with patch.object(mdl, "FEATURES_PATH", "/fake/features.pkl"):
                with patch("os.path.isfile", return_value=True):
                    with self.assertRaises(SystemExit) as ctx:
                        mdl.train_model()
        self.assertEqual(ctx.exception.code, 1)

    # ------------------------------------------------------------------
    # 10. Feature matrix has 0 samples
    # ------------------------------------------------------------------
    def test_empty_feature_matrix_exits_1(self):
        """train_model() exits 1 when the feature matrix contains 0 rows."""
        import numpy as np
        from scipy.sparse import csr_matrix
        import scripts.model as mdl

        X = csr_matrix((0, 100))   # 0 rows, 100 features
        y = np.array([])

        with patch("scripts.model.safe_load", return_value=(X, y)):
            with patch.object(mdl, "FEATURES_PATH", "/fake/features.pkl"):
                with patch("os.path.isfile", return_value=True):
                    with self.assertRaises(SystemExit) as ctx:
                        mdl.train_model()
        self.assertEqual(ctx.exception.code, 1)

    # ------------------------------------------------------------------
    # 11. Too few samples (below _MIN_SAMPLES=100)
    # ------------------------------------------------------------------
    def test_too_few_samples_exits_1(self):
        """train_model() exits 1 when fewer than 100 samples are present."""
        import numpy as np
        from scipy.sparse import csr_matrix
        import scripts.model as mdl

        n = 50
        X = csr_matrix((n, 10))
        y = np.array([0] * (n // 2) + [1] * (n // 2))

        with patch("scripts.model.safe_load", return_value=(X, y)):
            with patch.object(mdl, "FEATURES_PATH", "/fake/features.pkl"):
                with patch("os.path.isfile", return_value=True):
                    with self.assertRaises(SystemExit) as ctx:
                        mdl.train_model()
        self.assertEqual(ctx.exception.code, 1)

    # ------------------------------------------------------------------
    # 12. Single-class labels in model.py (only 0s)
    # ------------------------------------------------------------------
    def test_single_class_labels_only_zeros_exits_1(self):
        """train_model() exits 1 when all labels in the feature set are 0."""
        import numpy as np
        from scipy.sparse import csr_matrix
        import scripts.model as mdl

        n = 200
        X = csr_matrix((n, 50))
        y = np.zeros(n, dtype=int)

        with patch("scripts.model.safe_load", return_value=(X, y)):
            with patch.object(mdl, "FEATURES_PATH", "/fake/features.pkl"):
                with patch("os.path.isfile", return_value=True):
                    with self.assertRaises(SystemExit) as ctx:
                        mdl.train_model()
        self.assertEqual(ctx.exception.code, 1)

    # ------------------------------------------------------------------
    # 13. Single-class labels in model.py (only 1s)
    # ------------------------------------------------------------------
    def test_single_class_labels_only_ones_exits_1(self):
        """train_model() exits 1 when all labels in the feature set are 1."""
        import numpy as np
        from scipy.sparse import csr_matrix
        import scripts.model as mdl

        n = 200
        X = csr_matrix((n, 50))
        y = np.ones(n, dtype=int)

        with patch("scripts.model.safe_load", return_value=(X, y)):
            with patch.object(mdl, "FEATURES_PATH", "/fake/features.pkl"):
                with patch("os.path.isfile", return_value=True):
                    with self.assertRaises(SystemExit) as ctx:
                        mdl.train_model()
        self.assertEqual(ctx.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
