"""Regression tests for Layer 13 pipeline hardening fixes."""

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Ensure repo root importability
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts import process_data
from scripts import mine_hard_negatives
from scripts.optimize_threshold import _compute_oof_probabilities


class TestProcessDataFixes(unittest.TestCase):
    """Covers BUG-L13-1/2/5/7 behavior in process_data.py."""

    def _write_csv(self, path, rows):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerows(rows)

    def _write_jsonl(self, path, rows):
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def test_text_hash_nfkc_equivalent(self):
        self.assertEqual(
            process_data._text_hash("ＡＢＣ prompt"),
            process_data._text_hash("ABC prompt"),
        )

    def test_merge_datasets_unicode_dedup_stable_order_and_generated_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "data", "raw")
            aggregated = os.path.join(tmp, "data", "aggregated")
            harvest = os.path.join(tmp, "data", "harvest")
            holdout = os.path.join(tmp, "data", "holdout")
            benchmark = os.path.join(tmp, "data", "benchmark")
            processed = os.path.join(tmp, "data", "processed")
            out_csv = os.path.join(processed, "combined_data.csv")
            for d in (raw, aggregated, harvest, holdout, benchmark, processed):
                os.makedirs(d, exist_ok=True)

            # Raw CSV includes Unicode-equivalent duplicates + hard negatives.
            self._write_csv(
                os.path.join(raw, "base.csv"),
                [
                    {"text": "ＡＢＣ prompt", "label": 1},
                    {"text": "ABC prompt", "label": 1},
                ],
            )
            self._write_csv(
                os.path.join(raw, "hard_negatives.csv"),
                [{"text": "Benign hard negative sample", "label": 0}],
            )

            # Generated synthetic outputs (BUG-L13-7) must be ingested.
            self._write_jsonl(
                os.path.join(holdout, "malicious_holdout.jsonl"),
                [{"text": "holdout sample", "label": 1}],
            )
            self._write_jsonl(
                os.path.join(benchmark, "adversarial_evasion.jsonl"),
                [{"text": "benchmark sample", "label": 1}],
            )

            with patch.multiple(
                process_data,
                ROOT=tmp,
                RAW_DIR=raw,
                AGGREGATED_DIR=aggregated,
                HARVEST_DIR=harvest,
                HOLDOUT_DIR=holdout,
                BENCHMARK_DIR=benchmark,
                OUTPUT_PATH=out_csv,
            ):
                first = process_data.merge_datasets()
                second = process_data.merge_datasets()

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertTrue(os.path.isfile(out_csv))

            out = pd.read_csv(out_csv)
            texts = out["text"].astype(str).tolist()

            self.assertIn("Benign hard negative sample", texts)
            self.assertIn("holdout sample", texts)
            self.assertIn("benchmark sample", texts)

            # Unicode-equivalent duplicates should collapse to a single row.
            canon_count = sum(t in ("ＡＢＣ prompt", "ABC prompt") for t in texts)
            self.assertEqual(canon_count, 1)

            # Output should be idempotently ordered by hash.
            hashes = [process_data._text_hash(t) for t in texts]
            self.assertEqual(hashes, sorted(hashes))

            # Re-running merge should preserve exact ordering/content.
            self.assertEqual(
                first.to_dict("records"),
                second.to_dict("records"),
            )


class TestOptimizeThresholdFixes(unittest.TestCase):
    """Covers BUG-L13-3 cross-validation threshold optimization."""

    def test_oof_probabilities_shape_and_range(self):
        X = np.array([
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, 0.1],
            [1.0, 1.0],
            [0.9, 1.1],
            [1.1, 0.9],
        ])
        y = np.array([0, 0, 0, 1, 1, 1])
        model = LogisticRegression(max_iter=200, random_state=42)

        probs = _compute_oof_probabilities(model, X, y, n_splits=3)

        self.assertEqual(len(probs), len(y))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        self.assertGreater(float(np.std(probs)), 0.0)


class TestHardNegativeMergeFixes(unittest.TestCase):
    """Covers BUG-L13-1/5 merge behavior in mine_hard_negatives.py."""

    def test_phase4_updates_canonical_and_is_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            merged_csv = os.path.join(tmp, "combined_data_with_negatives.csv")
            combined_csv = os.path.join(tmp, "combined_data.csv")

            original_df = pd.DataFrame(
                [
                    {"text": "ＡＢＣ prompt", "label": 1, "technique_id": "", "category": ""},
                    {"text": "Safe baseline", "label": 0, "technique_id": "", "category": ""},
                ]
            )
            hard_neg_df = pd.DataFrame(
                [
                    {
                        "text": "ABC prompt",
                        "label": 0,
                        "technique_id": "",
                        "category": "",
                        "source": "hardneg",
                    },
                    {
                        "text": "New hard negative",
                        "label": 0,
                        "technique_id": "",
                        "category": "",
                        "source": "hardneg",
                    },
                ]
            )

            with patch.multiple(
                mine_hard_negatives,
                MERGED_CSV=merged_csv,
                COMBINED_CSV=combined_csv,
            ):
                mine_hard_negatives.phase4_merge(original_df, hard_neg_df)
                mine_hard_negatives.phase4_merge(original_df, hard_neg_df)

            self.assertTrue(os.path.isfile(merged_csv))
            self.assertTrue(os.path.isfile(combined_csv))

            out = pd.read_csv(combined_csv)
            texts = out["text"].astype(str).tolist()
            self.assertIn("New hard negative", texts)
            # Unicode-equivalent duplicate should only appear once.
            self.assertEqual(sum(t in ("ＡＢＣ prompt", "ABC prompt") for t in texts), 1)

            # Idempotent deterministic order by normalized hash
            self.assertEqual(texts, sorted(texts, key=process_data._text_hash))


class TestMergeTaxonomyFieldLimitFix(unittest.TestCase):
    """Covers BUG-L13-4 bounded CSV field-size limit."""

    def test_no_sys_maxsize_field_limit(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "scripts" / "merge_taxonomy_data.py"
        text = path.read_text(encoding="utf-8")
        self.assertNotIn("sys.maxsize", text)
        self.assertIn("NA0S_CSV_FIELD_LIMIT", text)


if __name__ == "__main__":
    unittest.main()
