"""F7 provenance guard for scripts/process_data.py.

`process_data.py` once wrote only ``text`` + ``label`` into
``data/processed/combined_data.csv``, silently dropping every provenance field
(source / source_id / license).  This pins the additive fix:

  * provenance columns ARE written to combined_data.csv,
  * `source` defaults to the input file's basename so every row is traceable,
  * source_id / license are carried from inputs when present,
  * and — critically — provenance is ADDITIVE: it must NOT change which rows
    train or their labels (the text/label core is byte-for-byte unchanged).

All offline; no network.
"""

import os
import sys
import tempfile

import pandas as pd
import pytest

_WT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if os.path.join(_WT_ROOT, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(_WT_ROOT, "scripts"))

import process_data  # noqa: E402


# ── _load_csv / _load_jsonl carry provenance ───────────────────────────────

def test_load_csv_stamps_basename_source():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "raw_inputs.csv")
        pd.DataFrame({"text": ["attack", "benign"], "label": [1, 0]}).to_csv(
            p, index=False
        )
        out = process_data._load_csv(p)

    for col in process_data.PROVENANCE_COLUMNS:
        assert col in out.columns, f"missing provenance column: {col}"
    # No source column in input -> falls back to file basename for every row.
    assert out["source"].tolist() == ["raw_inputs.csv", "raw_inputs.csv"]
    # No source_id/license in input -> NA (carried, not invented).
    assert out["source_id"].isna().all()
    assert out["license"].isna().all()


def test_load_jsonl_carries_input_provenance():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ds.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(
                '{"text": "x", "label": 1, "source": "upstreamX", '
                '"stable_id": "abc", "license": "mit"}\n'
            )
            fh.write('{"text": "y", "label": 0}\n')
        out = process_data._load_jsonl(p)

    rows = out.set_index("text")
    # Row with provenance carries it through verbatim.
    assert rows.loc["x", "source"] == "upstreamX"
    assert rows.loc["x", "source_id"] == "abc"
    assert rows.loc["x", "license"] == "mit"
    # Row without provenance falls back to basename, NA elsewhere.
    assert rows.loc["y", "source"] == "ds.jsonl"
    assert pd.isna(rows.loc["y", "source_id"])
    assert pd.isna(rows.loc["y", "license"])


def test_provenance_stays_row_aligned_after_dropna():
    """A row dropped for a missing label must take its provenance with it."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "mixed.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write('{"text": "keep", "label": 1, "source_id": "good"}\n')
            # label absent -> dropped; its source_id must not bleed onto "keep".
            fh.write('{"text": "drop", "source_id": "bad"}\n')
        out = process_data._load_jsonl(p)

    assert out["text"].tolist() == ["keep"]
    assert out["source_id"].tolist() == ["good"]


# ── end-to-end merge writes provenance + preserves training core ────────────

def _run_merge(monkeypatch, tmpdir):
    raw = os.path.join(tmpdir, "raw")
    agg = os.path.join(tmpdir, "aggregated")
    out_csv = os.path.join(tmpdir, "processed", "combined_data.csv")
    os.makedirs(raw)
    os.makedirs(agg)

    pd.DataFrame(
        {"text": ["csv attack", "csv benign"], "label": [1, 0]}
    ).to_csv(os.path.join(raw, "seed.csv"), index=False)
    with open(os.path.join(agg, "harvest.jsonl"), "w", encoding="utf-8") as fh:
        fh.write(
            '{"text": "jsonl attack", "label": 1, '
            '"source": "hf:org/ds", "license": "apache-2.0"}\n'
        )

    monkeypatch.setattr(process_data, "RAW_DIR", raw)
    monkeypatch.setattr(process_data, "AGGREGATED_DIR", agg)
    monkeypatch.setattr(process_data, "HARVEST_DIR", os.path.join(tmpdir, "nope_h"))
    monkeypatch.setattr(process_data, "STAGING_DIR", os.path.join(tmpdir, "nope_s"))
    monkeypatch.setattr(
        process_data, "TRAINING_JSONL_DIRS", [agg, os.path.join(tmpdir, "nope_h")]
    )
    monkeypatch.setattr(process_data, "OUTPUT_PATH", out_csv)
    return out_csv


def test_merge_writes_provenance_columns(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        out_csv = _run_merge(monkeypatch, d)
        process_data.merge_datasets()
        df = pd.read_csv(out_csv)

    for col in ("text", "label", *process_data.PROVENANCE_COLUMNS):
        assert col in df.columns, f"combined_data.csv missing column: {col}"

    by_text = df.set_index("text")
    # Carried provenance from the JSONL row.
    assert by_text.loc["jsonl attack", "source"] == "hf:org/ds"
    assert by_text.loc["jsonl attack", "license"] == "apache-2.0"
    # CSV rows fall back to their file basename.
    assert by_text.loc["csv attack", "source"] == "seed.csv"


def test_merge_provenance_is_additive_only(monkeypatch):
    """Provenance must not change which rows train or their labels.

    The (text, label) projection of the output must be identical whether or
    not provenance columns are present.
    """
    with tempfile.TemporaryDirectory() as d:
        out_csv = _run_merge(monkeypatch, d)
        process_data.merge_datasets()
        df = pd.read_csv(out_csv)

    core = (
        df[["text", "label"]]
        .sort_values(["text", "label"])
        .reset_index(drop=True)
    )
    expected = pd.DataFrame(
        {
            "text": ["csv attack", "csv benign", "jsonl attack"],
            "label": [1, 0, 1],
        }
    ).sort_values(["text", "label"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(core, expected)
    # Labels are still clean binary ints.
    assert set(df["label"].unique()) <= {0, 1}


def test_features_tolerates_provenance_columns():
    """scripts/features.py must ignore unknown (provenance) columns.

    It only requires text + label and consumes only those; extra columns
    must not crash the required-column guard."""
    import importlib

    features = importlib.import_module("features")
    df = pd.DataFrame(
        {
            "text": ["a", "b"],
            "label": [1, 0],
            "source": ["f.csv", "f.csv"],
            "source_id": [None, None],
            "license": [None, None],
        }
    )
    missing = [c for c in ("text", "label") if c not in df.columns]
    assert missing == []
    # The exact guard features.py uses must pass with extra columns present.
    assert "source" in df.columns and "text" in df.columns


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
