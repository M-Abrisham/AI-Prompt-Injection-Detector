#!/usr/bin/env python3
"""Harvester ingestion bridge: dataset reference -> decontam -> quarantine.

This is DATA-PIPELINE PLUMBING. It moves opaque candidate rows from a
dataset reference through the EXISTING quarantine gate. It does not generate,
analyze, or interpret attack content — every row is treated as an opaque
``{text, label}`` pair.

Flow
----
1. **Resolve + download** a dataset reference via the existing mechanism:
   - ``hf:<repo>`` / a bare ``owner/name`` HuggingFace repo id  -> reuse
     ``scripts.sync_datasets._download_huggingface`` (huggingface_hub /
     ``datasets``). Fails gracefully offline.
   - a raw GitHub / http(s) CSV URL                              -> reuse
     ``scripts.sync_datasets._download_github_csv``.
   - a local ``.csv`` / ``.jsonl`` path                          -> read in place.
2. **Normalize** every row to the canonical schema via
   :class:`na0s.dataset.schema.Na0SSample` (``text``, ``label``, ``source``,
   ``source_id``, optional ``attack_category`` -> ``technique_id``, ``license``).
3. **Eval-decontaminate** (MANDATORY): drop any row whose normalized text
   matches anything in ``data/holdout/*`` / ``data/benchmark/*`` /
   ``data/eval/scenarios/*`` — via
   :func:`na0s.eval.harvest.decontam.build_eval_decontam_set`, the same
   stable_id contract pinned by ``tests/test_no_holdout_leakage.py``.
4. **Route** accepted rows through ``scripts.quarantine.ingest`` — NEVER
   directly into ``data/raw`` or training. New/unknown sources resolve to
   tier3 -> ``data/quarantine/`` and require explicit validation + promotion
   to reach ``data/staging`` and eventually training.

Discovered datasets from ``data/harvest/new_datasets.jsonl`` (written by
``scripts/weekly_harvest.py``) can be fetched + ingested with
``--from-harvest`` — only ``source == "huggingface"`` records carry a
directly-fetchable repo id (the ``id`` field); arXiv / GitHub discovery
records carry no rows and are skipped.

Usage::

    python scripts/fetch_and_ingest.py --ref imoxto/prompt_injection_cleaned_dataset-v2
    python scripts/fetch_and_ingest.py --ref https://raw.githubusercontent.com/.../data.csv \
        --text-column prompt --label 1
    python scripts/fetch_and_ingest.py --ref ./local_fixture.jsonl
    python scripts/fetch_and_ingest.py --from-harvest --limit 5
    python scripts/fetch_and_ingest.py --ref owner/name --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from na0s.dataset.schema import DataLabel, Na0SSample  # noqa: E402
from na0s.eval.harvest.decontam import build_eval_decontam_set  # noqa: E402

from scripts import quarantine  # noqa: E402

HARVEST_JSONL = os.path.join(_ROOT, "data", "harvest", "new_datasets.jsonl")
# Per-source staged JSONL files land here BEFORE quarantine ingest, mirroring
# integrate_harvest.DEFAULT_STAGING_DIR convention.
DEFAULT_STAGE_INPUT_DIR = os.path.join(_ROOT, "data", "staging", "fetch_and_ingest")


# ── Reference resolution + download ─────────────────────────────────────────


def classify_ref(ref: str) -> str:
    """Classify a dataset reference into 'local' | 'github_csv' | 'huggingface'."""
    if ref.startswith("hf:"):
        return "huggingface"
    if os.path.isfile(ref):
        return "local"
    if re.match(r"^https?://", ref):
        return "github_csv"
    # Bare ``owner/name`` -> HuggingFace repo id.
    if re.match(r"^[\w.-]+/[\w.-]+$", ref):
        return "huggingface"
    return "local"  # let the loader raise a clear file-not-found error


def download_to_temp(
    ref: str,
    kind: str,
    text_column: str,
    label: Optional[str],
    label_column: Optional[str],
    label_map: Optional[dict],
    max_samples: Optional[int],
    workdir: str,
) -> str:
    """Download ``ref`` to a normalized CSV under ``workdir`` using the
    EXISTING sync_datasets download helpers. Returns the local file path.

    Raises ``RuntimeError`` on graceful failure (e.g. offline HF).
    """
    from scripts import sync_datasets

    if kind == "local":
        if not os.path.isfile(ref):
            raise RuntimeError(f"local file not found: {ref}")
        return ref

    out_path = os.path.join(workdir, "downloaded.csv")

    if kind == "github_csv":
        cfg = {"url": ref, "text_column": text_column, "label": label or "1"}
        try:
            sync_datasets._download_github_csv(cfg, out_path)
        except Exception as e:  # network / parse failure -> graceful
            raise RuntimeError(f"github_csv download failed for {ref}: {e}") from e
        return out_path

    if kind == "huggingface":
        if not getattr(sync_datasets, "HF_AVAILABLE", False):
            raise RuntimeError(
                "huggingface_hub / datasets not installed — cannot fetch "
                f"HF repo {ref} (offline-safe skip)"
            )
        repo = ref[3:] if ref.startswith("hf:") else ref
        cfg = {
            "repo": repo,
            "text_column": text_column,
            "max_samples": max_samples,
        }
        if label_column and label_map:
            cfg["label_column"] = label_column
            cfg["label_map"] = label_map
        else:
            cfg["label"] = label if label is not None else "1"
        try:
            sync_datasets._download_huggingface(cfg, out_path)
        except Exception as e:  # network / repo failure -> graceful
            raise RuntimeError(f"huggingface download failed for {repo}: {e}") from e
        if not os.path.isfile(out_path):
            raise RuntimeError(f"huggingface fetch produced no rows for {repo}")
        return out_path

    raise RuntimeError(f"unknown reference kind: {kind}")


# ── Row normalization ───────────────────────────────────────────────────────


def _iter_raw_rows(path: str, text_column: str, label_default: Optional[str]):
    """Yield raw ``{text, label}`` dicts from a CSV or JSONL file.

    Payloads are OPAQUE — no inspection of content, only column extraction.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        import csv as _csv

        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = _csv.DictReader(fh)
            cols = reader.fieldnames or []
            # Resolve the text column case-insensitively; the synced CSV uses
            # "text", but a raw local fixture may differ.
            tcol = next((c for c in cols if c.lower() == text_column.lower()), None)
            tcol = tcol or next((c for c in cols if c.lower() == "text"), None)
            lcol = next((c for c in cols if c.lower() == "label"), None)
            for row in reader:
                text = (row.get(tcol) if tcol else "") or ""
                lbl = (row.get(lcol) if lcol else None)
                if lbl is None:
                    lbl = label_default
                yield {"text": text, "label": lbl}
    elif ext in (".jsonl", ".ndjson"):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                text = obj.get(text_column) or obj.get("text") or obj.get("prompt") or ""
                lbl = obj.get("label", label_default)
                yield {"text": str(text), "label": lbl}
    else:
        raise RuntimeError(f"unsupported file type for normalization: {path}")


def normalize_rows(
    raw_rows,
    source: str,
    source_id: str,
    attack_category: Optional[str] = None,
    license_str: Optional[str] = None,
):
    """Normalize raw rows to canonical Na0SSample dicts.

    Rows with empty text or an unrecognized label are skipped (counted as
    ``dropped_invalid``). ``attack_category`` is attached as provenance only
    when it matches the canonical taxonomy; otherwise it is dropped (never
    invented) and the row is still ingested as an opaque sample.
    """
    technique_id = _validated_technique(attack_category)

    samples, dropped_invalid = [], 0
    for raw in raw_rows:
        text = (raw.get("text") or "").strip()
        if not text:
            dropped_invalid += 1
            continue
        raw_label = raw.get("label")
        try:
            sample = Na0SSample.from_legacy_csv_row(
                {"text": text, "label": raw_label}
            )
        except ValueError:
            dropped_invalid += 1
            continue
        sample.source = source
        sample.source_id = source_id
        sample.license = license_str
        if technique_id:
            sample.technique_id = technique_id
        samples.append(sample)
    return samples, dropped_invalid


def _validated_technique(attack_category: Optional[str]) -> Optional[str]:
    """Return ``attack_category`` only if it is a canonical taxonomy code."""
    if not attack_category:
        return None
    try:
        from na0s.eval.harvest.taxonomy import TaxonomyValidator

        if TaxonomyValidator().validate_code(attack_category):
            return attack_category
    except Exception:
        pass
    return None


# ── Staging + quarantine routing ────────────────────────────────────────────


def _sanitize(source_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", source_id).strip("_")
    return safe or "unknown"


def write_stage_jsonl(samples, source_id: str, stage_dir: str) -> str:
    """Write normalized samples to a per-source JSONL staging-input file.

    Rows are written in the canonical ``{text, label, ...}`` schema (full
    Na0SSample dict) so downstream label-quality checks have provenance.
    """
    os.makedirs(stage_dir, exist_ok=True)
    import hashlib

    h = hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:8]
    path = os.path.join(stage_dir, f"{_sanitize(source_id)}_{h}.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for s in samples:
            d = s.to_dict()
            # Quarantine + training read integer/binary labels; keep both.
            d["label"] = 1 if s.label == DataLabel.INJECTION else 0
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    return path


def fetch_and_ingest(
    ref: str,
    *,
    source_id: Optional[str] = None,
    text_column: str = "text",
    label: Optional[str] = None,
    label_column: Optional[str] = None,
    label_map: Optional[dict] = None,
    attack_category: Optional[str] = None,
    license_str: Optional[str] = None,
    max_samples: Optional[int] = None,
    stage_dir: str = DEFAULT_STAGE_INPUT_DIR,
    decontaminator=None,
    quarantine_config=None,
    dry_run: bool = False,
) -> dict:
    """End-to-end: fetch ``ref`` -> normalize -> decontam -> quarantine ingest.

    Returns a result dict with counts and the quarantine routing outcome.
    """
    kind = classify_ref(ref)
    # source_id drives trust-tier resolution; default to the HF repo id for HF
    # refs (so trust_tiers.yaml exact/prefix matches work), else a harvest tag.
    if source_id is None:
        if kind == "huggingface":
            source_id = ref[3:] if ref.startswith("hf:") else ref
        else:
            source_id = f"fetch/{_sanitize(os.path.basename(ref))}"

    result = {
        "ref": ref,
        "kind": kind,
        "source_id": source_id,
        "fetched": 0,
        "normalized": 0,
        "dropped_invalid": 0,
        "dropped_contaminated": 0,
        "accepted": 0,
        "action": None,
        "destination": None,
        "error": None,
    }

    if decontaminator is None:
        decontaminator = build_eval_decontam_set()

    with tempfile.TemporaryDirectory(prefix="na0s_fetch_") as workdir:
        try:
            local_path = download_to_temp(
                ref, kind, text_column, label, label_column, label_map,
                max_samples, workdir,
            )
        except RuntimeError as e:
            result["error"] = str(e)
            print(f"  SKIP {ref}: {e}", file=sys.stderr)
            return result

        raw_rows = list(_iter_raw_rows(local_path, text_column, label))
        result["fetched"] = len(raw_rows)

        samples, dropped_invalid = normalize_rows(
            raw_rows, source=kind, source_id=source_id,
            attack_category=attack_category, license_str=license_str,
        )
        result["normalized"] = len(samples)
        result["dropped_invalid"] = dropped_invalid

        # MANDATORY eval decontamination.
        accepted, dropped = decontaminator.filter_rows(
            samples, text_getter=lambda s: s.text
        )
        result["dropped_contaminated"] = len(dropped)
        result["accepted"] = len(accepted)

        if not accepted:
            result["action"] = "no_rows"
            print(f"  {ref}: no rows survived normalization/decontam.")
            return result

        stage_path = write_stage_jsonl(accepted, source_id, stage_dir)
        result["stage_path"] = stage_path

        if dry_run:
            result["action"] = "dry_run"
            print(
                f"  [DRY RUN] {ref}: {len(accepted)} row(s) staged to "
                f"{stage_path} (not ingested)."
            )
            return result

        cfg = quarantine_config or quarantine.load_trust_config()
        ingest_result = quarantine.ingest(stage_path, source_id, config=cfg)
        result["action"] = ingest_result.get("action")
        result["destination"] = ingest_result.get("destination")
        result["tier"] = ingest_result.get("tier")
        result["trust_gate"] = ingest_result.get("trust_gate")

    return result


# ── Harvest discovery call site ──────────────────────────────────────────────


def iter_fetchable_harvest_records(harvest_jsonl: str = HARVEST_JSONL):
    """Yield (repo_id, record) for HF discovery records that carry rows.

    arXiv / GitHub discovery records have no directly-fetchable labeled
    dataset and are skipped (the bridge feeds the TRAINING path).
    """
    if not os.path.isfile(harvest_jsonl):
        return
    with open(harvest_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("source") != "huggingface":
                continue
            repo_id = rec.get("id")
            if repo_id and re.match(r"^[\w.-]+/[\w.-]+$", repo_id):
                yield repo_id, rec


def run_from_harvest(
    harvest_jsonl: str,
    limit: Optional[int],
    stage_dir: str,
    dry_run: bool,
) -> int:
    decontaminator = build_eval_decontam_set()
    cfg = None if dry_run else quarantine.load_trust_config()
    count = 0
    results = []
    for repo_id, rec in iter_fetchable_harvest_records(harvest_jsonl):
        if limit is not None and count >= limit:
            break
        count += 1
        # Pull a license tag (e.g. "license:cc-by-nc-4.0") for provenance.
        license_str = None
        for tag in rec.get("tags", []) or []:
            if isinstance(tag, str) and tag.startswith("license:"):
                license_str = tag.split(":", 1)[1]
                break
        print(f"\n[harvest {count}] fetch+ingest HF repo: {repo_id}")
        res = fetch_and_ingest(
            repo_id,
            source_id=repo_id,
            license_str=license_str,
            stage_dir=stage_dir,
            decontaminator=decontaminator,
            quarantine_config=cfg,
            dry_run=dry_run,
        )
        results.append(res)

    print("\n" + "=" * 65)
    print("Harvest fetch-and-ingest summary")
    print("=" * 65)
    print(f"  HF records attempted:   {count}")
    print(f"  Ingested (quarantined): "
          f"{sum(1 for r in results if r['action'] == 'quarantined')}")
    print(f"  Direct-pass:            "
          f"{sum(1 for r in results if r['action'] == 'direct_pass')}")
    print(f"  Skipped (fetch error):  "
          f"{sum(1 for r in results if r['error'])}")
    print(f"  Rows dropped (contam):  "
          f"{sum(r['dropped_contaminated'] for r in results)}")
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Fetch a dataset reference, eval-decontaminate, and route "
        "rows through the quarantine gate (never directly into training).",
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ref", help="HF repo id, raw CSV URL, or local .csv/.jsonl path")
    src.add_argument(
        "--from-harvest", action="store_true",
        help="Fetch+ingest fetchable HF records from data/harvest/new_datasets.jsonl",
    )

    p.add_argument("--source-id", default=None,
                   help="Trust-tier source id (default: derived from --ref)")
    p.add_argument("--text-column", default="text")
    p.add_argument("--label", default=None,
                   help="Fixed label for all rows (e.g. 1 or injection)")
    p.add_argument("--label-column", default=None)
    p.add_argument("--attack-category", default=None,
                   help="Canonical taxonomy code for provenance (validated)")
    p.add_argument("--license", dest="license_str", default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--harvest-jsonl", default=HARVEST_JSONL)
    p.add_argument("--limit", type=int, default=None,
                   help="Max records to process from --from-harvest")
    p.add_argument("--stage-dir", default=DEFAULT_STAGE_INPUT_DIR)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    print("Na0S harvester ingestion bridge")
    print(f"  Stage dir: {args.stage_dir}")
    print(f"  Dry run:   {args.dry_run}")

    if args.from_harvest:
        return run_from_harvest(
            args.harvest_jsonl, args.limit, args.stage_dir, args.dry_run
        )

    res = fetch_and_ingest(
        args.ref,
        source_id=args.source_id,
        text_column=args.text_column,
        label=args.label,
        label_column=args.label_column,
        attack_category=args.attack_category,
        license_str=args.license_str,
        max_samples=args.max_samples,
        stage_dir=args.stage_dir,
        dry_run=args.dry_run,
    )
    print("\n" + "=" * 65)
    print("Fetch-and-ingest result")
    print("=" * 65)
    for k in ("ref", "kind", "source_id", "fetched", "normalized",
              "dropped_invalid", "dropped_contaminated", "accepted",
              "action", "destination", "tier", "trust_gate", "error"):
        if k in res:
            print(f"  {k}: {res[k]}")
    if res.get("error"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
