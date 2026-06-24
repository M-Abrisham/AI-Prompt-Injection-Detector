#!/usr/bin/env python3
"""Eval decontamination check — block F14 scenarios from leaking into training data.

Walks the F14 scenario library under ``data/eval/scenarios/v0.1/`` and collects
the ``stable_id`` (SHA-256 of NFKC-normalized text) of every scenario. Then
walks the training-data roots (``data/processed/``, ``data/staging/``,
``data/aggregated/``) and computes the stable_id of each row's ``text`` column.
Any intersection is a contamination event — the candidate model was trained on
eval data and the promotion gate would give artificially-inflated metrics.

The exact-hash leg is the always-on hard block; it only catches verbatim
copies. Three OPT-IN legs extend it, each catching a leak class the exact
hash misses. All three are OFF by default so the always-on CI gate stays
fast and dependency-light, and all three degrade gracefully on an empty
corpus / missing optional deps:

  * ``--near-dup`` — MinHash + LSH token-Jaccard (reuses
    ``na0s.dataset.near_duplicate``). Catches paraphrase / light-edit leaks.
  * ``--bff`` — 13-gram (word) presence-fraction. Catches *partial / spliced*
    copies where a long contiguous span of an eval scenario is embedded inside
    a larger training row (or vice-versa). The exact hash misses this (the
    whole-string hash differs) and MinHash-on-the-full-row can dilute a small
    shared span below the Jaccard cutoff. ``n=13`` is the de-facto dedup
    convention (GPT-3 / C4 / Allen-AI "BFF" all use contiguous 13-token spans
    as the partial-copy unit); overridable via ``--bff-n``. The fraction
    threshold is conservative by default (see ``--bff-threshold``).
  * ``--embedding`` — local, KEYLESS embedding-cosine. Catches semantic
    paraphrase that survives both exact-hash and token-Jaccard (synonym swaps,
    re-ordering). Reuses the same pinned offline model the admission gate's
    ``embedding_fn`` hook expects (``all-MiniLM-L6-v2`` via
    ``na0s.ml._st_loader.load_pinned_sentence_transformer``). If
    ``sentence-transformers`` is absent the leg prints a skip notice and is a
    no-op — it never fails the gate. Threshold inherits
    ``admission_gate.DEFAULT_NEAR_DUP_THRESHOLD`` (0.85); not a new constant.

The report also always prints a per-source overlap table so an operator can
see WHICH training dataset is the contamination vector, not just that
contamination exists.

The opt-in legs are warning-only unless ``--strict``; exact overlap is always
fatal. A run that scans zero training rows fails loud (exit 2) rather than
reporting a false "clean" — an empty/unbuilt corpus cannot certify
decontamination, regardless of which legs are enabled.

Usage:
    python scripts/check_eval_decontamination.py
    python scripts/check_eval_decontamination.py --training-roots data/processed/ data/staging/
    python scripts/check_eval_decontamination.py --near-dup            # warn on paraphrases
    python scripts/check_eval_decontamination.py --bff                 # warn on partial spans
    python scripts/check_eval_decontamination.py --embedding           # warn on semantic paraphrase
    python scripts/check_eval_decontamination.py --strict              # opt-in legs fatal too

Exit codes:
    0 — no exact overlap (near-dup/BFF/embedding matches, if any, were warning-only)
    1 — contamination: exact overlap, or near-dup/BFF/embedding overlap under --strict
    2 — configuration / IO error, or empty corpus (no --allow-empty-corpus)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from na0s.eval.scenarios.loader import load_scenarios_dir  # noqa: E402
from na0s.eval.scenarios.schema import ScenarioType  # noqa: E402
from na0s.dataset.near_duplicate import (  # noqa: E402
    LSH_BANDS,
    LSH_ROWS_PER_BAND,
    MINHASH_JACCARD_THRESHOLD,
    jaccard_from_minhash,
    lsh_buckets,
    minhash_signature,
)
# Reuse — do NOT reinvent — the admission gate's cosine helper and its
# already-shipped 0.85 near-dup cutoff so the embedding leg here and the
# F14 admission gate stay in lock-step.
from na0s.eval.scenarios.admission_gate import (  # noqa: E402
    DEFAULT_NEAR_DUP_THRESHOLD,
    _cosine,
)

# The pinned local embedding model (mirrors na0s.ml.predict_embedding.
# DEFAULT_EMBEDDING_MODEL). Defined here as a plain literal — NOT imported —
# because predict_embedding raises ImportError at import time when
# sentence-transformers is absent (the local reality), and the always-on gate
# must stay importable / dependency-light. The embedding leg loads the model
# lazily only when --embedding is passed AND sentence-transformers is present.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# 13-gram is the de-facto partial-copy dedup unit (GPT-3 / C4 / Allen-AI BFF).
# Documented convention, NOT a tuned magic number; overridable via --bff-n.
_DEFAULT_BFF_N = 13
# CONSERVATIVE default: the local training corpus (data/processed/*.csv) is
# gitignored and EMPTY here, so --bff-threshold cannot be calibrated locally.
# 0.90 flags only near-total span copies (>=90% of an eval scenario's 13-grams
# present in training) — high-precision / low-FP without a calibration corpus.
# Precise calibration is deferred to a CI run with a real corpus (sweep the
# F14 v0.1 scenarios vs a known-clean training sample). See module tests.
_DEFAULT_BFF_THRESHOLD = 0.90
# Mask used to bound the training 13-gram set's memory (store hash(gram)&mask
# as a 64-bit int rather than the gram string). Collision rate is negligible.
_BFF_HASH_MASK = (1 << 64) - 1


_DEFAULT_SCENARIOS = _PROJECT_ROOT / "data" / "eval" / "scenarios" / "v0.1"
_DEFAULT_TRAINING_ROOTS = [
    _PROJECT_ROOT / "data" / "processed",
    _PROJECT_ROOT / "data" / "staging",
    _PROJECT_ROOT / "data" / "aggregated",
]


def compute_stable_id(text: str) -> str:
    """Return SHA-256 of NFKC + whitespace-collapsed text.

    Mirrors ``Na0SSample.__post_init__`` and ``Scenario._compute_stable_id``
    so training-row hashes compare equal to scenario hashes. Kept as a
    free function here to avoid a circular import of Na0SSample.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = " ".join(normalized.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _collect_scenario_ids(scenarios_dir: Path) -> dict[str, tuple[str, Path]]:
    """Return ``{stable_id: (scenario_name, source_yaml_path)}``.

    Multi-turn scenarios contribute the stable_id of each turn's text in
    addition to the scenario-level id, so training rows that happen to
    match a single turn still get caught.
    """
    ids: dict[str, tuple[str, Path]] = {}
    for scn in load_scenarios_dir(scenarios_dir):
        source = Path(getattr(scn, "_source_file", scenarios_dir))
        if scn.type == ScenarioType.SINGLE_PROMPT and scn.payload:
            ids[compute_stable_id(scn.payload)] = (scn.name, source)
        elif scn.type == ScenarioType.MULTI_TURN:
            for idx, turn in enumerate(scn.turns):
                ids[compute_stable_id(turn.text)] = (
                    f"{scn.name}[turn={idx}]", source,
                )
        # stable_id-populated form (auto-computed in __post_init__)
        if scn.stable_id:
            ids.setdefault(scn.stable_id, (scn.name, source))
    return ids


def _iter_training_texts(roots: list[Path]):
    """Yield ``(text, source_path, row_num)`` for every text-bearing training row.

    Accepts CSV (expects a ``text`` column; case-insensitive) and JSONL
    (expects a ``text`` field). Silently skips non-matching files. Roots
    that don't exist are skipped — a training stage not yet populated is
    not a decontamination error.
    """
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix == ".csv":
                yield from _iter_csv_texts(path)
            elif suffix in (".jsonl", ".ndjson"):
                yield from _iter_jsonl_texts(path)


def _iter_csv_texts(path: Path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return
        # Find the text column (case-insensitive).
        text_col = next(
            (c for c in reader.fieldnames if c.lower() == "text"),
            None,
        )
        if text_col is None:
            return
        for i, row in enumerate(reader, start=2):  # header is line 1
            text = row.get(text_col) or ""
            if text:
                yield text, path, i


def _iter_jsonl_texts(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text") if isinstance(obj, dict) else None
            if text:
                yield text, path, i


def _collect_scenario_texts(scenarios_dir: Path) -> list[tuple[str, str, Path]]:
    """Return ``[(text, scenario_name, source_yaml_path), ...]``.

    The text-level granularity mirrors ``_collect_scenario_ids`` (one entry
    per single-prompt payload and per multi-turn turn) so the near-dup leg
    catches a training row that paraphrases any individual turn.
    """
    rows: list[tuple[str, str, Path]] = []
    for scn in load_scenarios_dir(scenarios_dir):
        source = Path(getattr(scn, "_source_file", scenarios_dir))
        if scn.type == ScenarioType.SINGLE_PROMPT and scn.payload:
            rows.append((scn.payload, scn.name, source))
        elif scn.type == ScenarioType.MULTI_TURN:
            for idx, turn in enumerate(scn.turns):
                rows.append((turn.text, f"{scn.name}[turn={idx}]", source))
    return rows


def scan_exact(
    scenarios_dir: Path,
    training_roots: list[Path],
) -> tuple[list[dict], int, int, dict[str, dict[str, int]]]:
    """Run the exact stable_id overlap scan.

    Returns ``(overlaps, n_training_rows_scanned, n_scenario_ids, per_source)``.
    The row count lets the CLI distinguish "scanned a real corpus and found it
    clean" from "scanned nothing" (an empty/unbuilt corpus must not pass
    silently).

    ``per_source`` is ``{str(training_file): {"rows": int, "overlaps": int}}``
    — a per-dataset breakdown of how many rows each training file contributed
    and how many of those were exact contamination hits, so an operator can see
    WHICH dataset is the contamination vector, not just that contamination
    exists. Pure reporting; it never changes exit-code logic.
    """
    scenario_ids = _collect_scenario_ids(scenarios_dir)
    overlaps: list[dict] = []
    per_source: dict[str, dict[str, int]] = {}
    n_rows = 0
    if not scenario_ids:
        return overlaps, n_rows, 0, per_source
    for text, source, row in _iter_training_texts(training_roots):
        n_rows += 1
        bucket = per_source.setdefault(str(source), {"rows": 0, "overlaps": 0})
        bucket["rows"] += 1
        sid = compute_stable_id(text)
        hit = scenario_ids.get(sid)
        if hit is not None:
            scn_name, scn_source = hit
            bucket["overlaps"] += 1
            overlaps.append({
                "stable_id": sid,
                "scenario_name": scn_name,
                "scenario_source": str(scn_source),
                "training_file": str(source),
                "training_row": row,
            })
    return overlaps, n_rows, len(scenario_ids), per_source


def find_overlaps(
    scenarios_dir: Path,
    training_roots: list[Path],
) -> list[dict]:
    """Return list of exact-overlap records (empty if no contamination).

    Thin back-compat wrapper around :func:`scan_exact`.
    """
    overlaps, _, _, _ = scan_exact(scenarios_dir, training_roots)
    return overlaps


def find_near_dup_overlaps(
    scenarios_dir: Path,
    training_roots: list[Path],
    threshold: float = MINHASH_JACCARD_THRESHOLD,
) -> list[dict]:
    """Return near-duplicate (paraphrase / light-edit) overlap records.

    Exact stable_id collisions are excluded here — they are already reported
    by :func:`scan_exact`; this leg surfaces the *additional* leaks that an
    exact hash misses (whitespace/casing/token-level edits, partial copies).

    Uses MinHash + LSH (``na0s.dataset.near_duplicate``) so cost is roughly
    linear in the training corpus rather than ``scenarios x rows``. The
    default ``threshold`` is the repo-canonical ``MINHASH_JACCARD_THRESHOLD``
    (0.8) — not a new arbitrary constant.
    """
    scenario_rows = _collect_scenario_texts(scenarios_dir)
    if not scenario_rows:
        return []
    # Pre-compute scenario signatures + their exact ids (to skip exact hits).
    scenario_sigs: list[list[int]] = []
    scenario_exact_ids: set[str] = set()
    bucket_to_scn: dict[int, list[int]] = {}
    for i, (text, _name, _src) in enumerate(scenario_rows):
        sig = minhash_signature(text)
        scenario_sigs.append(sig)
        scenario_exact_ids.add(compute_stable_id(text))
        for key in lsh_buckets(sig, LSH_BANDS, LSH_ROWS_PER_BAND):
            bucket_to_scn.setdefault(key, []).append(i)

    overlaps: list[dict] = []
    seen: set[tuple[int, str, int]] = set()
    for text, source, row in _iter_training_texts(training_roots):
        if compute_stable_id(text) in scenario_exact_ids:
            continue  # exact dup — already reported by scan_exact
        sig = minhash_signature(text)
        # LSH candidate set: scenarios sharing at least one band bucket.
        candidates: set[int] = set()
        for key in lsh_buckets(sig, LSH_BANDS, LSH_ROWS_PER_BAND):
            candidates.update(bucket_to_scn.get(key, ()))
        for i in candidates:
            sim = jaccard_from_minhash(sig, scenario_sigs[i])
            if sim >= threshold:
                dedup_key = (i, str(source), row)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                scn_text, scn_name, scn_source = scenario_rows[i]
                overlaps.append({
                    "scenario_name": scn_name,
                    "scenario_source": str(scn_source),
                    "training_file": str(source),
                    "training_row": row,
                    "jaccard": round(sim, 4),
                })
    return overlaps


def _word_ngrams(text: str, n: int) -> list[int]:
    """Return the list of word ``n``-gram hashes for ``text``.

    Normalization matches :func:`compute_stable_id` (NFKC + whitespace-collapse)
    so the BFF leg's notion of "same span" is consistent with the exact leg's
    canonicalization. Each n-gram is reduced to ``hash(gram) & _BFF_HASH_MASK``
    (a 64-bit int) so a large training corpus's gram set stays bounded in
    memory; the collision rate at 64 bits is negligible.

    A text with fewer than ``n`` tokens has ZERO n-grams and returns ``[]`` —
    callers must treat that as "no signal from this leg" (such scenarios stay
    covered by the exact + MinHash legs), never as 0.0 or 1.0 overlap.
    """
    normalized = unicodedata.normalize("NFKC", text)
    tokens = normalized.split()
    if len(tokens) < n:
        return []
    grams = []
    for i in range(len(tokens) - n + 1):
        gram = " ".join(tokens[i:i + n])
        grams.append(hash(gram) & _BFF_HASH_MASK)
    return grams


def find_bff_overlaps(
    scenarios_dir: Path,
    training_roots: list[Path],
    *,
    n: int = _DEFAULT_BFF_N,
    min_presence_fraction: float = _DEFAULT_BFF_THRESHOLD,
) -> tuple[list[dict], int]:
    """Return 13-gram presence-fraction overlap records + a skipped-short count.

    For each eval scenario text, presence-fraction =
    ``|scenario n-grams that also appear in the training n-gram set| /
    |scenario n-grams|``. A scenario is flagged when that fraction is
    ``>= min_presence_fraction`` — i.e. a large contiguous span of the scenario
    is present somewhere in training even if no single training row is a
    whole-string or high-Jaccard match.

    Exact stable_id collisions are excluded (already reported by
    :func:`scan_exact`). Scenarios with fewer than ``n`` tokens have no
    n-grams; they are SKIPPED in this leg and counted in the returned
    ``skipped_short`` integer (they remain covered by exact + MinHash).

    Returns ``(overlaps, skipped_short)``. ``overlaps`` is empty when the
    corpus is empty or no scenario clears the threshold.
    """
    scenario_rows = _collect_scenario_texts(scenarios_dir)
    if not scenario_rows:
        return [], 0

    # Build the training n-gram set in a single streaming pass over the corpus.
    training_grams: set[int] = set()
    scenario_exact_ids: set[str] = {
        compute_stable_id(text) for text, _name, _src in scenario_rows
    }
    has_training_rows = False
    for text, _source, _row in _iter_training_texts(training_roots):
        has_training_rows = True
        if compute_stable_id(text) in scenario_exact_ids:
            continue  # exact dup — reported by scan_exact, not this leg
        training_grams.update(_word_ngrams(text, n))

    overlaps: list[dict] = []
    skipped_short = 0
    if not has_training_rows or not training_grams:
        # Empty corpus (or only exact dups / sub-n-gram rows): no BFF signal.
        # Still count short scenarios so the report is honest.
        for text, _name, _src in scenario_rows:
            if len(unicodedata.normalize("NFKC", text).split()) < n:
                skipped_short += 1
        return overlaps, skipped_short

    for text, name, src in scenario_rows:
        grams = _word_ngrams(text, n)
        if not grams:
            skipped_short += 1
            continue
        present = sum(1 for g in grams if g in training_grams)
        fraction = present / len(grams)
        if fraction >= min_presence_fraction:
            overlaps.append({
                "scenario_name": name,
                "scenario_source": str(src),
                "presence_fraction": round(fraction, 4),
                "n_grams": len(grams),
                "n": n,
            })
    return overlaps, skipped_short


def _load_default_embedding_fn():
    """Return a keyless local ``embedding_fn`` or ``None`` if unavailable.

    Lazily constructs the pinned offline ``all-MiniLM-L6-v2`` model via
    ``na0s.ml._st_loader.load_pinned_sentence_transformer`` (the SAME model the
    F14 admission gate's ``embedding_fn`` hook expects). Returns ``None`` —
    never raises — when ``sentence-transformers`` is absent or the model cannot
    be constructed, so the embedding leg degrades to a no-op skip rather than
    failing the gate. No raw API key is ever required.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        from na0s.ml._st_loader import load_pinned_sentence_transformer

        model = load_pinned_sentence_transformer(
            SentenceTransformer, DEFAULT_EMBEDDING_MODEL,
        )
    except Exception:  # pragma: no cover - model download / construction issues
        return None

    def _embed(text: str) -> list[float]:
        vec = model.encode([text])[0]
        return [float(x) for x in vec]

    return _embed


def find_embedding_overlaps(
    scenarios_dir: Path,
    training_roots: list[Path],
    *,
    embedding_fn=None,
    threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
) -> list[dict]:
    """Return semantic embedding-cosine overlap records.

    Embeds every eval scenario text and every (non-exact-dup) training row via
    ``embedding_fn`` and flags a scenario when its max cosine over the corpus is
    ``>= threshold``. Mirrors ``admission_gate._check_near_dup_decontam``'s
    cosine leg and reuses its ``_cosine`` helper and 0.85 cutoff.

    ``embedding_fn`` defaults to :func:`_load_default_embedding_fn`. If that is
    ``None`` (``sentence-transformers`` absent) — or if it raises — this leg
    returns ``[]`` and never fails the gate (the caller prints a skip notice).
    Exact stable_id collisions are excluded (already reported by
    :func:`scan_exact`).

    NOTE: 0.85 was set for the admission gate's near-dup proxy; it has NOT been
    re-validated for short injection strings, so this leg is warning-only and
    must not be made fatal without re-calibration.
    """
    if embedding_fn is None:
        embedding_fn = _load_default_embedding_fn()
    if embedding_fn is None:
        return []

    scenario_rows = _collect_scenario_texts(scenarios_dir)
    if not scenario_rows:
        return []

    try:
        scenario_vecs = [embedding_fn(text) for text, _n, _s in scenario_rows]
    except Exception:  # pragma: no cover - depends on user fn
        return []
    scenario_exact_ids = {
        compute_stable_id(text) for text, _n, _s in scenario_rows
    }

    overlaps: list[dict] = []
    best: dict[int, dict] = {}
    for text, source, row in _iter_training_texts(training_roots):
        if compute_stable_id(text) in scenario_exact_ids:
            continue  # exact dup — already reported by scan_exact
        try:
            row_vec = embedding_fn(text)
        except Exception:  # pragma: no cover - depends on user fn
            return []
        for i, scn_vec in enumerate(scenario_vecs):
            cos = _cosine(scn_vec, row_vec)
            if cos >= threshold and cos > best.get(i, {}).get("cosine", -1.0):
                scn_text, scn_name, scn_source = scenario_rows[i]
                best[i] = {
                    "scenario_name": scn_name,
                    "scenario_source": str(scn_source),
                    "training_file": str(source),
                    "training_row": row,
                    "cosine": round(cos, 4),
                }
    for i in sorted(best):
        overlaps.append(best[i])
    return overlaps


def main() -> int:
    args = _parse_args()

    scenarios_dir = Path(args.scenarios_dir)
    if not scenarios_dir.exists():
        print(f"ERROR: scenarios dir not found: {scenarios_dir}", file=sys.stderr)
        return 2

    training_roots = [Path(p) for p in (args.training_roots or _DEFAULT_TRAINING_ROOTS)]
    near_dup_enabled = args.near_dup or args.strict
    bff_enabled = args.bff or args.strict
    embedding_enabled = args.embedding or args.strict

    exact_overlaps, n_rows, n_ids, per_source = scan_exact(
        scenarios_dir, training_roots,
    )

    print("=" * 70)
    print("  Na0S F14 Eval Decontamination Check")
    print("=" * 70)
    print(f"  Scenarios dir:    {scenarios_dir}")
    print(f"  Training roots:   {[str(r) for r in training_roots]}")
    print(f"  Scenario ids:     {n_ids}")
    print(f"  Training rows:    {n_rows}")
    print(f"  Near-dup leg:     {'on (threshold=%s)' % args.near_dup_threshold if near_dup_enabled else 'off'}")
    print(f"  BFF leg:          {'on (n=%s, fraction>=%s)' % (args.bff_n, args.bff_threshold) if bff_enabled else 'off'}")
    print(f"  Embedding leg:    {'on (threshold=%s)' % args.embedding_threshold if embedding_enabled else 'off'}")
    print()

    # Empty/unbuilt corpus must fail loud — silently passing here would
    # certify "no contamination" against zero rows (false green). This guard
    # holds regardless of which opt-in legs are enabled.
    if n_ids > 0 and n_rows == 0 and not args.allow_empty_corpus:
        print(
            "  ERROR: scanned 0 training rows — corpus is empty or unbuilt; "
            "cannot certify decontamination.\n"
            "  (run scripts/process_data.py first, or pass --allow-empty-corpus)",
            file=sys.stderr,
        )
        print("=" * 70)
        return 2

    # Opt-in legs run only after the empty-corpus guard so they never mask it.
    near_overlaps = (
        find_near_dup_overlaps(scenarios_dir, training_roots, args.near_dup_threshold)
        if near_dup_enabled else []
    )
    bff_overlaps: list[dict] = []
    bff_skipped_short = 0
    if bff_enabled:
        bff_overlaps, bff_skipped_short = find_bff_overlaps(
            scenarios_dir, training_roots,
            n=args.bff_n, min_presence_fraction=args.bff_threshold,
        )
    embedding_overlaps: list[dict] = []
    embedding_skipped = False
    if embedding_enabled:
        embedding_fn = _load_default_embedding_fn()
        if embedding_fn is None:
            embedding_skipped = True
        else:
            embedding_overlaps = find_embedding_overlaps(
                scenarios_dir, training_roots,
                embedding_fn=embedding_fn, threshold=args.embedding_threshold,
            )

    if exact_overlaps:
        print(f"  CONTAMINATION DETECTED (exact): {len(exact_overlaps)} overlap(s)")
        for o in exact_overlaps:
            print(f"    scenario={o['scenario_name']}  (from {o['scenario_source']})")
            print(f"      -> {o['training_file']} at row {o['training_row']}")

    if near_overlaps:
        label = "CONTAMINATION DETECTED (near-dup)" if args.strict else "WARNING (near-dup)"
        print(f"  {label}: {len(near_overlaps)} match(es) >= {args.near_dup_threshold}")
        for o in near_overlaps:
            print(
                f"    scenario={o['scenario_name']}  (from {o['scenario_source']})"
                f"  jaccard={o['jaccard']}"
            )
            print(f"      -> {o['training_file']} at row {o['training_row']}")

    if bff_overlaps:
        label = "CONTAMINATION DETECTED (bff)" if args.strict else "WARNING (bff)"
        print(
            f"  {label}: {len(bff_overlaps)} scenario(s) with 13-gram "
            f"presence-fraction >= {args.bff_threshold}"
        )
        for o in bff_overlaps:
            print(
                f"    scenario={o['scenario_name']}  (from {o['scenario_source']})"
                f"  presence_fraction={o['presence_fraction']} ({o['n_grams']} {o['n']}-grams)"
            )
    if bff_enabled and bff_skipped_short:
        print(
            f"  (bff leg skipped {bff_skipped_short} scenario(s) shorter than "
            f"{args.bff_n} tokens — covered by exact + near-dup legs)"
        )

    if embedding_skipped:
        print(
            "  (embedding leg: sentence-transformers unavailable; "
            "semantic cosine leg skipped — not a failure)"
        )
    elif embedding_overlaps:
        label = "CONTAMINATION DETECTED (embedding)" if args.strict else "WARNING (embedding)"
        print(
            f"  {label}: {len(embedding_overlaps)} match(es) cosine "
            f">= {args.embedding_threshold}"
        )
        for o in embedding_overlaps:
            print(
                f"    scenario={o['scenario_name']}  (from {o['scenario_source']})"
                f"  cosine={o['cosine']}"
            )
            print(f"      -> {o['training_file']} at row {o['training_row']}")

    # Per-source attribution table — always printed (free; no perf cost). Shows
    # WHICH training dataset each exact overlap came from.
    if per_source:
        print()
        print("  Per-source overlap (exact):")
        for src in sorted(per_source):
            stats = per_source[src]
            flag = "  <== CONTAMINATED" if stats["overlaps"] else ""
            print(
                f"    {src}: {stats['rows']} row(s), "
                f"{stats['overlaps']} overlap(s){flag}"
            )

    # Only exact overlap is fatal by default; opt-in legs escalate under --strict.
    opt_in_hits = bool(near_overlaps) or bool(bff_overlaps) or bool(embedding_overlaps)
    fatal = bool(exact_overlaps) or (opt_in_hits and args.strict)
    if not fatal:
        if opt_in_hits:
            print(
                "  (near-dup/bff/embedding matches are warnings; "
                "pass --strict to make them fatal)"
            )
        elif not exact_overlaps:
            print("  OK — no scenario stable_ids found in training data.")
    print("=" * 70)
    return 1 if fatal else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that F14 eval scenarios are not in training data",
    )
    parser.add_argument(
        "--scenarios-dir", default=str(_DEFAULT_SCENARIOS),
        help=f"Scenario YAML dir (default: {_DEFAULT_SCENARIOS})",
    )
    parser.add_argument(
        "--training-roots", nargs="*", default=None,
        help=(
            "One or more directories to scan for training rows "
            "(default: data/processed, data/staging, data/aggregated)"
        ),
    )
    parser.add_argument(
        "--near-dup", action="store_true",
        help=(
            "Also run the MinHash/LSH near-duplicate leg (catches paraphrase / "
            "light-edit leaks that exact hashing misses). Warning-only unless "
            "--strict. Off by default to keep the always-on gate fast."
        ),
    )
    parser.add_argument(
        "--near-dup-threshold", type=float, default=MINHASH_JACCARD_THRESHOLD,
        help=(
            "Jaccard threshold for the near-dup leg "
            f"(default: repo-canonical MINHASH_JACCARD_THRESHOLD={MINHASH_JACCARD_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--bff", action="store_true",
        help=(
            "Also run the 13-gram (word) presence-fraction leg (catches partial "
            "/ spliced span copies that exact + MinHash miss). Warning-only "
            "unless --strict. Off by default."
        ),
    )
    parser.add_argument(
        "--bff-n", type=int, default=_DEFAULT_BFF_N,
        help=(
            "Word n-gram size for the BFF leg "
            f"(default {_DEFAULT_BFF_N} — the GPT-3/C4/Allen-AI dedup convention)."
        ),
    )
    parser.add_argument(
        "--bff-threshold", type=float, default=_DEFAULT_BFF_THRESHOLD,
        help=(
            "Min fraction of a scenario's 13-grams present in training to flag "
            f"(default {_DEFAULT_BFF_THRESHOLD} — CONSERVATIVE; the local corpus "
            "is empty so this is uncalibrated and only flags near-total span "
            "copies. Precise calibration is a real-corpus CI follow-up)."
        ),
    )
    parser.add_argument(
        "--embedding", action="store_true",
        help=(
            "Also run the local KEYLESS embedding-cosine leg (catches semantic "
            "paraphrase). Reuses the pinned all-MiniLM-L6-v2 model; if "
            "sentence-transformers is absent the leg skips (no failure). "
            "Warning-only unless --strict. Off by default."
        ),
    )
    parser.add_argument(
        "--embedding-threshold", type=float, default=DEFAULT_NEAR_DUP_THRESHOLD,
        help=(
            "Cosine threshold for the embedding leg (default: inherited from "
            f"admission_gate.DEFAULT_NEAR_DUP_THRESHOLD={DEFAULT_NEAR_DUP_THRESHOLD}; "
            "not a new constant). Warning-only — un-revalidated for short strings."
        ),
    )
    parser.add_argument(
        "--strict", action="store_true",
        help=(
            "Enable ALL opt-in legs (near-dup + bff + embedding) AND treat their "
            "matches as fatal (exit 1). Exact overlap is always fatal."
        ),
    )
    parser.add_argument(
        "--allow-empty-corpus", action="store_true",
        help="Do not fail (exit 2) when zero training rows are scanned.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
