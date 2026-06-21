"""Stratified train/dev/test splitting + a one-time-test-use guard.

Why this exists
~~~~~~~~~~~~~~~
A judge-calibration harness is only trustworthy if (a) the dev/test slices
preserve BOTH the class balance and the per-attack-category mix of the full
corpus — otherwise a recall number is measured on an unrepresentative slice —
and (b) the held-out test set is genuinely held out: scored at most once, so it
can't quietly become a tuning set (the classic "train-on-test" failure this
codebase has already been bitten by; see BENCHMARK_SPRINT decontam tasks).

Two pieces:

``stratified_split``
    Splits a list of samples into ``{"train", "dev", "test"}`` stratifying on
    the COMPOSITE key ``(label, attack_category)`` so each split keeps class
    AND category proportions. Uses
    :class:`sklearn.model_selection.StratifiedShuffleSplit` (sklearn is a core
    dep) for the groups large enough to split; groups too small to give every
    split at least one member are assigned round-robin (documented below).
    stable_id sets across the three splits are guaranteed disjoint.

``test_manifest_hash`` / ``record_test_use`` / ``check_test_unused``
    A sha256 manifest over the test set's stable_ids plus an append-only log,
    so a CI step can assert a given test slice has never been scored before
    (``allow_reuse=False`` raises on a second use).

Naming note (DataSplit reconciliation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The public split dict uses the keys ``"train" / "dev" / "test"`` (the contract
this harness is built to). The repo's :class:`na0s.dataset.schema.DataSplit`
enum spells the middle split ``VAL`` (``DataSplit.VAL.value == "val"``), so the
mapping ``"dev" -> DataSplit.VAL`` is exposed via :data:`SPLIT_TO_DATASPLIT` and
:func:`split_key_to_datasplit` rather than hard-coded at call sites — "dev" and
"val" are the same held-out-tuning slice under two names.

Dependency posture: stdlib + ``numpy`` + ``scikit-learn`` (both core deps in
``pyproject.toml``). No network, no LLM — this is pure bookkeeping.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from na0s.dataset.schema import DataLabel, DataSplit

# ── tunables (documented, not magic) ─────────────────────────────────────────

#: Default split proportions (train, dev, test). 70/15/15 is the conventional
#: choice for a held-out dev+test pair: a large-enough train slice with two
#: equal evaluation slices (one to tune on = dev, one scored once = test).
#: Passed as a parameter so callers can override; never hard-coded downstream.
DEFAULT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)

#: Default RNG seed. Fixed so splits are reproducible across runs/machines —
#: a different seed yields a different (still valid) partition. Exposed as a
#: parameter, not buried, so an experiment can vary it deliberately.
DEFAULT_SEED: int = 0

#: Maps the public split-dict keys to the repo's DataSplit enum. "dev" and the
#: enum's "val" are the same tuning slice; see module docstring.
SPLIT_TO_DATASPLIT: dict[str, DataSplit] = {
    "train": DataSplit.TRAIN,
    "dev": DataSplit.VAL,
    "test": DataSplit.TEST,
}

#: Default filename component for the test-use log.
_DEFAULT_LOG_NAME = "test_use_log.jsonl"


def split_key_to_datasplit(key: str) -> DataSplit:
    """Translate a split-dict key (``"train"``/``"dev"``/``"test"``) to DataSplit.

    Raises ``KeyError`` for an unknown key so a typo surfaces loudly rather than
    silently mis-tagging a sample.
    """
    return SPLIT_TO_DATASPLIT[key]


# ── label / category / id extraction (duck-typed) ────────────────────────────


def _coerce_label(raw: Any) -> int:
    """Normalize a sample's label to ``0`` (benign) / ``1`` (injection).

    Accepts the shapes that flow through this codebase: ``int`` 0/1, ``bool``,
    :class:`DataLabel`, and the string spellings (``"injection"``/``"benign"``,
    ``"malicious"``/``"safe"``, ``"1"``/``"0"``, ``"true"``/``"false"``). Raises
    ``ValueError`` on anything else so a mislabeled sample can't silently land
    in the wrong stratum.
    """
    if isinstance(raw, bool):  # bool before int (bool IS an int in Python)
        return int(raw)
    if isinstance(raw, (int, np.integer)):
        if int(raw) in (0, 1):
            return int(raw)
        raise ValueError(f"integer label must be 0 or 1, got {raw!r}")
    if isinstance(raw, DataLabel):
        return 1 if raw == DataLabel.INJECTION else 0
    if isinstance(raw, str):
        key = raw.strip().lower()
        positive = {"injection", "malicious", "attack", "unsafe", "1", "true", "blocked"}
        negative = {"benign", "safe", "0", "false", "allowed"}
        if key in positive:
            return 1
        if key in negative:
            return 0
    raise ValueError(f"unrecognized label value: {raw!r}")


def _get_attr_or_key(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from either an object attribute or a mapping key."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_label(sample: Any) -> int:
    """Pull and normalize the label from a sample object/dict.

    Looks at ``label`` first (Na0SSample / dict convention); falls back to
    ``expected_verdict`` (Scenario convention: ``"blocked"`` == positive).
    """
    raw = _get_attr_or_key(sample, "label", None)
    if raw is None:
        raw = _get_attr_or_key(sample, "expected_verdict", None)
    if raw is None:
        raise ValueError(
            f"sample {sample!r} exposes neither a 'label' nor an "
            "'expected_verdict' field"
        )
    return _coerce_label(raw)


def _extract_category(sample: Any) -> str:
    """Pull the attack category, normalized to a non-empty string.

    Missing/empty categories collapse to the sentinel ``"__none__"`` so benign
    samples (which legitimately have no attack category) still form a coherent
    stratum keyed on ``(label, "__none__")`` rather than scattering.
    """
    cat = _get_attr_or_key(sample, "attack_category", None)
    if cat is None:
        # secondary conventions seen in the codebase
        cat = _get_attr_or_key(sample, "functional_category", None)
    if cat is None:
        cat = _get_attr_or_key(sample, "semantic_category", None)
    if cat is None or (isinstance(cat, str) and not cat.strip()):
        return "__none__"
    return str(cat)


def _extract_stable_id(sample: Any) -> str:
    """Pull the stable_id; raise if absent (it's the disjointness key)."""
    sid = _get_attr_or_key(sample, "stable_id", None)
    if not sid:
        raise ValueError(
            f"sample {sample!r} has no stable_id; cannot guarantee disjoint "
            "splits without a stable identity key"
        )
    return str(sid)


def _composite_key(sample: Any) -> tuple[int, str]:
    """The stratification key: ``(label, attack_category)``."""
    return (_extract_label(sample), _extract_category(sample))


# ── the split ────────────────────────────────────────────────────────────────


def _normalize_ratios(ratios: Sequence[float]) -> tuple[float, float, float]:
    """Validate + normalize a 3-tuple of ratios to sum to 1.0."""
    if len(ratios) != 3:
        raise ValueError(f"ratios must be a 3-tuple (train, dev, test), got {ratios!r}")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative, got {ratios!r}")
    total = float(sum(ratios))
    if total <= 0:
        raise ValueError(f"ratios must sum to a positive number, got {ratios!r}")
    return (ratios[0] / total, ratios[1] / total, ratios[2] / total)


def stratified_split(
    samples: Sequence[Any],
    ratios: Sequence[float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> dict[str, list]:
    """Stratified train/dev/test split on the composite ``(label, category)`` key.

    Parameters
    ----------
    samples
        Sequence of sample objects or dicts, each exposing a ``label`` (0/1,
        :class:`DataLabel`, or a known string spelling — see
        :func:`_coerce_label`), an ``attack_category`` (or
        ``functional_category``/``semantic_category``; missing collapses to a
        shared benign stratum), and a ``stable_id``.
    ratios
        ``(train, dev, test)`` proportions; normalized to sum to 1.0. Defaults
        to :data:`DEFAULT_RATIOS` (70/15/15) — documented, overridable.
    seed
        RNG seed for reproducibility (default :data:`DEFAULT_SEED`).

    Returns
    -------
    dict
        ``{"train": [...], "dev": [...], "test": [...]}`` containing the SAME
        object types that were passed in. The three splits' stable_id sets are
        guaranteed disjoint and their union equals the input id set.

    Algorithm
    ---------
    Samples are bucketed by ``(label, attack_category)``. A group is "splittable"
    only if it has at least 3 members (one minimum per split); for those groups
    we draw indices with two nested
    :class:`~sklearn.model_selection.StratifiedShuffleSplit` passes (first carve
    off test, then split the remainder into train/dev) so each split keeps the
    group's proportions.

    Groups too small to give every split a member (< 3 members) cannot be split
    by sklearn without dropping a split, so they are assigned ROUND-ROBIN across
    train -> dev -> test in a deterministic (seed-shuffled) order. Round-robin
    (rather than dumping every small group into train) keeps tiny categories
    represented in dev/test instead of vanishing from evaluation — important for
    the long tail of attack categories where each may have only 1-2 samples.
    """
    r_train, r_dev, r_test = _normalize_ratios(ratios)
    n = len(samples)
    out: dict[str, list] = {"train": [], "dev": [], "test": []}
    if n == 0:
        return out

    # Bucket indices by composite key, preserving input order within a bucket.
    groups: dict[tuple[int, str], list[int]] = {}
    for i, s in enumerate(samples):
        groups.setdefault(_composite_key(s), []).append(i)

    rng = np.random.default_rng(seed)

    # Deterministic group order so round-robin assignment is reproducible.
    ordered_keys = sorted(groups.keys())

    # Counters to balance round-robin across splits globally (so small groups
    # don't all pile into "train"). Rotating start point comes from the seed.
    rr_targets = ["train", "dev", "test"]
    rr_cursor = int(rng.integers(0, 3))

    assigned_train: list[int] = []
    assigned_dev: list[int] = []
    assigned_test: list[int] = []

    for key in ordered_keys:
        idx = np.array(groups[key])
        g = len(idx)

        if g >= 3 and r_test > 0 and r_dev > 0:
            # Splittable: two-stage StratifiedShuffleSplit. The y here is a
            # constant (all members share the composite key) so SSS just gives
            # a reproducible proportional shuffle within the group.
            y = np.zeros(g, dtype=int)

            # Stage 1: carve off the test fraction.
            n_test = max(1, int(round(g * r_test)))
            n_test = min(n_test, g - 2)  # leave >=1 each for train+dev
            sss1 = StratifiedShuffleSplit(
                n_splits=1, test_size=n_test, random_state=seed
            )
            rest_local, test_local = next(sss1.split(idx.reshape(-1, 1), y))
            assigned_test.extend(idx[test_local].tolist())

            rest = idx[rest_local]
            gr = len(rest)
            # Stage 2: split the remainder into train/dev preserving their ratio.
            dev_frac_of_rest = r_dev / (r_train + r_dev) if (r_train + r_dev) else 0.0
            n_dev = max(1, int(round(gr * dev_frac_of_rest)))
            n_dev = min(n_dev, gr - 1)  # leave >=1 for train
            y2 = np.zeros(gr, dtype=int)
            sss2 = StratifiedShuffleSplit(
                n_splits=1, test_size=n_dev, random_state=seed
            )
            train_local, dev_local = next(sss2.split(rest.reshape(-1, 1), y2))
            assigned_train.extend(rest[train_local].tolist())
            assigned_dev.extend(rest[dev_local].tolist())
        else:
            # Too small to split without starving a split: round-robin. Shuffle
            # within the group first (seeded) so it's not always the input-first
            # element that lands in train.
            shuffled = idx.copy()
            rng.shuffle(shuffled)
            for j in shuffled.tolist():
                target = rr_targets[rr_cursor % 3]
                rr_cursor += 1
                if target == "train":
                    assigned_train.append(j)
                elif target == "dev":
                    assigned_dev.append(j)
                else:
                    assigned_test.append(j)

    # Materialize splits in original input order for stable, readable output.
    for j in sorted(assigned_train):
        out["train"].append(samples[j])
    for j in sorted(assigned_dev):
        out["dev"].append(samples[j])
    for j in sorted(assigned_test):
        out["test"].append(samples[j])

    # Hard invariant: the three stable_id sets must be disjoint and cover all.
    _assert_disjoint(out)
    return out


def _assert_disjoint(split: dict[str, list]) -> None:
    """Raise if any stable_id appears in more than one split (defensive check)."""
    seen: dict[str, str] = {}
    for name in ("train", "dev", "test"):
        for s in split[name]:
            sid = _extract_stable_id(s)
            if sid in seen:
                raise RuntimeError(
                    f"stable_id {sid!r} appears in both {seen[sid]!r} and "
                    f"{name!r} splits — splits are not disjoint"
                )
            seen[sid] = name


# ── one-time-test-use guard ──────────────────────────────────────────────────


def test_manifest_hash(test_samples: Sequence[Any]) -> str:
    """sha256 over the SORTED, de-duplicated stable_id set of a test slice.

    Order-independent (ids are sorted) and content-addressed: two test slices
    with the same membership produce the same hash regardless of construction
    order, so the use-log keys on identity-of-contents, not object identity.
    """
    ids = sorted({_extract_stable_id(s) for s in test_samples})
    h = hashlib.sha256()
    for sid in ids:
        h.update(sid.encode("utf-8"))
        h.update(b"\n")  # delimiter so concatenation is unambiguous
    return h.hexdigest()


def _resolve_log_path(log_path: str | Path) -> Path:
    """A directory log_path gets the default filename; a file path is used as-is."""
    p = Path(log_path)
    if p.is_dir():
        return p / _DEFAULT_LOG_NAME
    return p


def record_test_use(manifest_hash: str, log_path: str | Path) -> None:
    """Append a usage record for ``manifest_hash`` to the JSONL log.

    Each line is ``{"manifest_hash": ..., "recorded_at": <iso8601 utc>}``.
    The parent directory is created if needed. Append-only — never rewrites
    history, so the audit trail of test scorings is tamper-evident by inspection.
    """
    path = _resolve_log_path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "manifest_hash": manifest_hash,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _logged_hashes(path: Path) -> set[str]:
    """Return the set of manifest hashes already present in the log (if any)."""
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A corrupt line shouldn't mask a real prior use of OTHER lines;
                # skip it rather than crash the guard.
                continue
            h = rec.get("manifest_hash")
            if h:
                seen.add(h)
    return seen


def check_test_unused(
    manifest_hash: str,
    log_path: str | Path,
    allow_reuse: bool = False,
) -> None:
    """Raise ``RuntimeError`` if ``manifest_hash`` was already used.

    This is the one-time-test guard: a held-out test slice should be SCORED
    exactly once so it can't drift into a tuning set. Call this BEFORE scoring,
    then :func:`record_test_use` after, so a second attempt to score the same
    test membership fails loudly.

    ``allow_reuse=True`` downgrades a prior use to a no-op (for the legitimate
    case of re-running an already-final evaluation deliberately).
    """
    path = _resolve_log_path(log_path)
    if manifest_hash in _logged_hashes(path):
        if allow_reuse:
            return
        raise RuntimeError(
            f"test manifest {manifest_hash[:12]}... has already been used "
            f"(logged in {path}). Scoring a held-out test set twice turns it "
            "into a tuning set; pass allow_reuse=True only if this re-run is "
            "intentional."
        )
