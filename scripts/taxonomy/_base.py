"""Base class for all taxonomy probes."""

import functools
import importlib.resources
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from scripts.safe_yaml import safe_load_yaml


def _find_project_root():
    """Locate the project root using importlib.resources with Path fallback.

    Tries importlib.resources first (works in zipped/installed packages),
    then falls back to __file__-based resolution for script-mode execution.
    """
    try:
        # importlib.resources can resolve package location even in zipped installs
        pkg_dir = importlib.resources.files("scripts.taxonomy")
        # scripts/taxonomy -> scripts -> project root
        root = Path(str(pkg_dir)).resolve().parent.parent
        if (root / "data" / "taxonomy.yaml").exists():
            return root
    except (TypeError, FileNotFoundError, ModuleNotFoundError, Exception):
        pass
    # Fallback for script-mode execution
    return Path(__file__).resolve().parent.parent.parent


@dataclass
class ClassifierOutput:
    """Structured output contract for classify_fn.

    Every classifier (rule engine, ML model, Llama, HF pipeline) must
    return this so evaluate() never breaks on format drift.
    """
    label: str
    confidence: float
    hits: list = field(default_factory=list)
    rejected: bool = False
    anomaly_flags: list = field(default_factory=list)

    @classmethod
    def from_tuple(cls, tup):
        """Wrap a legacy (label, prob, hits, l0) or (label, prob, hits, l0, detailed_hits) tuple."""
        if not isinstance(tup, (list, tuple)) or len(tup) not in (4, 5):
            raise TypeError(
                "classify_fn must return (label, prob, hits, l0[, detailed_hits]) or "
                "ClassifierOutput, got {!r}".format(type(tup).__name__)
            )
        label, prob, hits, l0 = tup[:4]
        return cls(
            label=label,
            confidence=prob,
            hits=hits if hits else [],
            rejected=getattr(l0, "rejected", False) if l0 is not None else False,
            anomaly_flags=list(getattr(l0, "anomaly_flags", [])) if l0 is not None else [],
        )

_PROJECT_ROOT = _find_project_root()
_TAXONOMY_PATH = Path(
    os.environ.get("TAXONOMY_YAML_PATH", _PROJECT_ROOT / "data" / "taxonomy.yaml")
)

# Detection labels — single source of truth for label matching
DETECTION_LABELS = frozenset({"MALICIOUS", "BLOCKED"})

_STRIP_NON_ALPHA = re.compile(r"[^\w\s]")

_L0_FLAG_MAP = {
    "nfkc_changed": "D5", "zero_width_stripped": "D5.2",
    "hidden_html_content": "I2", "high_compression_ratio": "D8",
    "known_malicious_exact": "D1", "known_malicious_normalized": "D1",
    "known_malicious_token_pattern": "D1",
    "base64": "D4.1", "url_encoded": "D4.2", "hex": "D4.3",
    "high_entropy": "D4", "punctuation_flood": "D4",
    "weird_casing": "D4",
}


def _is_detected_label(label):
    """Check if label indicates a malicious/blocked detection.

    Strips emojis and punctuation, then does exact word matching
    against DETECTION_LABELS.  This avoids substring false positives
    like 'NOT_MALICIOUS' matching 'MALICIOUS'.
    """
    words = set(_STRIP_NON_ALPHA.sub("", label).upper().split())
    return bool(words & DETECTION_LABELS)


@functools.lru_cache(maxsize=1)
def _load_taxonomy():
    """Load and cache taxonomy YAML (thread-safe via lru_cache)."""
    path = _TAXONOMY_PATH

    # Path-containment check — prevent path traversal via env var
    resolved = path.resolve()
    allowed = (_PROJECT_ROOT / "data").resolve()
    if not str(resolved).startswith(str(allowed) + os.sep) and resolved != allowed:
        raise ValueError(
            "TAXONOMY_YAML_PATH must be within data/ directory, "
            "got: {}".format(resolved)
        )

    data = safe_load_yaml(path)
    if not isinstance(data, dict) or "categories" not in data:
        raise ValueError(
            "Taxonomy YAML missing 'categories' key: {}".format(path)
        )

    # Schema validation — each category must be a dict with a 'name' key
    categories = data["categories"]
    if not isinstance(categories, dict):
        raise ValueError(
            "Taxonomy 'categories' must be a dict, got {}: {}".format(
                type(categories).__name__, path
            )
        )
    for cat_id, cat in categories.items():
        if not isinstance(cat, dict):
            raise ValueError(
                "Category '{}' must be a dict, got {}: {}".format(
                    cat_id, type(cat).__name__, path
                )
            )
        if "name" not in cat:
            raise ValueError(
                "Category '{}' missing required 'name' field: {}".format(
                    cat_id, path
                )
            )

    return data


def clear_taxonomy_cache():
    """Reset cached taxonomy data (for tests and live-reload)."""
    _load_taxonomy.cache_clear()


class Probe:
    """Base class for all taxonomy probes.

    Subclasses must set ``category_id`` and implement ``generate()``.
    Tags, name, severity, and expected_layers are auto-loaded from
    ``data/taxonomy.yaml`` so the YAML file remains the single source of truth.
    """

    category_id = ""  # e.g. "D1", "E", "T"

    def __init__(self):
        if not self.category_id:
            raise ValueError(
                "{} must set category_id".format(type(self).__name__)
            )
        tax = _load_taxonomy()
        cat = tax["categories"].get(self.category_id)
        if cat is None:
            available = ", ".join(sorted(tax["categories"].keys()))
            raise KeyError(
                "category_id '{}' not found in taxonomy. "
                "Available: {}".format(self.category_id, available)
            )
        self.name = cat.get("name", "")
        self.tags = cat.get("tags", [])
        self.severity = cat.get("severity", "")
        self.expected_layers = cat.get("expected_layers", [])
        self.techniques = cat.get("techniques", {})

    def generate(self):
        """Return list of (text, technique_id) or (text, technique_id, metadata) tuples."""
        raise NotImplementedError

    def _validated_samples(self):
        """Call generate() and validate its return structure.

        Accepts both 2-tuples (text, technique_id) and 3-tuples
        (text, technique_id, metadata_dict).  Normalizes all to
        3-tuples with an empty dict when metadata is absent.
        """
        samples = self.generate()
        probe_name = type(self).__name__
        if not isinstance(samples, (list, tuple)):
            raise TypeError(
                "{}.generate() must return a list of (text, technique_id) "
                "tuples, got {}".format(probe_name, type(samples).__name__)
            )
        normalized = []
        for i, item in enumerate(samples):
            if not isinstance(item, (list, tuple)) or len(item) not in (2, 3):
                raise TypeError(
                    "{}.generate() item [{}] must be a (text, technique_id) "
                    "or (text, technique_id, metadata) tuple, "
                    "got {!r}".format(probe_name, i, item)
                )
            text, tech_id = item[0], item[1]
            meta = item[2] if len(item) == 3 else {}
            if not isinstance(text, str):
                raise TypeError(
                    "{}.generate() item [{}]: text must be str, "
                    "got {}".format(probe_name, i, type(text).__name__)
                )
            if not isinstance(tech_id, str):
                raise TypeError(
                    "{}.generate() item [{}]: technique_id must be str, "
                    "got {}".format(probe_name, i, type(tech_id).__name__)
                )
            if not isinstance(meta, dict):
                raise TypeError(
                    "{}.generate() item [{}]: metadata must be dict, "
                    "got {}".format(probe_name, i, type(meta).__name__)
                )
            normalized.append((text, tech_id, meta))
        return normalized

    _rule_map = None  # cached {rule_name: [technique_ids]}

    @classmethod
    def _resolve_techniques(cls, output):
        """Extract attributed technique IDs from a ClassifierOutput.

        Returns a set of technique ID strings the detector *claims*
        were present.  This mirrors the resolution logic in scan()
        but is self-contained to avoid circular imports.
        """
        if cls._rule_map is None:
            from na0s.rules import RULES      # late import — avoids circular deps
            cls._rule_map = {r.name: r.technique_ids for r in RULES}
        techniques = set()
        for name in output.hits:
            techniques.update(cls._rule_map.get(name, []))
        for flag in output.anomaly_flags:
            mapped = _L0_FLAG_MAP.get(flag)
            if mapped:
                techniques.add(mapped)
        return techniques

    def evaluate(self, classify_fn, confidence_threshold=0.0):
        """Run all samples through the detector.

        Classifies every sample once and stores per-sample scores so
        metrics can be recomputed at any threshold via
        ``recall_at_threshold()`` without re-running the classifier.

        Each sample tracks two independent booleans:
        - **flagged**: the detector said "malicious" (detection recall)
        - **attributed**: the detector identified the correct technique
          (attribution accuracy)

        Args:
            classify_fn: callable(text) -> ClassifierOutput or
                (label, prob, hits, l0) tuple (legacy).
            confidence_threshold: minimum prob to count as detected
                (default 0.0 = any confidence counts).

        Returns:
            dict with per-probe and per-technique recall, plus a
            ``scores`` list of per-sample dicts for threshold tuning.
        """
        samples = self._validated_samples()
        scores = []

        for text, tech_id, meta in samples:
            raw = classify_fn(text)
            output = raw if isinstance(raw, ClassifierOutput) else ClassifierOutput.from_tuple(raw)
            flagged = output.rejected or _is_detected_label(output.label)
            attributed_ids = self._resolve_techniques(output)
            score = {
                "text": text[:200],
                "technique_id": tech_id,
                "confidence": output.confidence,
                "flagged": flagged,
                "attributed": tech_id in attributed_ids,
                "attributed_ids": sorted(attributed_ids),
            }
            if meta:
                score["metadata"] = meta
            scores.append(score)

        tax = _load_taxonomy()
        results = {
            "probe": self.category_id,
            "name": self.name,
            "severity": self.severity,
            "tags": self.tags,
            "total": len(scores),
            "scores": scores,
            "meta": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "confidence_threshold": confidence_threshold,
                "taxonomy_version": tax.get("version", "unknown"),
            },
        }
        results.update(recall_at_threshold(results, confidence_threshold))
        return results


def recall_at_threshold(results, threshold=0.0):
    """Compute recall metrics from stored scores at any threshold.

    Can be called repeatedly on the same results dict to sweep
    thresholds without re-running the classifier.

    Reports:
    - **detection rate**: flagged as malicious (any technique)
    - **attribution accuracy**: correct technique identified (top-1
      exact match *and* top-k set membership are equivalent here
      because rule-based detection has no ranked output — all
      matching rules fire equally)
    - **confusion**: when expected T_x but detector attributed T_y,
      ``confusion[T_x][T_y]`` counts how often that happened

    Returns a dict ready to merge into the results dict.
    """
    detected = 0
    missed = 0
    attributed = 0
    false_positives = 0
    true_negatives = 0
    by_technique = {}
    by_difficulty = {}
    by_evasion_type = {}
    missed_samples = []
    confusion = {}  # {expected_id: {attributed_id: count}}

    for s in results["scores"]:
        tech_id = s["technique_id"]
        is_benign = tech_id.endswith("_benign")
        is_detected = s["flagged"] and s["confidence"] >= threshold
        is_attributed = is_detected and s.get("attributed", False)
        attr_ids = s.get("attributed_ids", [])
        meta = s.get("metadata", {})
        difficulty = meta.get("difficulty")
        evasion = meta.get("evasion_type")

        if tech_id not in by_technique:
            by_technique[tech_id] = {"detected": 0, "missed": 0, "attributed": 0}

        # Track by difficulty when metadata is present
        if difficulty and not is_benign:
            if difficulty not in by_difficulty:
                by_difficulty[difficulty] = {"detected": 0, "missed": 0, "total": 0}
            by_difficulty[difficulty]["total"] += 1

        # Track by evasion_type when metadata is present
        if evasion and not is_benign:
            if evasion not in by_evasion_type:
                by_evasion_type[evasion] = {"detected": 0, "missed": 0, "total": 0}
            by_evasion_type[evasion]["total"] += 1

        if is_benign:
            # Benign samples: flagging is a false positive
            if is_detected:
                false_positives += 1
                by_technique[tech_id]["detected"] += 1
            else:
                true_negatives += 1
            continue

        if is_detected:
            detected += 1
            by_technique[tech_id]["detected"] += 1
            if difficulty:
                by_difficulty[difficulty]["detected"] += 1
            if evasion:
                by_evasion_type[evasion]["detected"] += 1
            # Build confusion: record every attributed technique
            if tech_id not in confusion:
                confusion[tech_id] = {}
            if attr_ids:
                for aid in attr_ids:
                    confusion[tech_id][aid] = confusion[tech_id].get(aid, 0) + 1
            else:
                # Flagged but no technique attributed (ML-only detection)
                confusion[tech_id]["_none"] = confusion[tech_id].get("_none", 0) + 1
        else:
            missed += 1
            by_technique[tech_id]["missed"] += 1
            if difficulty:
                by_difficulty[difficulty]["missed"] += 1
            if evasion:
                by_evasion_type[evasion]["missed"] += 1
            missed_samples.append(
                (s["text"], tech_id, s["confidence"])
            )

        if is_attributed:
            attributed += 1
            by_technique[tech_id]["attributed"] += 1

    # Compute per-difficulty recall and effective_difficulty
    for d in by_difficulty.values():
        d["recall"] = d["detected"] / d["total"] if d["total"] else 0.0
        d["effective_difficulty"] = round(1.0 - d["recall"], 4)

    # Compute per-evasion_type recall and effective_difficulty
    for e in by_evasion_type.values():
        e["recall"] = e["detected"] / e["total"] if e["total"] else 0.0
        e["effective_difficulty"] = round(1.0 - e["recall"], 4)

    malicious_total = detected + missed
    benign_total = false_positives + true_negatives
    return {
        "detected": detected,
        "missed": missed,
        "attributed": attributed,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "recall": detected / malicious_total if malicious_total else 0.0,
        "attribution_rate": attributed / detected if detected else 0.0,
        "false_positive_rate": (
            false_positives / benign_total if benign_total else 0.0
        ),
        "by_technique": by_technique,
        "by_difficulty": by_difficulty,
        "by_evasion_type": by_evasion_type,
        "confusion": confusion,
        "missed_samples": missed_samples,
    }
