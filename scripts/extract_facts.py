#!/usr/bin/env python3
"""Extract structural facts about Na0S directly from source.

Writes docs/facts.yaml. No hardcoded values — every number is derived from
AST parses, runtime imports, taxonomy YAML, or filesystem queries.

The README and other docs should source numbers from this file rather than
hand-typing them. Wired into .pre-commit-config.yaml so it regenerates on
every commit; if the output diverges from the committed version, pre-commit
will fail and ask the user to stage the new docs/facts.yaml.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "docs" / "facts.yaml"


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _read_ast(rel_path: str) -> ast.AST:
    path = REPO_ROOT / rel_path
    return ast.parse(path.read_text())


def _find_module_assign(tree: ast.AST, name: str):
    """Return the rhs of `name = <rhs>` at module scope (first match)."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return node.value
    return None


def _list_literal_len(rel_path: str, name: str):
    tree = _read_ast(rel_path)
    rhs = _find_module_assign(tree, name)
    if isinstance(rhs, ast.List):
        return len(rhs.elts)
    return None


def _module_to_path(module: str, level: int, anchor: str):
    """Resolve an ImportFrom (`from MODULE import ...`) to a file path.

    `anchor` is the rel path of the file doing the import. `level` is the
    number of leading dots in a relative import (0 for absolute).
    Absolute imports must start with `na0s.` and resolve to `src/na0s/...`.
    Returns None if the import can't be mapped to a single .py file.
    """
    if level == 0:
        if not module or not module.startswith("na0s"):
            return None
        rel_parts = module.split(".")
        # na0s.X.Y -> src/na0s/X/Y.py
        return "src/" + "/".join(rel_parts) + ".py"
    # relative: walk up from anchor's directory
    here = Path(anchor).parent
    for _ in range(level - 1):
        here = here.parent
    if module:
        return str(here / (module.replace(".", "/") + ".py"))
    return str(here / "__init__.py")


def _resolve_list_len(rel_path: str, name: str, visited=None):
    """Find len(<list literal>) for `name` by following AST imports.

    Handles three cases:
      1. Direct literal: `name = [...]` in this file.
      2. Aliased import: `from X import NAME` (or `from X import NAME as L`).
      3. Wildcard re-export: `from X import *`.
    Returns None if no list literal can be located.
    """
    if visited is None:
        visited = set()
    key = (rel_path, name)
    if key in visited:
        return None
    visited.add(key)

    path = REPO_ROOT / rel_path
    if not path.exists():
        return None

    direct = _list_literal_len(rel_path, name)
    if direct is not None:
        return direct

    tree = _read_ast(rel_path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        target = _module_to_path(node.module or "", node.level, rel_path)
        if not target:
            continue
        for alias in node.names:
            # Exact-name re-export
            if alias.name == name and (alias.asname is None or alias.asname == name):
                v = _resolve_list_len(target, name, visited)
                if v is not None:
                    return v
            # Wildcard re-export
            if alias.name == "*":
                v = _resolve_list_len(target, name, visited)
                if v is not None:
                    return v
    return None


def _call_name(call: ast.Call):
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


# ---------------------------------------------------------------------------
# Fact extractors
# ---------------------------------------------------------------------------


def get_public_exports():
    """__all__ from src/na0s/__init__.py via AST."""
    tree = _read_ast("src/na0s/__init__.py")
    rhs = _find_module_assign(tree, "__all__")
    if not isinstance(rhs, ast.List):
        return []
    return [e.value for e in rhs.elts if isinstance(e, ast.Constant)]


def get_rule_count():
    """Total len(RULES) for the L1 rule registry, derived purely via AST.

    rules/rules_registry.py builds RULES as a literal list, then calls
    RULES.extend(_X) for several aliases imported via `from .. import X as _X`.
    For each extend, we resolve the alias to the source module and AST-count
    its list literal. Returns the sum plus an itemized breakdown.
    """
    rel = "src/na0s/rules/rules_registry.py"
    tree = _read_ast(rel)
    literal = _list_literal_len(rel, "RULES") or 0

    # Map local alias -> (level, module, source_attr) from ImportFrom nodes
    alias_to_source = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            level = node.level
            for alias in node.names:
                local = alias.asname or alias.name
                alias_to_source[local] = (level, mod, alias.name)

    extends = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "extend"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "RULES"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Name)
        ):
            arg_name = node.args[0].id
            if arg_name not in alias_to_source:
                extends.append({
                    "alias": arg_name,
                    "count": None,
                    "error": "alias not found in ImportFrom",
                })
                continue
            level, mod, attr = alias_to_source[arg_name]
            src_rel = _module_to_path(mod, level, rel)
            count = _resolve_list_len(src_rel, attr) if src_rel else None
            extends.append({
                "alias": arg_name,
                "source_attr": attr,
                "source_file": src_rel,
                "count": count,
            })

    total = literal + sum(e.get("count") or 0 for e in extends)
    return {
        "literal": literal,
        "extends": extends,
        "total_ast": total,
    }


def get_taxonomy():
    """Parse data/taxonomy.yaml for category and technique counts."""
    p = REPO_ROOT / "data" / "taxonomy.yaml"
    with p.open() as f:
        data = yaml.safe_load(f)
    cats = data.get("categories", {})
    counts = {cid: len(c.get("techniques", {})) for cid, c in cats.items()}
    return {
        "category_count": len(cats),
        "technique_count_total": sum(counts.values()),
        "techniques_by_category": counts,
    }


def get_test_count():
    """Run pytest --collect-only and parse the collected count."""
    try:
        r = subprocess.run(
            ["python3", "-m", "pytest", "--collect-only", "-q", "tests/"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return {"count": None, "error": repr(exc)}

    blob = (r.stdout or "") + "\n" + (r.stderr or "")
    # Match "N tests collected" or "N/M tests collected"
    m = re.search(r"(\d+)\s+tests?\s+collected", blob)
    err_m = re.search(r"(\d+)\s+errors?\s+", blob)
    return {
        "count": int(m.group(1)) if m else None,
        "collection_errors": int(err_m.group(1)) if err_m else 0,
        "pytest_exit_code": r.returncode,
    }


def get_test_files():
    """Glob count of tests/**/test_*.py."""
    return sum(1 for _ in (REPO_ROOT / "tests").rglob("test_*.py"))


# Names accepted as "detection-related" inside classify_prompt(). Either a
# name matching one of the verb-prefix patterns, or an exact-match against
# the EXACT set below.
_VERB_PREFIX_RE = re.compile(
    r"^(detect_|scan_|analyze_|extract_|score_|compute_|classify|predict|"
    r"get_|sniff_)"
)
_EXACT_DETECTION_NAMES = frozenset({
    "rule_score_detailed",
    "obfuscation_scan",
    "_weighted_decision",
    "_decode_literal_escapes",
    "_extract_concatenation_game",
    "_transform",
    "quick_normalize_concat",
    "register_malicious",
    "calculate_safe_content_score",
    "get_embedding_classifier",
    "_get_pg_classifier_score",
    "detect_multilingual_intents",
    "_is_legitimate_roleplay",
    "_has_contextual_framing",
})


def get_detection_signals():
    """AST-walk classify_prompt() and list detection-related calls in order."""
    rel = "src/na0s/predict.py"
    tree = _read_ast(rel)
    target = None
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == "classify_prompt":
            target = n
            break
    if target is None:
        return {"calls": [], "filter_doc": ""}

    seen_order = []
    seen_set = set()
    for node in ast.walk(target):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if not name:
                continue
            if name in _EXACT_DETECTION_NAMES or _VERB_PREFIX_RE.match(name):
                if name not in seen_set:
                    seen_set.add(name)
                    seen_order.append(name)

    return {
        "filter_doc": (
            "Names matching prefixes (detect_, scan_, analyze_, extract_, "
            "score_, compute_, classify*, predict*, get_, sniff_) plus an "
            "exact-match allowlist for non-verb-prefixed callables in "
            "classify_prompt(). The list preserves AST traversal order."
        ),
        "calls": seen_order,
    }


def get_constants():
    """Pull constants from na0s.predict at runtime (single source of truth)."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    os.environ.setdefault("NA0S_EMBEDDING_ENABLED", "0")
    warnings.filterwarnings("ignore")
    out = {}
    try:
        import importlib

        mod = importlib.import_module("na0s.predict")
        for name in (
            "DECISION_THRESHOLD",
            "MAX_INPUT_LENGTH",
            "SCAN_TIMEOUT",
            "MAX_CHUNKS",
            "_CHUNK_WORD_THRESHOLD",
        ):
            out[name] = getattr(mod, name, None)
    except Exception as exc:
        out["_error"] = repr(exc)
    return out


def get_l16_detectors():
    """Class names extending MultiTurnDetector under layer16/detectors/."""
    base = REPO_ROOT / "src" / "na0s" / "layer16" / "detectors"
    found = []
    for p in sorted(base.glob("*.py")):
        if p.name in ("__init__.py", "base_detector.py"):
            continue
        tree = ast.parse(p.read_text())
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for base_node in node.bases:
                bname = None
                if isinstance(base_node, ast.Name):
                    bname = base_node.id
                elif isinstance(base_node, ast.Attribute):
                    bname = base_node.attr
                if bname == "MultiTurnDetector":
                    found.append({
                        "name": node.name,
                        "file": str(p.relative_to(REPO_ROOT)),
                        "line": node.lineno,
                    })
                    break
    return found


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_facts():
    return {
        "_meta": {
            "generated_by": "scripts/extract_facts.py",
            "edit_source_not_this_file": True,
        },
        "public_exports": get_public_exports(),
        "rule_count": get_rule_count(),
        "taxonomy": get_taxonomy(),
        "test_count": get_test_count(),
        "test_files": get_test_files(),
        "detection_signals_in_scan": get_detection_signals(),
        "constants": get_constants(),
        "L16_detectors": get_l16_detectors(),
    }


def main():
    facts = build_facts()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        yaml.safe_dump(facts, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
