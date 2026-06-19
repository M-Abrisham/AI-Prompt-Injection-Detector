#!/usr/bin/env python3
"""Verify every numeric claim in README.md is grounded in docs/facts.yaml.

Stripping rules (numbers inside these are NOT claims and are ignored):
  - fenced code blocks ```...```
  - HTML comments  <!-- ... -->  (citation markers contain line numbers)
  - inline code   `...`
  - markdown image / link URLs (badge URLs contain version numbers)
  - layer identifiers  L0..L99, Layer 0..Layer 99
  - version tokens  v0.1, v1.2.3
  - section / list markers like "1.", "2." at line start

Anything left is treated as a quantitative claim. Each must be representable
by some value in docs/facts.yaml — bare or comma-formatted.

Exit code: 0 if all claims grounded, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
FACTS = REPO_ROOT / "docs" / "facts.yaml"
BENCHMARKS = REPO_ROOT / "docs" / "benchmarks.yaml"


def _collect_numeric(obj, allowed: set[str]) -> None:
    """Recursively add every numeric leaf of `obj` to `allowed` (bare + comma forms)."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        allowed.add(str(obj))
        if float(obj).is_integer():
            allowed.add(str(int(obj)))
            allowed.add(f"{int(obj):,}")
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_numeric(v, allowed)
    elif isinstance(obj, list):
        for v in obj:
            _collect_numeric(v, allowed)


def build_allowed_values(facts: dict) -> set[str]:
    """Flatten every numeric value in facts.yaml into a set of string forms."""
    allowed: set[str] = set()

    def add(v):
        if v is None or isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            allowed.add(str(v))
            if isinstance(v, float) and v.is_integer():
                allowed.add(str(int(v)))
            allowed.add(f"{int(v):,}" if isinstance(v, (int, float)) and float(v).is_integer() else str(v))
        elif isinstance(v, str):
            allowed.add(v)

    # Headline counts
    rc = facts.get("rule_count", {})
    add(rc.get("literal"))
    add(rc.get("total_ast"))
    for ext in rc.get("extends", []) or []:
        add(ext.get("count"))

    tax = facts.get("taxonomy", {})
    add(tax.get("category_count"))
    add(tax.get("technique_count_total"))
    for v in (tax.get("techniques_by_category") or {}).values():
        add(v)

    tc = facts.get("test_count", {})
    add(tc.get("count"))
    add(tc.get("collection_errors"))

    add(facts.get("test_files"))

    # Derived lengths from list facts
    add(len(facts.get("public_exports") or []))
    add(len(facts.get("L16_detectors") or []))
    add(len(((facts.get("detection_signals_in_scan") or {}).get("calls")) or []))

    for v in (facts.get("constants") or {}).values():
        add(v)

    return allowed


def _blank_preserving_newlines(match: "re.Match") -> str:
    """Replace match content with newlines so line numbers stay aligned."""
    return "\n" * match.group(0).count("\n")


def strip_for_claim_extraction(text: str) -> str:
    # Order matters: strip fenced code first. Preserve newlines in every
    # stripper so the line numbers in error messages match the README.
    text = re.sub(r"```.*?```", _blank_preserving_newlines, text, flags=re.DOTALL)
    text = re.sub(r"<!--.*?-->", _blank_preserving_newlines, text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)         # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)     # link → keep text
    text = re.sub(r"`[^`]*`", "", text)                       # inline code
    # Layer identifiers
    text = re.sub(r"\bL\d{1,3}\b", "", text)
    text = re.sub(r"\bLayer\s+\d{1,3}\b", "", text)
    # External-standard naming: OWASP Top 10, OWASP LLM Top 10, etc.
    text = re.sub(r"\bTop\s+\d+\b", "", text, flags=re.IGNORECASE)
    # Version tokens (v0.1, v1.2.3)
    text = re.sub(r"\bv\d+(?:\.\d+)*\b", "", text)
    # Numeric section bullets at line start: "1." "2." etc.
    text = re.sub(r"^\s*\d+\.\s", "", text, flags=re.MULTILINE)
    return text


def extract_numeric_tokens(prose: str) -> list[tuple[str, int]]:
    """Return (token, line_number) for every numeric token left in prose."""
    rows = []
    for lineno, line in enumerate(prose.splitlines(), start=1):
        for m in re.finditer(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b", line):
            rows.append((m.group(0), lineno))
    return rows


def main() -> int:
    facts = yaml.safe_load(FACTS.read_text())
    allowed = build_allowed_values(facts)

    # Benchmark numbers are grounded in docs/benchmarks.yaml (scripts/extract_benchmarks.py).
    if BENCHMARKS.exists():
        _collect_numeric(yaml.safe_load(BENCHMARKS.read_text()) or {}, allowed)

    raw = README.read_text()
    prose = strip_for_claim_extraction(raw)
    tokens = extract_numeric_tokens(prose)

    unverified: list[tuple[str, int]] = []
    for tok, lineno in tokens:
        if tok in allowed:
            continue
        # Comma-stripped form for "50,000" → "50000" check
        bare = tok.replace(",", "")
        if bare in allowed:
            continue
        unverified.append((tok, lineno))

    print(f"README.md numeric claims: {len(tokens)}")
    print(f"facts.yaml allowed values: {len(allowed)}")
    if unverified:
        print(f"\nUNVERIFIED ({len(unverified)}):")
        for tok, lineno in unverified:
            print(f"  README.md:{lineno}  '{tok}'  not in docs/facts.yaml")
        print("\nFix: either (a) add an extractor to scripts/extract_facts.py,")
        print("(b) move the number into a code block / inline code, or")
        print("(c) remove it from the README.")
        return 1

    print("\nAll numeric claims grounded in docs/facts.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
