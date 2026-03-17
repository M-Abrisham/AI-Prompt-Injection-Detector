After ``pip install na0s``, the ``na0s`` command is available::

    na0s scan "Ignore all previous instructions"
    na0s scan --file suspicious.txt
    na0s scan --jsonl batch.jsonl
    echo "payload" | na0s scan -
    na0s scan-output "Sure! Here is the system prompt..."
    na0s version

Exit codes:
    0 = safe / clean
    1 = malicious / detected
    2 = blocked / error
    3 = usage error (bad arguments / bad input)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import sys
from dataclasses import asdict

from na0s._version import __version__

# Maximum input size (10 MB) to prevent OOM on adversarial input
_MAX_INPUT_BYTES = 10 * 1024 * 1024

# ── Exit codes (standard security-tool convention) ─────────────────
EXIT_CLEAN = 0       # No injection detected
EXIT_SAFE = 0        # Alias for backward compatibility
EXIT_DETECTED = 1    # Injection detected (CI/CD pipelines fail on this)
EXIT_MALICIOUS = 1   # Alias for backward compatibility
EXIT_ERROR = 2       # Runtime / configuration error
EXIT_BLOCKED = 2     # Alias for backward compatibility
EXIT_BAD_INPUT = 3   # Invalid input (empty, unsupported file, etc.)
EXIT_USAGE = 3       # Alias for backward compatibility


# ── Output helpers ─────────────────────────────────────────────────

def _print_scan_result(result, fmt, verbose):
    """Render a ScanResult for the terminal."""
    if fmt == "json":
        data = asdict(result)
        # Remove bulky sanitized_text from JSON unless verbose
        if not verbose:
            data.pop("sanitized_text", None)
        print(json.dumps(data, indent=2, default=str))
        return

    icon = "X" if result.is_malicious else "OK"
    print(f"[{icon}] {result.label.upper()}  (risk_score: {result.risk_score:.2f})")
    if result.technique_tags:
        print(f"  Techniques: {', '.join(result.technique_tags)}")
    if result.rule_hits and verbose:
        print(f"  Rule hits:  {len(result.rule_hits)}")
    if verbose:
        print(f"  ML conf:    {result.ml_confidence:.4f} ({result.ml_label})")
        if result.anomaly_flags:
            print(f"  Anomalies:  {', '.join(result.anomaly_flags)}")
        if result.cascade_stage:
            print(f"  Stage:      {result.cascade_stage}")


def _print_output_result(result, fmt, verbose):
    """Render an OutputScanResult for the terminal."""
    if fmt == "json":
        data = asdict(result)
        if not verbose:
            data.pop("redacted_text", None)
        print(json.dumps(data, indent=2, default=str))
        return

    icon = "X" if result.is_suspicious else "OK"
    tag = "SUSPICIOUS" if result.is_suspicious else "CLEAN"
    print(f"[{icon}] {tag}  (risk_score: {result.risk_score:.2f})")
    if result.flags:
        print(f"  Flags: {', '.join(result.flags)}")


def _format_text(result):
    """Format a ScanResult as human-readable text."""
    lines = [
        "Label: {}".format(result.label),
        "Risk Score: {}".format(result.risk_score),
        "Technique Tags: {}".format(", ".join(result.technique_tags) if result.technique_tags else "(none)"),
        "Rule Hits: {}".format(", ".join(result.rule_hits) if result.rule_hits else "(none)"),
        "Latency: {}ms".format(result.elapsed_ms),
    ]
    return "\n".join(lines)


def _format_csv(result):
    """Format a ScanResult as a single CSV line (with header)."""
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["label", "risk_score", "is_malicious", "technique_tags"])
    writer.writerow([
        result.label,
        result.risk_score,
        result.is_malicious,
        ",".join(result.technique_tags) if result.technique_tags else "",
    ])
    return buf.getvalue().rstrip("\n")


# ── Input resolution ───────────────────────────────────────────────

def _resolve_text(text_arg, file_arg):
    """Return input text from positional arg, --file, or stdin.

    Enforces a ``_MAX_INPUT_BYTES`` cap to prevent OOM on adversarial input.
    """
    if text_arg == "-" or (text_arg is None and file_arg is None
                           and not sys.stdin.isatty()):
        data = sys.stdin.buffer.read(_MAX_INPUT_BYTES + 1)
        if len(data) > _MAX_INPUT_BYTES:
            print(
                f"Error: input exceeds {_MAX_INPUT_BYTES // (1024 * 1024)} MB limit.",
                file=sys.stderr,
            )
            return None
        return data.decode("utf-8", errors="replace")
    if file_arg:
        import os
        try:
            size = os.path.getsize(file_arg)
        except OSError:
            size = 0
        if size > _MAX_INPUT_BYTES:
            print(
                f"Error: file exceeds {_MAX_INPUT_BYTES // (1024 * 1024)} MB limit.",
                file=sys.stderr,
            )
            return None
        with open(file_arg, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    return text_arg


def _exit_code_for(result):
    """Return the appropriate exit code for a ScanResult."""
    if result.label == "blocked":
        return EXIT_BLOCKED
    if result.is_malicious:
        return EXIT_MALICIOUS
    return EXIT_SAFE


# ── Subcommands ────────────────────────────────────────────────────

def _cmd_scan(args):
    text = _resolve_text(args.text, getattr(args, 'file', None))
    if not text or not text.strip():
        print("Error: empty input. Provide text, --file, or pipe to stdin.",
              file=sys.stderr)
        return EXIT_BAD_INPUT

    try:
        from na0s import scan
        result = scan(text)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    fmt = "json" if args.json else "text"
    _print_scan_result(result, fmt, args.verbose)
    return EXIT_DETECTED if result.is_malicious else EXIT_CLEAN


def _cmd_scan_output(args):
    text = _resolve_text(args.text, getattr(args, 'file', None))
    if not text or not text.strip():
        print("Error: empty input.", file=sys.stderr)
        return EXIT_BAD_INPUT

    try:
        from na0s import scan_output
        result = scan_output(
            text,
            original_prompt=args.original_prompt,
            system_prompt=args.system_prompt,
            sensitivity=args.sensitivity,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    fmt = "json" if args.json else "text"
    _print_output_result(result, fmt, args.verbose)
    return EXIT_DETECTED if result.is_suspicious else EXIT_CLEAN


def _scan_jsonl(jsonl_path, threshold):
    """Scan each line of a JSONL file. Returns the worst exit code seen."""
    from na0s.predict import scan

    worst_exit = EXIT_SAFE
    try:
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    err = {"error": "Invalid JSON on line {}".format(line_no), "detail": str(exc)}
                    print(json.dumps(err))
                    worst_exit = max(worst_exit, EXIT_BLOCKED)
                    continue

                text = record.get("text", "")
                if not text:
                    err = {"error": "Missing 'text' field on line {}".format(line_no)}
                    print(json.dumps(err))
                    worst_exit = max(worst_exit, EXIT_BLOCKED)
                    continue

                try:
                    result = scan(text, threshold=threshold)
                    print(result.to_json())
                    worst_exit = max(worst_exit, _exit_code_for(result))
                except Exception as exc:
                    err = {"error": "Scan failed on line {}".format(line_no), "detail": str(exc)}
                    print(json.dumps(err))
                    worst_exit = max(worst_exit, EXIT_BLOCKED)

    except FileNotFoundError:
        print("Error: file not found: {}".format(jsonl_path), file=sys.stderr)
        return EXIT_USAGE
    except OSError as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        return EXIT_BLOCKED

    return worst_exit


def _cmd_version(args):
    info = {"version": __version__, "extras": []}

    for name, mod in [("embedding", "sentence_transformers"),
                      ("ocr", "easyocr"), ("docs", "pymupdf"),
                      ("llm", "openai"), ("lang", "langdetect")]:
        if importlib.util.find_spec(mod) is not None:
            info["extras"].append(name)

    if getattr(args, "json", False):
        print(json.dumps(info, indent=2))
    else:
        print(f"na0s {__version__}")
        if info["extras"]:
            print(f"Installed extras: {', '.join(info['extras'])}")
        else:
            print("No optional extras installed.")
    return EXIT_CLEAN


# ── Main parser ────────────────────────────────────────────────────

def main(argv=None):
    """Entry point for the ``na0s`` CLI."""
    parser = argparse.ArgumentParser(
        prog="na0s",
        description="Na0S -- Multi-layer AI prompt injection detector",
    )
    parser.add_argument(
        "--version", action="version", version=f"na0s {__version__}",
    )
    sub = parser.add_subparsers(dest="command")

    # ── scan ───────────────────────────────────────────────────────
    p_scan = sub.add_parser(
        "scan", help="Scan input text for prompt injection",
    )
    p_scan.add_argument(
        "text", nargs="?", default=None,
        help="Text to scan (use - for stdin, omit to read stdin if piped)",
    )
    p_scan.add_argument(
        "-f", "--file", default=None,
        help="Path to a text file to scan",
    )
    p_scan.add_argument(
        "--jsonl",
        default=None,
        help="Path to a JSONL file for batch scanning",
    )
    p_scan.add_argument("--json", action="store_true", help="JSON output")
    p_scan.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show ML confidence, anomaly flags, rule hits",
    )
    p_scan.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Decision threshold for the composite score (default: 0.55)",
    )
    p_scan.set_defaults(func=_cmd_scan)

    # ── scan-output ────────────────────────────────────────────────
    p_out = sub.add_parser(
        "scan-output", help="Scan LLM output for injection success signs",
    )
    p_out.add_argument("text", nargs="?", default=None)
    p_out.add_argument("-f", "--file", default=None)
    p_out.add_argument("--json", action="store_true")
    p_out.add_argument("-v", "--verbose", action="store_true")
    p_out.add_argument("--system-prompt", default=None,
                       help="System prompt (for leak detection)")
    p_out.add_argument("--original-prompt", default=None,
                       help="Original user prompt (for echo detection)")
    p_out.add_argument("--sensitivity", default="medium",
                       choices=["low", "medium", "high"])
    p_out.set_defaults(func=_cmd_scan_output)

    # ── version ────────────────────────────────────────────────────
    p_ver = sub.add_parser("version", help="Show version and installed extras")
    p_ver.add_argument("--json", action="store_true")
    p_ver.set_defaults(func=_cmd_version)

    # ── parse & dispatch ───────────────────────────────────────────
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_ERROR

    # Handle JSONL batch mode for scan command
    if args.command == "scan" and getattr(args, 'jsonl', None) is not None:
        return _scan_jsonl(args.jsonl, getattr(args, 'threshold', 0.55))

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
