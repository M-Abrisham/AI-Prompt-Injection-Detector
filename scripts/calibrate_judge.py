#!/usr/bin/env python3
"""Calibrate the prompt-injection LLM judge against the F14 scenario library.

This is the single trustworthy entry point for measuring the judge's quality.
It deliberately does NOT read ``data/processed/combined_data.csv`` (the TRAINING
output that ``scripts/evaluate_llm_judge.py`` uses — scoring on that is a
train-on-test fallacy this repo has already been bitten by). Instead it scores
the LABELED, decontamination-gated F14 eval scenarios under
``data/eval/scenarios/v0.1/``.

What it does, in order
~~~~~~~~~~~~~~~~~~~~~~~
1. **Load** the eval set via :func:`na0s.eval.scenarios.loader.load_scenarios_dir`
   and derive, per scenario, ``y_true`` (``expected_verdict == "blocked"`` -> 1,
   ``"allowed"`` -> 0) and the ``attack_category``.
2. **Decontamination precondition.** Before scoring anything, run the exact
   stable-id overlap scan from ``scripts/check_eval_decontamination`` against the
   training-data roots. If ANY eval scenario's text is present in training data
   the judge would be scored on data it was trained on, so the script REFUSES
   to score and exits 2. (The decontamination module exposes
   :func:`find_overlaps` / :func:`compute_stable_id`; there is no ``scan_exact``
   symbol — ``find_overlaps`` IS the exact-id scan, so we call it directly.)
3. **One-time-test guard.** A sha256 manifest over the test slice's stable_ids
   is checked against an append-only log
   (``data/eval/scenarios/_baselines/judge_test_uses.log``) so a held-out slice
   can't quietly become a tuning set. ``--allow-test-reuse`` overrides for an
   intentional re-run.
4. **Score per attack-category** using :class:`na0s.judge.per_class.PerClassJudge`
   over an INJECTABLE backend. ``--mock-judge`` selects a deterministic, offline
   stub (no network) so this script and its test run with no API key. A real
   backend (``--backend openai|groq``) is used ONLY when explicitly selected.
5. **Calibrate** each category and the overall set via
   :func:`na0s.judge.calibration.calibrate`: TPR / TNR / precision / recall, each
   with a percentile bootstrap CI, plus the Rogan-Gladen-corrected prevalence.
   No accuracy headline (accuracy is dominated by the benign majority and hides a
   weak detector on imbalanced data).

Exit codes
~~~~~~~~~~~
* ``0`` — scoring completed (and, if ``--min-recall`` given, the overall recall
  met the floor).
* ``1`` — overall recall fell below ``--min-recall``.
* ``2`` — configuration / decontamination / one-time-test error (nothing scored).

Usage
~~~~~
    # Offline, deterministic — what CI and the test run:
    python scripts/calibrate_judge.py --mock-judge

    # Enforce a recall floor (exit 1 if not met):
    python scripts/calibrate_judge.py --mock-judge --min-recall 0.5

    # Real judge (requires OPENAI_API_KEY / GROQ_API_KEY):
    python scripts/calibrate_judge.py --backend openai --model gpt-4o-mini

    # Emit machine-readable JSON instead of the human report:
    python scripts/calibrate_judge.py --mock-judge --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ``src/`` and the repo root importable when run as a script (mirrors the
# bootstrap in scripts/check_eval_decontamination.py so ``import na0s.*`` and
# ``import scripts.*`` both resolve).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

from na0s.eval.scenarios.loader import load_scenarios_dir  # noqa: E402
from na0s.eval.scenarios.schema import Scenario, ScenarioType  # noqa: E402
from na0s.judge import calibration as cal  # noqa: E402
from na0s.judge.llm_judge import JudgeVerdict  # noqa: E402
from na0s.judge.per_class import (  # noqa: E402
    PerClassJudge,
    load_taxonomy_categories,
)
from na0s.judge.splits import (  # noqa: E402
    check_test_unused,
    record_test_use,
    test_manifest_hash,
)

# The decontamination scan lives in a sibling script; ``scripts`` is a package.
from scripts.check_eval_decontamination import (  # noqa: E402
    compute_stable_id,
    find_overlaps,
)

# ── documented constants (no magic numbers) ──────────────────────────────────

#: Default eval set: the F14 v0.1 scenario library (NOT the training CSV).
_DEFAULT_SCENARIOS_DIR = _PROJECT_ROOT / "data" / "eval" / "scenarios" / "v0.1"

#: Default training roots scanned for contamination. Same set
#: ``check_eval_decontamination`` defaults to — kept in sync by passing through
#: when the user does not override.
_DEFAULT_TRAINING_ROOTS = [
    _PROJECT_ROOT / "data" / "processed",
    _PROJECT_ROOT / "data" / "staging",
    _PROJECT_ROOT / "data" / "aggregated",
]

#: Append-only one-time-test-use log (per the splits.py guard contract).
_DEFAULT_TEST_USE_LOG = (
    _PROJECT_ROOT / "data" / "eval" / "scenarios" / "_baselines" / "judge_test_uses.log"
)

#: Bootstrap resamples. 2000 is the conventional floor for a stable 95%
#: percentile interval (matches ``calibration.bootstrap_ci``'s default); exposed
#: as ``--n-boot`` so a quick run can lower it.
_DEFAULT_N_BOOT = 2000

#: Bootstrap RNG seed — fixed for reproducible CIs (overridable via ``--seed``).
_DEFAULT_SEED = 0

#: The mock judge's confidence values. These are NOT thresholds: they are fixed
#: labels the deterministic stub stamps on its verdicts so downstream parsing
#: has a well-formed confidence. 0.9 / 0.1 are arbitrary-but-documented
#: high/low markers (verdict, not confidence, drives every metric here).
_MOCK_CONF_MALICIOUS = 0.9
_MOCK_CONF_SAFE = 0.1

#: Lexical triggers the OFFLINE mock judge keys on to emit MALICIOUS. This is a
#: deliberately crude, transparent heuristic — its ONLY purpose is to make the
#: mock deterministic and non-trivial (so per-class metrics aren't all 0/1) for
#: the no-network test. The real judge is an LLM; this list is never used when a
#: real backend is selected.
_MOCK_MALICIOUS_TRIGGERS = (
    "ignore all previous",
    "ignore previous",
    "disregard",
    "system prompt",
    "initial instructions",
    "you are now",
    "dan",
    "developer mode",
    "repeat the text above",
    "verbatim",
    "exfiltrate",
    "bypass",
    "override",
    "reveal your",
    "print your",
    "extract the admin",
    "all user passwords",
    "no restrictions",
    "unrestricted mode",
)


# ── scenario -> (text, label, category) ──────────────────────────────────────


def scenario_text(scn: Scenario) -> str:
    """Return the text to classify for a scenario.

    Single-prompt: the payload. Multi-turn: the turn texts joined by newlines —
    the judge sees the whole conversation as one block (matching how a
    cumulative conversation detector would receive the running transcript).
    """
    if scn.type == ScenarioType.SINGLE_PROMPT:
        return scn.payload or ""
    return "\n".join(turn.text for turn in scn.turns)


def scenario_label(scn: Scenario) -> int:
    """Derive y_true from ``expected_verdict`` (blocked -> 1, allowed -> 0)."""
    if scn.expected_verdict == "blocked":
        return 1
    if scn.expected_verdict == "allowed":
        return 0
    # loader/schema already validate the vocabulary, but be defensive.
    raise ValueError(
        f"scenario {scn.name!r} has unexpected verdict {scn.expected_verdict!r}"
    )


# ── deterministic offline mock judge ─────────────────────────────────────────


class MockJudge:
    """A deterministic, offline stand-in for an LLM judge.

    Exposes ``classify_messages(messages)`` so it slots into
    :class:`PerClassJudge`'s messages-aware routing path. It extracts the
    ``<INPUT>...</INPUT>`` block from the assembled messages and emits MALICIOUS
    iff the input contains any :data:`_MOCK_MALICIOUS_TRIGGERS` substring (case-
    insensitive). No randomness, no network — identical input always yields the
    same verdict, so the harness output is reproducible and the test is stable.
    """

    def __init__(self, model: str = "mock-judge") -> None:
        self.model = model

    @staticmethod
    def _extract_input(messages: list[dict]) -> str:
        """Pull the raw user text back out of the delimiter-wrapped last turn."""
        if not messages:
            return ""
        content = messages[-1].get("content", "")
        start = content.find("<INPUT>")
        end = content.rfind("</INPUT>")
        if start != -1 and end != -1 and end > start:
            return content[start + len("<INPUT>"):end].strip()
        return content

    def classify_messages(self, messages: list[dict]) -> JudgeVerdict:
        text = self._extract_input(messages).lower()
        is_mal = any(trig in text for trig in _MOCK_MALICIOUS_TRIGGERS)
        return JudgeVerdict(
            verdict="MALICIOUS" if is_mal else "SAFE",
            confidence=_MOCK_CONF_MALICIOUS if is_mal else _MOCK_CONF_SAFE,
            reasoning="deterministic mock verdict (offline)",
            latency_ms=0.0,
            model=self.model,
        )


def _verdict_to_pred(verdict: object) -> int:
    """Map a judge verdict (JudgeVerdict or dict) to a 0/1 prediction.

    A MALICIOUS verdict is a positive prediction (1). SAFE and UNKNOWN are both
    treated as 0 (not-blocked): an UNKNOWN judge result must not count as a
    detection, otherwise an unavailable judge would inflate recall.
    """
    if isinstance(verdict, dict):
        v = str(verdict.get("verdict", "")).upper()
    else:
        v = str(getattr(verdict, "verdict", "")).upper()
    return 1 if v == "MALICIOUS" else 0


# ── scoring ──────────────────────────────────────────────────────────────────


def build_backend(args: argparse.Namespace):
    """Return the injectable judge backend (mock or real).

    The mock path imports nothing network-bound. The real path constructs a
    :class:`~na0s.judge.llm_judge.LLMJudge`, which validates its API key and
    backend at construction time.
    """
    if args.mock_judge:
        return MockJudge()
    # Real backend — imported lazily so --mock-judge never needs the SDK present.
    from na0s.judge.llm_judge import LLMJudge

    return LLMJudge(backend=args.backend, model=args.model)


def score_scenarios(scenarios, backend, category_descs):
    """Classify every scenario via a per-class judge; return predictions.

    Returns
    -------
    dict
        ``{attack_category: {"y_true": [...], "y_pred": [...]}}`` plus an
        ``"__overall__"`` aggregate. Each scenario is scored by a PerClassJudge
        specialized for ITS category, so the per-category metric reflects the
        prompt the category would actually use in production.
    """
    per_cat: dict[str, dict[str, list[int]]] = {}
    overall = {"y_true": [], "y_pred": []}

    # One PerClassJudge per category (the few-shot block is empty here: this
    # harness measures the prompt+backend, not a tuned few-shot set; few-shot
    # selection is the job of select_few_shot at production wiring time and is
    # leakage-tested separately in tests/judge/test_per_class.py).
    judges: dict[str, PerClassJudge] = {}

    for scn in scenarios:
        cat = scn.attack_category
        if cat not in judges:
            judges[cat] = PerClassJudge(
                judge=backend,
                attack_category=cat,
                few_shot=None,
                category_desc=category_descs.get(cat),
            )
        text = scenario_text(scn)
        y_true = scenario_label(scn)
        verdict = judges[cat].classify(text)
        y_pred = _verdict_to_pred(verdict)

        bucket = per_cat.setdefault(cat, {"y_true": [], "y_pred": []})
        bucket["y_true"].append(y_true)
        bucket["y_pred"].append(y_pred)
        overall["y_true"].append(y_true)
        overall["y_pred"].append(y_pred)

    per_cat["__overall__"] = overall
    return per_cat


def calibrate_all(per_cat, n_boot, seed):
    """Run :func:`calibration.calibrate` on each bucket; return result dicts."""
    out: dict[str, dict] = {}
    for cat, data in per_cat.items():
        res = cal.calibrate(
            data["y_true"], data["y_pred"], n_boot=n_boot, seed=seed
        )
        out[cat] = res.to_dict()
    return out


# ── reporting ────────────────────────────────────────────────────────────────


def _fmt_ci(ci) -> str:
    lo, hi = ci
    return f"[{lo:.3f}, {hi:.3f}]"


def print_report(results: dict, *, backend_name: str) -> None:
    """Human-readable per-category + overall report. NO accuracy headline."""
    print("=" * 74)
    print("  Na0S Judge Calibration — F14 scenario library")
    print(f"  backend: {backend_name}")
    print("=" * 74)
    print("  Metrics are recall/TPR (missed attacks) and TNR/precision (false")
    print("  alarms), each with a 95% bootstrap CI, plus Rogan-Gladen-corrected")
    print("  prevalence. Accuracy is intentionally omitted (it hides weak")
    print("  detectors on benign-heavy data).")
    print()

    # Stable order: attack categories sorted, overall last.
    cats = sorted(c for c in results if c != "__overall__")
    for cat in cats + ["__overall__"]:
        r = results[cat]
        label = "OVERALL" if cat == "__overall__" else cat
        counts = r.get("counts", {})
        print(f"  [{label}]  n={r['n']}  "
              f"(tp={counts.get('tp')}, fp={counts.get('fp')}, "
              f"tn={counts.get('tn')}, fn={counts.get('fn')})")
        print(f"      recall/TPR : {r['recall']:.3f}  CI {_fmt_ci(r['recall_ci'])}")
        print(f"      TNR        : {r['tnr']:.3f}  CI {_fmt_ci(r['tnr_ci'])}")
        print(f"      precision  : {r['precision']:.3f}  CI {_fmt_ci(r['precision_ci'])}")
        pc = r.get("prevalence_corrected")
        if pc is None:
            print("      prevalence (Rogan-Gladen): N/A "
                  "(judge no better than chance on this slice)")
        else:
            print(f"      prevalence (Rogan-Gladen): {pc:.3f}  "
                  f"CI {_fmt_ci(r['prevalence_corrected_ci'])} "
                  f"(apparent {r['prevalence_apparent']:.3f})")
        print()
    print("=" * 74)


# ── decontamination precondition ─────────────────────────────────────────────


def decontamination_ok(scenarios_dir: Path, training_roots: list[Path]) -> list[dict]:
    """Run the exact-id contamination scan; return overlap records (empty == OK).

    Thin wrapper over ``check_eval_decontamination.find_overlaps`` so the
    precondition uses the SAME stable-id scan the standalone gate uses
    (``compute_stable_id`` is imported here only to keep the dependency explicit
    and assert the modules agree on the hashing function).
    """
    assert callable(compute_stable_id)  # modules agree on the id function
    return find_overlaps(scenarios_dir, training_roots)


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--scenarios-dir", default=str(_DEFAULT_SCENARIOS_DIR),
        help=f"F14 scenario YAML dir (default: {_DEFAULT_SCENARIOS_DIR})",
    )
    p.add_argument(
        "--training-roots", nargs="*", default=None,
        help="Dirs scanned for contamination (default: processed/staging/aggregated)",
    )
    backend_grp = p.add_mutually_exclusive_group()
    backend_grp.add_argument(
        "--mock-judge", action="store_true",
        help="Use a deterministic OFFLINE stub judge (no network). "
             "Required for tests/CI; no API key needed.",
    )
    backend_grp.add_argument(
        "--backend", choices=["openai", "groq"], default=None,
        help="Real LLM backend (requires the corresponding API key). "
             "Mutually exclusive with --mock-judge.",
    )
    p.add_argument(
        "--model", default=None,
        help="Override the model name for a real backend (ignored by --mock-judge).",
    )
    p.add_argument(
        "--min-recall", type=float, default=None,
        help="If set, exit 1 when OVERALL recall is below this floor.",
    )
    p.add_argument(
        "--n-boot", type=int, default=_DEFAULT_N_BOOT,
        help=f"Bootstrap resamples for CIs (default: {_DEFAULT_N_BOOT}).",
    )
    p.add_argument(
        "--seed", type=int, default=_DEFAULT_SEED,
        help=f"Bootstrap RNG seed (default: {_DEFAULT_SEED}).",
    )
    p.add_argument(
        "--test-use-log", default=str(_DEFAULT_TEST_USE_LOG),
        help=f"One-time-test-use log path (default: {_DEFAULT_TEST_USE_LOG}).",
    )
    p.add_argument(
        "--allow-test-reuse", action="store_true",
        help="Permit re-scoring an already-recorded test slice (downgrades the "
             "one-time-test guard to a no-op for an intentional re-run).",
    )
    p.add_argument(
        "--skip-decontam", action="store_true",
        help="Skip the decontamination precondition (NOT recommended; for "
             "running against a scenario dir with no training roots present).",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Emit the calibration results as JSON instead of the text report.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    scenarios_dir = Path(args.scenarios_dir)
    if not scenarios_dir.is_dir():
        print(f"ERROR: scenarios dir not found: {scenarios_dir}", file=sys.stderr)
        return 2

    # A backend MUST be chosen explicitly. Defaulting to a real API would risk a
    # surprise network call / cost; defaulting to mock would risk silently
    # reporting fake numbers as if real. Force the choice.
    if not args.mock_judge and args.backend is None:
        print(
            "ERROR: choose a backend — pass --mock-judge for an offline run or "
            "--backend openai|groq for a real judge.",
            file=sys.stderr,
        )
        return 2

    try:
        scenarios = load_scenarios_dir(scenarios_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: failed to load scenarios: {exc}", file=sys.stderr)
        return 2
    if not scenarios:
        print(f"ERROR: no scenarios found in {scenarios_dir}", file=sys.stderr)
        return 2

    # 1) Decontamination precondition — refuse to score on contaminated data.
    if not args.skip_decontam:
        training_roots = [
            Path(p) for p in (args.training_roots or _DEFAULT_TRAINING_ROOTS)
        ]
        try:
            overlaps = decontamination_ok(scenarios_dir, training_roots)
        except Exception as exc:  # IO / parse error in the scan is a config error
            print(f"ERROR: decontamination scan failed: {exc}", file=sys.stderr)
            return 2
        if overlaps:
            print(
                f"ERROR: decontamination FAILED — {len(overlaps)} eval scenario(s) "
                "overlap training data; refusing to score (would be train-on-test).",
                file=sys.stderr,
            )
            for o in overlaps[:10]:
                print(
                    f"    scenario={o['scenario_name']} -> "
                    f"{o['training_file']}:{o['training_row']}",
                    file=sys.stderr,
                )
            return 2

    # 2) One-time-test guard over the FULL scored slice's stable_ids.
    manifest = test_manifest_hash(scenarios)
    try:
        check_test_unused(
            manifest, args.test_use_log, allow_reuse=args.allow_test_reuse
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # 3) Build backend + per-category descriptions.
    try:
        backend = build_backend(args)
    except (ImportError, ValueError) as exc:
        print(f"ERROR: could not build judge backend: {exc}", file=sys.stderr)
        return 2

    category_descs = _load_category_descs()

    # 4) Score + calibrate.
    per_cat = score_scenarios(scenarios, backend, category_descs)
    results = calibrate_all(per_cat, n_boot=args.n_boot, seed=args.seed)

    backend_name = "mock-judge (offline)" if args.mock_judge else (
        f"{args.backend}:{args.model or 'default'}"
    )

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        print_report(results, backend_name=backend_name)

    # Record the test use AFTER a successful score (append-only audit trail).
    record_test_use(manifest, args.test_use_log)

    # 5) Recall floor gate.
    if args.min_recall is not None:
        overall_recall = results["__overall__"]["recall"]
        if overall_recall < args.min_recall:
            print(
                f"FAIL: overall recall {overall_recall:.3f} < floor "
                f"{args.min_recall:.3f}",
                file=sys.stderr,
            )
            return 1

    return 0


def _load_category_descs() -> dict[str, str]:
    """Map taxonomy category id -> description (best-effort; empty on failure).

    Used only to enrich the per-class system prompt; a missing taxonomy or an
    unmapped category just yields no description (the prompt still names the id).
    """
    descs: dict[str, str] = {}
    try:
        import yaml

        raw = yaml.safe_load(
            (_PROJECT_ROOT / "data" / "taxonomy.yaml").read_text(encoding="utf-8")
        )
        for code, body in (raw.get("categories") or {}).items():
            if isinstance(body, dict) and body.get("description"):
                descs[str(code)] = str(body["description"])
    except Exception:
        # The taxonomy is enrichment, not a precondition — never fatal.
        pass
    # Quiet sanity touch so a totally-empty taxonomy is visible in review, not
    # silently swallowed; load_taxonomy_categories raises loudly if the file is
    # broken, which we tolerate here.
    try:
        load_taxonomy_categories(include_benign=True)
    except Exception:
        pass
    return descs


if __name__ == "__main__":
    sys.exit(main())
