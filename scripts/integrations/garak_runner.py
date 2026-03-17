"""Na0S Garak integration — runs Garak probes against the Na0S detector.

Install: pip install garak  (requires Python <=3.12)
Usage:   python scripts/integrations/garak_runner.py --probes all
"""

import argparse
import datetime
import json
import os
import sys

try:
    import garak
    from garak.generators.base import Generator
    GARAK_AVAILABLE = True
    GARAK_VERSION = getattr(garak, "__version__", "unknown")
except ImportError:
    GARAK_AVAILABLE = False
    GARAK_VERSION = None


class NaOSGarakTarget:
    """Wraps na0s.predict() as a Garak-compatible target.

    Garak sends adversarial prompts; this target runs them through
    Na0S's detection pipeline and returns the verdict as a response.
    """

    def __init__(self):
        # Lazy import to avoid loading models when garak isn't available
        self._predict = None

    def _load(self):
        if self._predict is None:
            from na0s.predict import scan
            self._predict = scan

    def generate(self, prompt: str) -> list:
        """Generate a response for the given prompt.

        Returns a list of strings (Garak convention).
        The response encodes Na0S's detection verdict.
        """
        self._load()
        result = self._predict(prompt)
        label = getattr(result, "label", str(result))
        confidence = getattr(result, "confidence", 0.0)
        return [f"INJECTION_DETECTED={label} confidence={confidence:.3f}"]


def run_garak_suite(probe_categories=None, output_path=None):
    """Run Garak probe suite against NaOSGarakTarget.

    Parameters
    ----------
    probe_categories : list[str] or None
        Probe categories to run. None or ["all"] runs all available.
    output_path : str or None
        Path to write results JSON. Default: data/evaluation/garak_report.json

    Returns
    -------
    dict
        Results dict with probes_run, results, and summary.
    """
    if not GARAK_AVAILABLE:
        print("Garak not installed. Run: pip install garak (requires Python <=3.12)")
        sys.exit(0)

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "evaluation", "garak_report.json"
        )

    target = NaOSGarakTarget()

    # Discover available probes
    from garak import _plugins
    available_probes = _plugins.enumerate_plugins("probes")

    if probe_categories and "all" not in probe_categories:
        available_probes = [
            p for p in available_probes
            if any(cat.lower() in p.lower() for cat in probe_categories)
        ]

    results = []
    total_attempts = 0
    total_hits = 0

    for probe_name in available_probes:
        try:
            probe_module = _plugins.load_plugin(probe_name)
            probe_instance = probe_module()

            attempts = 0
            hits = 0
            examples = []

            for test_case in probe_instance.generate():
                prompt_text = str(test_case)
                response = target.generate(prompt_text)
                attempts += 1

                # A "hit" means Na0S failed to detect the injection
                if "SAFE" in response[0]:
                    hits += 1
                    examples.append({
                        "prompt": prompt_text[:200],
                        "response": response[0],
                    })

            results.append({
                "probe": probe_name,
                "attempts": attempts,
                "hits": hits,
                "hit_rate": hits / max(attempts, 1),
                "examples": examples[:5],
            })
            total_attempts += attempts
            total_hits += hits

        except Exception as e:
            results.append({
                "probe": probe_name,
                "attempts": 0,
                "hits": 0,
                "hit_rate": 0.0,
                "examples": [],
                "error": str(e),
            })

    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "garak_version": GARAK_VERSION,
        "probes_run": [r["probe"] for r in results],
        "results": results,
        "summary": {
            "total_attempts": total_attempts,
            "total_hits": total_hits,
            "overall_hit_rate": total_hits / max(total_attempts, 1),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Garak report written to {output_path}")
    print(f"Summary: {total_attempts} attempts, {total_hits} hits, "
          f"hit rate: {total_hits / max(total_attempts, 1):.1%}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run Garak vulnerability probes against Na0S"
    )
    parser.add_argument(
        "--probes", default="all",
        help="Probe categories (comma-separated) or 'all' (default: all)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: data/evaluation/garak_report.json)"
    )
    args = parser.parse_args()

    categories = None if args.probes == "all" else args.probes.split(",")
    run_garak_suite(probe_categories=categories, output_path=args.output)


if __name__ == "__main__":
    main()
