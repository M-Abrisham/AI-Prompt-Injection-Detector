"""Na0S PyRIT integration — runs Microsoft red-teaming campaigns against Na0S.

Install: pip install pyrit-ai  (requires Python <=3.12)
Usage:   python scripts/integrations/pyrit_runner.py --strategy crescendo
"""

import argparse
import asyncio
import datetime
import json
import os
import sys

try:
    from pyrit.orchestrator import RedTeamingOrchestrator
    from pyrit.prompt_target import PromptTarget
    PYRIT_AVAILABLE = True
except ImportError:
    PYRIT_AVAILABLE = False

SUPPORTED_STRATEGIES = ["crescendo", "skeleton_key", "prompt_injection"]


class NaOSPromptTarget:
    """Wraps na0s.predict() as a PyRIT PromptTarget.

    PyRIT sends adversarial prompts; this target runs them through
    Na0S's detection pipeline and returns the verdict.
    """

    def __init__(self):
        self._predict = None

    def _load(self):
        if self._predict is None:
            from na0s.predict import scan
            self._predict = scan

    async def send_prompt_async(self, prompt: str) -> str:
        """Process a prompt through Na0S and return the verdict.

        PyRIT calls this method during red-teaming campaigns.
        """
        self._load()
        result = self._predict(prompt)
        label = getattr(result, "label", str(result))
        confidence = getattr(result, "confidence", 0.0)
        return f"label={label} confidence={confidence:.3f}"


def run_redteam_campaign(strategy, max_turns=10, output_path=None):
    """Run a PyRIT red-teaming campaign against Na0S.

    Parameters
    ----------
    strategy : str
        One of: crescendo, skeleton_key, prompt_injection
    max_turns : int
        Maximum conversation turns per campaign.
    output_path : str or None
        Path to write results JSON.

    Returns
    -------
    dict
        Campaign results.
    """
    if not PYRIT_AVAILABLE:
        print("PyRIT not installed. Run: pip install pyrit-ai (requires Python <=3.12)")
        sys.exit(0)

    if strategy not in SUPPORTED_STRATEGIES:
        print(f"Unknown strategy: {strategy}. Choose from: {SUPPORTED_STRATEGIES}")
        sys.exit(1)

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "evaluation", "pyrit_report.json"
        )

    target = NaOSPromptTarget()

    # Strategy-specific orchestration
    examples = []
    successful_jailbreaks = 0
    total_attempts = 0

    async def _run_campaign():
        nonlocal successful_jailbreaks, total_attempts

        orchestrator = RedTeamingOrchestrator(
            attack_strategy=strategy,
            prompt_target=target,
            max_turns=max_turns,
        )

        result = await orchestrator.execute_async()

        for turn in result.turns:
            total_attempts += 1
            jailbroken = "SAFE" in turn.response
            if jailbroken:
                successful_jailbreaks += 1
            examples.append({
                "turn": turn.turn_number,
                "prompt": turn.prompt[:200],
                "response": turn.response,
                "jailbroken": jailbroken,
            })

    asyncio.run(_run_campaign())

    report = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "strategy": strategy,
        "max_turns": max_turns,
        "successful_jailbreaks": successful_jailbreaks,
        "total_attempts": total_attempts,
        "jailbreak_rate": successful_jailbreaks / max(total_attempts, 1),
        "examples": examples,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"PyRIT report written to {output_path}")
    print(f"Strategy: {strategy} | Turns: {total_attempts} | "
          f"Jailbreaks: {successful_jailbreaks} | "
          f"Rate: {successful_jailbreaks / max(total_attempts, 1):.1%}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Run PyRIT red-teaming against Na0S"
    )
    parser.add_argument(
        "--strategy", default="crescendo",
        choices=SUPPORTED_STRATEGIES,
        help="Attack strategy (default: crescendo)"
    )
    parser.add_argument(
        "--max-turns", type=int, default=10,
        help="Max turns per campaign (default: 10)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: data/evaluation/pyrit_report.json)"
    )
    args = parser.parse_args()

    run_redteam_campaign(
        strategy=args.strategy,
        max_turns=args.max_turns,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
