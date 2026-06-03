#!/usr/bin/env python3
"""CLI entry point for running individual agent tasks.

Usage:
    python scripts/run_agent.py --gate-analyzer
    python scripts/run_agent.py --quarantine-review
    python scripts/run_agent.py --deploy-approval
    python scripts/run_agent.py --synthetic-validation
    python scripts/run_agent.py --full-pipeline
    python scripts/run_agent.py --continuous
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from na0s.agents.orchestrator import PipelineOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Na0S Agent CLI — run individual agent tasks",
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory path (default: data)",
    )
    parser.add_argument(
        "--openclaw-url",
        default="http://localhost:3000",
        help="OpenClaw API endpoint (default: http://localhost:3000)",
    )

    # Individual agent tasks
    parser.add_argument(
        "--gate-analyzer",
        action="store_true",
        help="Run gate failure analysis",
    )
    parser.add_argument(
        "--quarantine-review",
        action="store_true",
        help="Run quarantine backlog review",
    )
    parser.add_argument(
        "--deploy-approval",
        action="store_true",
        help="Run deployment approval workflow",
    )
    parser.add_argument(
        "--synthetic-validation",
        action="store_true",
        help="Run synthetic data validation",
    )

    # Full pipeline
    parser.add_argument(
        "--full-pipeline",
        action="store_true",
        help="Run complete orchestration pipeline",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run continuous monitoring loop",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=300,
        help="Polling interval for continuous mode (seconds)",
    )

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(
        data_dir=args.data_dir,
        openclaw_url=args.openclaw_url,
    )

    try:
        if args.gate_analyzer:
            logger.info("Running gate analysis...")
            orchestrator.run_gate_analysis()
        elif args.quarantine_review:
            logger.info("Running quarantine review...")
            orchestrator.run_quarantine_review()
        elif args.deploy_approval:
            logger.info("Running deployment approval...")
            orchestrator.run_deployment_approval()
        elif args.synthetic_validation:
            logger.info("Running synthetic validation...")
            orchestrator.run_synthetic_validation()
        elif args.full_pipeline:
            logger.info("Running full pipeline...")
            orchestrator.run_full_pipeline()
        elif args.continuous:
            logger.info("Starting continuous monitoring...")
            orchestrator.run_continuous_monitoring(interval=args.poll_interval)
        else:
            # Default: run full pipeline
            logger.info("No task specified, running full pipeline...")
            orchestrator.run_full_pipeline()

        logger.info("Task completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.info("Task interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Task failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
