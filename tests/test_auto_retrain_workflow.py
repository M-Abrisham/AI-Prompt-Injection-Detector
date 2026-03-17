"""Static tests for the auto-retrain.yml GitHub Actions workflow."""

import pathlib
import unittest

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
AUTO_RETRAIN_YML = WORKFLOWS / "auto-retrain.yml"


def _load_yaml(path: pathlib.Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _all_steps(data: dict) -> list[dict]:
    steps = []
    for _job_name, job in data.get("jobs", {}).items():
        steps.extend(job.get("steps", []))
    return steps


class TestWorkflowExists(unittest.TestCase):
    def test_workflow_exists(self):
        self.assertTrue(
            AUTO_RETRAIN_YML.exists(),
            f"Expected workflow file at {AUTO_RETRAIN_YML}",
        )


class TestWorkflowQuarantineGates(unittest.TestCase):
    def test_workflow_has_validate_quarantined_step(self):
        data = _load_yaml(AUTO_RETRAIN_YML)
        runs = "\n".join(step.get("run", "") for step in _all_steps(data))
        self.assertIn(
            "python -m scripts.quarantine --validate-quarantined",
            runs,
            "Workflow must validate quarantined datasets before training",
        )

    def test_workflow_has_explicit_promotion_step(self):
        data = _load_yaml(AUTO_RETRAIN_YML)
        runs = "\n".join(step.get("run", "") for step in _all_steps(data))
        self.assertIn(
            "python -m scripts.quarantine --promote-validated",
            runs,
            "Workflow must explicitly promote validated quarantine entries",
        )

    def test_quarantine_steps_run_before_process_data(self):
        data = _load_yaml(AUTO_RETRAIN_YML)
        steps = _all_steps(data)

        validate_idx = None
        promote_idx = None
        process_idx = None
        for idx, step in enumerate(steps):
            run = step.get("run", "")
            if "python -m scripts.quarantine --validate-quarantined" in run:
                validate_idx = idx
            if "python -m scripts.quarantine --promote-validated" in run:
                promote_idx = idx
            if "python -m scripts.process_data" in run:
                process_idx = idx

        self.assertIsNotNone(validate_idx, "Missing --validate-quarantined step")
        self.assertIsNotNone(promote_idx, "Missing --promote-validated step")
        self.assertIsNotNone(process_idx, "Missing process_data step")
        self.assertLess(validate_idx, promote_idx)
        self.assertLess(promote_idx, process_idx)


if __name__ == "__main__":
    unittest.main()
