"""Static tests for the weekly-harvest.yml GitHub Actions workflow.

These tests validate that weekly-harvest.yml is well-formed YAML,
contains the required schedule, steps, permissions, and environment
variables without needing Docker or GitHub Actions runners.
"""

import pathlib
import re
import unittest

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
HARVEST_YML = WORKFLOWS / "weekly-harvest.yml"


def _load_yaml(path: pathlib.Path) -> dict:
    """Load a YAML file and return the parsed dict."""
    with open(path) as fh:
        return yaml.safe_load(fh)


def _get_on_trigger(data: dict) -> dict:
    """Return the 'on' trigger block from a workflow dict.

    PyYAML parses the bare YAML key ``on:`` as the Python boolean True
    (YAML 1.1 treats 'on'/'off' as booleans).  This helper transparently
    handles both ``data["on"]`` and ``data[True]``.
    """
    if "on" in data:
        return data["on"]
    if True in data:
        return data[True]
    raise KeyError("Workflow has no 'on' trigger block")


def _all_steps(data: dict) -> list[dict]:
    """Collect all steps across all jobs."""
    steps = []
    for _job_name, job in data.get("jobs", {}).items():
        steps.extend(job.get("steps", []))
    return steps


def _step_runs(data: dict) -> str:
    """Concatenate all 'run' blocks into a single searchable string."""
    parts = []
    for s in _all_steps(data):
        if "run" in s:
            parts.append(s["run"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 1. File existence
# ---------------------------------------------------------------------------

class TestWorkflowExists(unittest.TestCase):
    """weekly-harvest.yml must exist at the expected path."""

    def test_workflow_exists(self):
        self.assertTrue(
            HARVEST_YML.exists(),
            f"Expected workflow file at {HARVEST_YML}",
        )


# ---------------------------------------------------------------------------
# 2. Valid YAML
# ---------------------------------------------------------------------------

class TestWorkflowValidYAML(unittest.TestCase):
    """weekly-harvest.yml must parse as valid YAML."""

    def test_workflow_valid_yaml(self):
        data = _load_yaml(HARVEST_YML)
        self.assertIsInstance(data, dict)


# ---------------------------------------------------------------------------
# 3. Schedule trigger
# ---------------------------------------------------------------------------

class TestWorkflowSchedule(unittest.TestCase):
    """The workflow must have a cron schedule trigger."""

    def test_workflow_has_schedule(self):
        data = _load_yaml(HARVEST_YML)
        triggers = _get_on_trigger(data)
        self.assertIn(
            "schedule", triggers,
            "Workflow must have a 'schedule' trigger",
        )

    def test_workflow_cron_syntax(self):
        """Cron expression must be a valid weekly schedule."""
        data = _load_yaml(HARVEST_YML)
        triggers = _get_on_trigger(data)
        schedule_list = triggers["schedule"]
        self.assertIsInstance(schedule_list, list)
        self.assertTrue(len(schedule_list) > 0, "schedule must have at least one entry")

        cron_expr = schedule_list[0].get("cron", "")
        parts = cron_expr.split()
        self.assertEqual(
            len(parts), 5,
            f"Cron expression must have 5 fields, got {len(parts)}: '{cron_expr}'",
        )
        # Day-of-week field (5th) must specify a single day (0-6) for weekly
        dow = parts[4]
        self.assertTrue(
            re.match(r"^[0-6]$", dow),
            f"Day-of-week field must be a single digit 0-6 for weekly schedule, got: '{dow}'",
        )


# ---------------------------------------------------------------------------
# 4. Manual dispatch
# ---------------------------------------------------------------------------

class TestWorkflowManualDispatch(unittest.TestCase):
    """The workflow must support manual triggering."""

    def test_workflow_has_manual_dispatch(self):
        data = _load_yaml(HARVEST_YML)
        triggers = _get_on_trigger(data)
        self.assertIn(
            "workflow_dispatch", triggers,
            "Workflow must have a 'workflow_dispatch' trigger for manual runs",
        )


# ---------------------------------------------------------------------------
# 5. Python 3.12
# ---------------------------------------------------------------------------

class TestWorkflowPython(unittest.TestCase):
    """The workflow must use Python 3.12."""

    def test_workflow_python_312(self):
        data = _load_yaml(HARVEST_YML)
        versions = []
        for _job_name, job in data.get("jobs", {}).items():
            for step in job.get("steps", []):
                with_block = step.get("with", {})
                if "python-version" in with_block:
                    versions.append(str(with_block["python-version"]))
        self.assertTrue(
            any("3.12" in v for v in versions),
            f"Expected Python 3.12 in harvest workflow, found: {versions}",
        )


# ---------------------------------------------------------------------------
# 6. Installs requests
# ---------------------------------------------------------------------------

class TestWorkflowDependencies(unittest.TestCase):
    """The workflow must install the requests library."""

    def test_workflow_installs_requests(self):
        data = _load_yaml(HARVEST_YML)
        runs = _step_runs(data)
        self.assertIn(
            "requests", runs,
            "Workflow must install the 'requests' package",
        )
        self.assertIn(
            "pip install", runs,
            "Workflow must use 'pip install' to install dependencies",
        )


# ---------------------------------------------------------------------------
# 7. Runs harvest script
# ---------------------------------------------------------------------------

class TestWorkflowRunsHarvestScript(unittest.TestCase):
    """The workflow must invoke the weekly_harvest.py script."""

    def test_workflow_runs_harvest_script(self):
        data = _load_yaml(HARVEST_YML)
        runs = _step_runs(data)
        self.assertIn(
            "weekly_harvest.py", runs,
            "Workflow must run 'weekly_harvest.py'",
        )


# ---------------------------------------------------------------------------
# 8. Checks new count
# ---------------------------------------------------------------------------

class TestWorkflowChecksNewCount(unittest.TestCase):
    """The workflow must have a step that checks for new discoveries."""

    def test_workflow_checks_new_count(self):
        data = _load_yaml(HARVEST_YML)
        steps = _all_steps(data)
        check_steps = [
            s for s in steps
            if "new_count" in s.get("run", "")
            or "check" in s.get("id", "")
        ]
        self.assertTrue(
            len(check_steps) > 0,
            "Workflow must have a step that checks for new dataset count",
        )

    def test_check_step_has_id(self):
        """The check step must have an 'id' so later steps can reference it."""
        data = _load_yaml(HARVEST_YML)
        steps = _all_steps(data)
        id_steps = [s for s in steps if s.get("id") == "check"]
        self.assertTrue(
            len(id_steps) > 0,
            "Workflow must have a step with id 'check'",
        )


# ---------------------------------------------------------------------------
# 9. Creates PR
# ---------------------------------------------------------------------------

class TestWorkflowCreatesPR(unittest.TestCase):
    """The workflow must create a pull request when new datasets are found."""

    def test_workflow_creates_pr(self):
        data = _load_yaml(HARVEST_YML)
        runs = _step_runs(data)
        self.assertIn(
            "gh pr create", runs,
            "Workflow must use 'gh pr create' to open a pull request",
        )

    def test_pr_step_is_conditional(self):
        """The PR creation step must be conditional on new_count != 0."""
        data = _load_yaml(HARVEST_YML)
        steps = _all_steps(data)
        pr_steps = [s for s in steps if "gh pr create" in s.get("run", "")]
        self.assertTrue(len(pr_steps) > 0, "Must have a PR creation step")
        for s in pr_steps:
            condition = s.get("if", "")
            self.assertIn(
                "new_count", condition,
                "PR step must be conditional on 'new_count'",
            )


# ---------------------------------------------------------------------------
# 10. Permissions
# ---------------------------------------------------------------------------

class TestWorkflowPermissions(unittest.TestCase):
    """The workflow must declare contents:write and pull-requests:write."""

    def test_workflow_has_permissions(self):
        data = _load_yaml(HARVEST_YML)
        permissions = data.get("permissions", {})
        self.assertIn(
            "contents", permissions,
            "Workflow must declare 'contents' permission",
        )
        self.assertEqual(
            permissions["contents"], "write",
            "Workflow must have 'contents: write' permission",
        )
        self.assertIn(
            "pull-requests", permissions,
            "Workflow must declare 'pull-requests' permission",
        )
        self.assertEqual(
            permissions["pull-requests"], "write",
            "Workflow must have 'pull-requests: write' permission",
        )


# ---------------------------------------------------------------------------
# 11. Environment tokens
# ---------------------------------------------------------------------------

class TestWorkflowEnvTokens(unittest.TestCase):
    """The workflow must pass HF_TOKEN and GITHUB_TOKEN as env vars."""

    def _all_env_keys(self, data: dict) -> set[str]:
        """Collect all environment variable names from all steps."""
        keys = set()
        for _job_name, job in data.get("jobs", {}).items():
            for step in job.get("steps", []):
                env = step.get("env", {})
                keys.update(env.keys())
        return keys

    def test_workflow_uses_env_tokens(self):
        data = _load_yaml(HARVEST_YML)
        env_keys = self._all_env_keys(data)
        self.assertIn(
            "HF_TOKEN", env_keys,
            "Workflow must pass HF_TOKEN as an environment variable",
        )
        self.assertIn(
            "GITHUB_TOKEN", env_keys,
            "Workflow must pass GITHUB_TOKEN as an environment variable",
        )

    def test_tokens_reference_secrets(self):
        """Token values must reference GitHub secrets."""
        data = _load_yaml(HARVEST_YML)
        for _job_name, job in data.get("jobs", {}).items():
            for step in job.get("steps", []):
                env = step.get("env", {})
                for key in ("HF_TOKEN", "GITHUB_TOKEN"):
                    if key in env:
                        value = str(env[key])
                        self.assertIn(
                            "secrets.", value,
                            f"{key} must reference a GitHub secret, got: {value}",
                        )


# ---------------------------------------------------------------------------
# 12. Workflow name
# ---------------------------------------------------------------------------

class TestWorkflowName(unittest.TestCase):
    """The workflow must have a descriptive name."""

    def test_workflow_has_name(self):
        data = _load_yaml(HARVEST_YML)
        self.assertIn("name", data)
        self.assertIsInstance(data["name"], str)
        self.assertTrue(
            len(data["name"]) > 0,
            "Workflow name must not be empty",
        )


if __name__ == "__main__":
    unittest.main()
