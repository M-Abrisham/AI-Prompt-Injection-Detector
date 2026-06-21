"""Static tests for the social-scraper.yml GitHub Actions workflow.

These tests validate that social-scraper.yml is well-formed YAML,
contains the required schedule, steps, permissions, and environment
variables without needing Docker or GitHub Actions runners.
"""

import pathlib
import unittest

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
SCRAPER_YML = WORKFLOWS / "social-scraper.yml"


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
    """social-scraper.yml must exist at the expected path."""

    def test_workflow_exists(self):
        self.assertTrue(
            SCRAPER_YML.exists(),
            f"Expected workflow file at {SCRAPER_YML}",
        )


# ---------------------------------------------------------------------------
# 2. Valid YAML
# ---------------------------------------------------------------------------

class TestWorkflowValidYAML(unittest.TestCase):
    """social-scraper.yml must parse as valid YAML."""

    def test_workflow_valid_yaml(self):
        data = _load_yaml(SCRAPER_YML)
        self.assertIsInstance(data, dict)


# ---------------------------------------------------------------------------
# 3. Schedule trigger
# ---------------------------------------------------------------------------

class TestWorkflowSchedule(unittest.TestCase):
    """The workflow must have a cron schedule trigger."""

    def test_workflow_has_schedule(self):
        data = _load_yaml(SCRAPER_YML)
        triggers = _get_on_trigger(data)
        self.assertIn(
            "schedule", triggers,
            "Workflow must have a 'schedule' trigger",
        )


# ---------------------------------------------------------------------------
# 4. Cron is every 3 hours
# ---------------------------------------------------------------------------

class TestWorkflowCronEvery3Hours(unittest.TestCase):
    """The cron expression must fire every 3 hours."""

    def test_cron_every_3_hours(self):
        data = _load_yaml(SCRAPER_YML)
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
        # Minute must be 0
        self.assertEqual(parts[0], "0", f"Minute field must be '0', got: '{parts[0]}'")
        # Hour must be */3
        self.assertEqual(parts[1], "*/3", f"Hour field must be '*/3', got: '{parts[1]}'")
        # Day, month, day-of-week must be *
        for i, label in [(2, "day-of-month"), (3, "month"), (4, "day-of-week")]:
            self.assertEqual(
                parts[i], "*",
                f"{label} field must be '*', got: '{parts[i]}'",
            )


# ---------------------------------------------------------------------------
# 5. Manual dispatch
# ---------------------------------------------------------------------------

class TestWorkflowManualDispatch(unittest.TestCase):
    """The workflow must support manual triggering."""

    def test_workflow_has_manual_dispatch(self):
        data = _load_yaml(SCRAPER_YML)
        triggers = _get_on_trigger(data)
        self.assertIn(
            "workflow_dispatch", triggers,
            "Workflow must have a 'workflow_dispatch' trigger for manual runs",
        )


# ---------------------------------------------------------------------------
# 6. Python 3.12
# ---------------------------------------------------------------------------

class TestWorkflowPython(unittest.TestCase):
    """The workflow must use Python 3.12."""

    def test_workflow_python_312(self):
        data = _load_yaml(SCRAPER_YML)
        versions = []
        for _job_name, job in data.get("jobs", {}).items():
            for step in job.get("steps", []):
                with_block = step.get("with", {})
                if "python-version" in with_block:
                    versions.append(str(with_block["python-version"]))
        self.assertTrue(
            any("3.12" in v for v in versions),
            f"Expected Python 3.12 in scraper workflow, found: {versions}",
        )


# ---------------------------------------------------------------------------
# 7. Runs social_scraper.py script
# ---------------------------------------------------------------------------

class TestWorkflowRunsScraperScript(unittest.TestCase):
    """The workflow must invoke the social_scraper.py script."""

    def test_workflow_runs_scraper_script(self):
        data = _load_yaml(SCRAPER_YML)
        runs = _step_runs(data)
        self.assertIn(
            "social_scraper.py", runs,
            "Workflow must run 'social_scraper.py'",
        )


# ---------------------------------------------------------------------------
# 8. Permissions
# ---------------------------------------------------------------------------

class TestWorkflowPermissions(unittest.TestCase):
    """The workflow declares contents:write only (least-privilege; it commits to
    a holding branch instead of opening a PR per run — see commit ecc2e1d)."""

    def test_workflow_has_permissions(self):
        data = _load_yaml(SCRAPER_YML)
        permissions = data.get("permissions", {})
        self.assertIn(
            "contents", permissions,
            "Workflow must declare 'contents' permission",
        )
        self.assertEqual(
            permissions["contents"], "write",
            "Workflow must have 'contents: write' permission",
        )


# ---------------------------------------------------------------------------
# 9. Environment tokens reference secrets
# ---------------------------------------------------------------------------

class TestWorkflowEnvTokens(unittest.TestCase):
    """The workflow must pass TWITTER_BEARER_TOKEN as an env var from secrets."""

    def _all_env_keys(self, data: dict) -> set[str]:
        """Collect all environment variable names from all steps."""
        keys = set()
        for _job_name, job in data.get("jobs", {}).items():
            for step in job.get("steps", []):
                env = step.get("env", {})
                keys.update(env.keys())
        return keys

    def test_workflow_uses_twitter_token(self):
        data = _load_yaml(SCRAPER_YML)
        env_keys = self._all_env_keys(data)
        self.assertIn(
            "TWITTER_BEARER_TOKEN", env_keys,
            "Workflow must pass TWITTER_BEARER_TOKEN as an environment variable",
        )

    def test_tokens_reference_secrets(self):
        """Token values must reference GitHub secrets."""
        data = _load_yaml(SCRAPER_YML)
        for _job_name, job in data.get("jobs", {}).items():
            for step in job.get("steps", []):
                env = step.get("env", {})
                for key in ("TWITTER_BEARER_TOKEN",):
                    if key in env:
                        value = str(env[key])
                        self.assertIn(
                            "secrets.", value,
                            f"{key} must reference a GitHub secret, got: {value}",
                        )


# ---------------------------------------------------------------------------
# 10. Workflow name
# ---------------------------------------------------------------------------

class TestWorkflowName(unittest.TestCase):
    """The workflow must have a descriptive name."""

    def test_workflow_has_name(self):
        data = _load_yaml(SCRAPER_YML)
        self.assertIn("name", data)
        self.assertIsInstance(data["name"], str)
        self.assertTrue(
            len(data["name"]) > 0,
            "Workflow name must not be empty",
        )


# ---------------------------------------------------------------------------
# 11. Commits new data to the data/scraped-raw holding branch
# ---------------------------------------------------------------------------

class TestWorkflowCommitsToHoldingBranch(unittest.TestCase):
    """When new data is found the workflow commits it to the data/scraped-raw
    holding branch (replaces the old PR-per-run flow — see commit ecc2e1d)."""

    def test_workflow_pushes_to_holding_branch(self):
        data = _load_yaml(SCRAPER_YML)
        runs = _step_runs(data)
        self.assertIn(
            "data/scraped-raw", runs,
            "Workflow must push new data to the 'data/scraped-raw' holding branch",
        )

    def test_commit_step_is_conditional(self):
        """The data-export step must be conditional on new_count."""
        data = _load_yaml(SCRAPER_YML)
        steps = _all_steps(data)
        commit_steps = [s for s in steps if "data/scraped-raw" in s.get("run", "")]
        self.assertTrue(len(commit_steps) > 0, "Must have a holding-branch commit step")
        for s in commit_steps:
            condition = s.get("if", "")
            self.assertIn(
                "new_count", condition,
                "Holding-branch commit step must be conditional on 'new_count'",
            )


if __name__ == "__main__":
    unittest.main()
