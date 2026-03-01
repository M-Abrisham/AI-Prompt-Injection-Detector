"""Static tests for GitHub Actions workflow files.

These tests validate that publish.yml and ci.yml are well-formed YAML,
contain the required triggers, steps, and configuration without needing
Docker or GitHub Actions runners.
"""

import pathlib
import unittest

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
PUBLISH_YML = WORKFLOWS / "publish.yml"
CI_YML = WORKFLOWS / "ci.yml"


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


# ---------------------------------------------------------------------------
# publish.yml tests
# ---------------------------------------------------------------------------

class TestPublishWorkflowExists(unittest.TestCase):
    """Basic existence and YAML validity checks for publish.yml."""

    def test_publish_yml_exists(self):
        self.assertTrue(
            PUBLISH_YML.exists(),
            f"Expected workflow file at {PUBLISH_YML}",
        )

    def test_publish_yml_is_valid_yaml(self):
        data = _load_yaml(PUBLISH_YML)
        self.assertIsInstance(data, dict)

    def test_publish_yml_has_name(self):
        data = _load_yaml(PUBLISH_YML)
        self.assertIn("name", data)
        self.assertIsInstance(data["name"], str)


class TestPublishTrigger(unittest.TestCase):
    """Verify the publish workflow triggers on tag push."""

    def test_triggers_on_push(self):
        data = _load_yaml(PUBLISH_YML)
        triggers = _get_on_trigger(data)
        self.assertIn("push", triggers)

    def test_triggers_on_tag_push(self):
        data = _load_yaml(PUBLISH_YML)
        triggers = _get_on_trigger(data)
        push_config = triggers["push"]
        self.assertIn("tags", push_config)

    def test_tag_pattern_includes_v_prefix(self):
        data = _load_yaml(PUBLISH_YML)
        triggers = _get_on_trigger(data)
        tags = triggers["push"]["tags"]
        self.assertIsInstance(tags, list)
        # At least one tag pattern must start with 'v'
        v_patterns = [t for t in tags if t.startswith("v")]
        self.assertTrue(
            len(v_patterns) > 0,
            f"Expected at least one 'v*' tag pattern, got: {tags}",
        )


class TestPublishPython(unittest.TestCase):
    """Verify the publish workflow uses Python 3.12."""

    def _find_python_version(self, data: dict) -> list[str]:
        """Walk jobs/steps and collect all python-version values."""
        versions = []
        for _job_name, job in data.get("jobs", {}).items():
            for step in job.get("steps", []):
                with_block = step.get("with", {})
                if "python-version" in with_block:
                    versions.append(str(with_block["python-version"]))
        return versions

    def test_uses_python_312(self):
        data = _load_yaml(PUBLISH_YML)
        versions = self._find_python_version(data)
        self.assertTrue(
            any("3.12" in v for v in versions),
            f"Expected Python 3.12 in publish workflow, found: {versions}",
        )


class TestPublishBuildAndPublishSteps(unittest.TestCase):
    """Verify the workflow has build and publish steps."""

    def _all_steps(self, data: dict) -> list[dict]:
        steps = []
        for _job_name, job in data.get("jobs", {}).items():
            steps.extend(job.get("steps", []))
        return steps

    def _step_names(self, data: dict) -> list[str]:
        return [
            s.get("name", "") for s in self._all_steps(data) if s.get("name")
        ]

    def _step_runs(self, data: dict) -> str:
        """Concatenate all 'run' blocks into a single searchable string."""
        parts = []
        for s in self._all_steps(data):
            if "run" in s:
                parts.append(s["run"])
        return "\n".join(parts)

    def test_has_build_step(self):
        data = _load_yaml(PUBLISH_YML)
        runs = self._step_runs(data)
        self.assertIn("build", runs.lower())

    def test_has_twine_check_step(self):
        data = _load_yaml(PUBLISH_YML)
        runs = self._step_runs(data)
        self.assertIn("twine check", runs)

    def test_has_publish_step_or_action(self):
        data = _load_yaml(PUBLISH_YML)
        steps = self._all_steps(data)
        # Check for pypa publish action OR twine upload in run blocks
        uses_publish_action = any(
            "pypi-publish" in s.get("uses", "") for s in steps
        )
        runs = self._step_runs(data)
        uses_twine_upload = "twine upload" in runs
        self.assertTrue(
            uses_publish_action or uses_twine_upload,
            "Expected either pypa/gh-action-pypi-publish action or "
            "'twine upload' command in publish workflow",
        )

    def test_has_smoke_test_step(self):
        """The workflow should install the wheel and run na0s --help."""
        data = _load_yaml(PUBLISH_YML)
        runs = self._step_runs(data)
        self.assertIn("na0s --help", runs)


class TestPublishPermissions(unittest.TestCase):
    """Verify OIDC / trusted publisher permissions."""

    def test_id_token_write_permission(self):
        data = _load_yaml(PUBLISH_YML)
        permissions = data.get("permissions", {})
        self.assertEqual(
            permissions.get("id-token"),
            "write",
            "Expected top-level 'id-token: write' for OIDC trusted publishers",
        )


# ---------------------------------------------------------------------------
# ci.yml tests
# ---------------------------------------------------------------------------

class TestCIWorkflowExists(unittest.TestCase):
    """Basic checks for ci.yml."""

    def test_ci_yml_exists(self):
        self.assertTrue(
            CI_YML.exists(),
            f"Expected CI workflow file at {CI_YML}",
        )

    def test_ci_yml_is_valid_yaml(self):
        data = _load_yaml(CI_YML)
        self.assertIsInstance(data, dict)


class TestCIBenchmarkStep(unittest.TestCase):
    """Verify the CI workflow includes a non-blocking benchmark step."""

    def _all_steps(self, data: dict) -> list[dict]:
        steps = []
        for _job_name, job in data.get("jobs", {}).items():
            steps.extend(job.get("steps", []))
        return steps

    def test_has_bench_fast_step(self):
        data = _load_yaml(CI_YML)
        steps = self._all_steps(data)
        bench_steps = [
            s for s in steps
            if "bench-fast" in s.get("run", "")
            or "bench-fast" in s.get("name", "")
        ]
        self.assertTrue(
            len(bench_steps) > 0,
            "Expected a step running 'make bench-fast' in ci.yml",
        )

    def test_bench_step_is_non_blocking(self):
        data = _load_yaml(CI_YML)
        steps = self._all_steps(data)
        for s in steps:
            run_cmd = s.get("run", "")
            if "bench-fast" in run_cmd or "bench-fast" in s.get("name", ""):
                self.assertTrue(
                    s.get("continue-on-error", False),
                    "The bench-fast step must have continue-on-error: true",
                )


if __name__ == "__main__":
    unittest.main()
