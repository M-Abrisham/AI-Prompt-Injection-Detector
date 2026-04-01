"""Layer 15 configuration constants.

All magic strings, URLs, timeouts, and version thresholds live here.
If an upstream source changes its URL or schema, update this file only.

CRITICAL: Every URL below was assumed at implementation time and tagged
with a verification date. Re-verify before trusting.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the Na0S repository (two levels up from this file's package dir)
_PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _PACKAGE_DIR.parent.parent.parent  # src/na0s/layer15 -> repo root

TAXONOMY_PATH = REPO_ROOT / "data" / "taxonomy.yaml"
SNAPSHOTS_DIR = REPO_ROOT / "data" / "threat_intel_snapshots"

# ---------------------------------------------------------------------------
# HTTP defaults
# ---------------------------------------------------------------------------

HTTP_TIMEOUT_SECONDS = 30
HTTP_MAX_RETRIES = 3
HTTP_BACKOFF_FACTOR = 2  # Exponential backoff: 2, 4, 8 seconds

# ---------------------------------------------------------------------------
# MITRE ATLAS
# TODO: VERIFY — endpoint assumed (last checked: 2026-03)
# ---------------------------------------------------------------------------

ATLAS_GITHUB_OWNER = "mitre"
ATLAS_GITHUB_REPO = "atlas"
ATLAS_TECHNIQUES_PATH = "data/techniques"  # path within the repo
ATLAS_API_URL = (
    f"https://api.github.com/repos/{ATLAS_GITHUB_OWNER}/{ATLAS_GITHUB_REPO}"
)
ATLAS_RAW_URL = (
    f"https://raw.githubusercontent.com/{ATLAS_GITHUB_OWNER}/{ATLAS_GITHUB_REPO}"
)
ATLAS_MAPPING_FILE = SNAPSHOTS_DIR / "atlas_to_na0s_mapping.yaml"

# ---------------------------------------------------------------------------
# Garak
# TODO: VERIFY — endpoint assumed (last checked: 2026-03)
# ---------------------------------------------------------------------------

GARAK_GITHUB_OWNER = "leondz"
GARAK_GITHUB_REPO = "garak"
GARAK_API_URL = (
    f"https://api.github.com/repos/{GARAK_GITHUB_OWNER}/{GARAK_GITHUB_REPO}"
)
GARAK_PROBES_PATH = "garak/probes"  # path within the repo

# ---------------------------------------------------------------------------
# AIID (AI Incident Database)
# TODO: VERIFY — endpoint assumed (last checked: 2026-03)
# ---------------------------------------------------------------------------

AIID_GRAPHQL_URL = "https://incidentdatabase.ai/api/graphql"

# ---------------------------------------------------------------------------
# JailbreakBench / HarmBench
# TODO: VERIFY — endpoint assumed (last checked: 2026-03)
# ---------------------------------------------------------------------------

JAILBREAKBENCH_GITHUB_OWNER = "JailbreakBench"
JAILBREAKBENCH_GITHUB_REPO = "jailbreakbench"
HARMBENCH_GITHUB_OWNER = "centerforaisafety"
HARMBENCH_GITHUB_REPO = "HarmBench"

# ---------------------------------------------------------------------------
# OWASP LLM Top 10
# TODO: VERIFY — endpoint assumed (last checked: 2026-03)
# ---------------------------------------------------------------------------

OWASP_GITHUB_OWNER = "OWASP"
OWASP_GITHUB_REPO = "www-project-top-10-for-large-language-model-applications"

# ---------------------------------------------------------------------------
# SafetyPrompts / jailbreak_llms
# TODO: VERIFY — endpoint assumed (last checked: 2026-03)
# ---------------------------------------------------------------------------

SAFETYPROMPTS_GITHUB_OWNER = "verazuo"
SAFETYPROMPTS_GITHUB_REPO = "jailbreak_llms"

# ---------------------------------------------------------------------------
# GitHub API
# ---------------------------------------------------------------------------

GITHUB_API_BASE = "https://api.github.com"
# Authenticated: 5,000 req/hr.  Unauthenticated: 60 req/hr.
GITHUB_RATE_LIMIT_BUFFER = 100  # Stop making requests when remaining < this
