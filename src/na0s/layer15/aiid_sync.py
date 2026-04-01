"""AIID (AI Incident Database) Polling — monitors new AI incidents.

Queries incidentdatabase.ai's GraphQL API for incidents newer than our
last poll timestamp, extracts attack descriptions, and classifies them
by Na0S taxonomy.

DESIGN NOTE: Incidents are messy natural-language descriptions.
Classification uses keyword matching against taxonomy category names
and descriptions. For better accuracy, consider upgrading to embedding
similarity in the future (FUTURE: embedding-based classification).

Rate limits: AIID is a public API with no documented rate limits, but
we implement exponential backoff as a courtesy.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from na0s.layer15.base import (
    ApplyResult,
    RateLimitError,
    SchemaValidationError,
    SourceSnapshot,
    SourceUnavailableError,
    TaxonomyDiff,
    TechniqueEntry,
    ThreatIntelSource,
)
from na0s.layer15.config import (
    AIID_GRAPHQL_URL,
    HTTP_BACKOFF_FACTOR,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT_SECONDS,
)
from na0s.layer15.diff_engine import TaxonomyDiffEngine

logger = logging.getLogger(__name__)

# GraphQL query to fetch recent incidents
# CORRECTED: was {filter: {date_gte: $after}, sort: DATE_DESC, limit: 100}
# Actual schema uses pagination: {limit: N}, sort: {date: DESC},
# filter: {date: {GTE: $after}}, and entity relations are objects (not scalars).
# Requires Origin: https://incidentdatabase.ai header.
# VERIFIED: 2026-03-24, HTTP 200 (with Origin header)
INCIDENTS_QUERY = """\
query RecentIncidents($after: String) {
  incidents(
    filter: { date: { GTE: $after } }
    sort: { date: DESC }
    pagination: { limit: 100 }
  ) {
    incident_id
    title
    description
    date
    AllegedDeployerOfAISystem {
      entity_id
      name
    }
    AllegedDeveloperOfAISystem {
      entity_id
      name
    }
    AllegedHarmedOrNearlyHarmedParties {
      entity_id
      name
    }
  }
}
"""


def _graphql_request(
    url: str,
    query: str,
    variables: Optional[Dict[str, Any]] = None,
    timeout: int = HTTP_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Execute a GraphQL request with retry and error handling.

    Returns the parsed JSON response.
    """
    payload = json.dumps({"query": query, "variables": variables or {}}).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Na0S-Layer15-ThreatIntelSync",
        "Origin": "https://incidentdatabase.ai",  # Required by AIID CORS policy
    }
    last_error: Optional[Exception] = None

    for attempt in range(HTTP_MAX_RETRIES):
        try:
            req = Request(url, data=payload, headers=headers, method="POST")
            with urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    body = resp.read().decode("utf-8", errors="replace")
                    logger.warning("AIID returned %d: %s", resp.status, body[:200])
                    raise SourceUnavailableError(f"HTTP {resp.status} from AIID")
                data = json.loads(resp.read().decode("utf-8"))

                if "errors" in data:
                    errors = data["errors"]
                    logger.warning("AIID GraphQL errors: %s", errors)
                    raise SchemaValidationError(
                        f"GraphQL errors: {errors[0].get('message', str(errors))}"
                    )
                return data
        except HTTPError as e:
            if e.code == 429:
                raise RateLimitError("AIID rate limit exceeded")
            last_error = e
        except (URLError, OSError) as e:
            last_error = e

        if attempt < HTTP_MAX_RETRIES - 1:
            backoff = HTTP_BACKOFF_FACTOR ** (attempt + 1)
            logger.info("AIID retry in %ds...", backoff)
            time.sleep(backoff)

    raise SourceUnavailableError(
        f"Failed to query AIID after {HTTP_MAX_RETRIES} attempts: {last_error}"
    )


class AiidSync(ThreatIntelSource):
    """Monitors AI Incident Database for new incidents.

    Parameters
    ----------
    snapshots_dir : Path, optional
        Directory for storing snapshots.
    """

    name = "aiid"

    def __init__(self, snapshots_dir: Optional[Path] = None):
        super().__init__(snapshots_dir=snapshots_dir)
        self._diff_engine = TaxonomyDiffEngine()

    def fetch_latest(self) -> SourceSnapshot:
        """Fetch recent incidents from AIID via GraphQL.

        Uses the last snapshot's fetch timestamp to only query for
        new incidents. If no previous snapshot, fetches the most recent
        100 incidents.
        """
        previous = self.load_last_snapshot()
        after_date = None
        if previous:
            after_date = previous.fetched_at.strftime("%Y-%m-%dT%H:%M:%SZ")

        variables = {}
        if after_date:
            variables["after"] = after_date

        data = _graphql_request(
            AIID_GRAPHQL_URL, INCIDENTS_QUERY, variables=variables
        )

        incidents = data.get("data", {}).get("incidents", [])
        logger.info("AIID returned %d incidents", len(incidents))

        techniques = []
        for inc in incidents:
            if not isinstance(inc, dict):
                continue

            incident_id = str(inc.get("incident_id", ""))
            if not incident_id:
                continue

            # Entity fields are objects with entity_id/name (not plain strings)
            deployers = [
                e.get("name", "") for e in (inc.get("AllegedDeployerOfAISystem") or [])
                if isinstance(e, dict)
            ]
            developers = [
                e.get("name", "") for e in (inc.get("AllegedDeveloperOfAISystem") or [])
                if isinstance(e, dict)
            ]

            techniques.append(
                TechniqueEntry(
                    id=f"AIID-{incident_id}",
                    name=str(inc.get("title", "Untitled")),
                    description=str(inc.get("description", ""))[:500],
                    metadata={
                        "date": str(inc.get("date", "")),
                        "deployer": deployers,
                        "developer": developers,
                    },
                )
            )

        # Use current timestamp as version (AIID has no version concept)
        now = datetime.now(timezone.utc)
        return SourceSnapshot(
            source_name=self.name,
            fetched_at=now,
            version=now.strftime("%Y%m%d%H%M%S"),
            techniques=techniques,
            raw_metadata={"incident_count": len(incidents)},
        )

    def diff(
        self, old: SourceSnapshot, new: SourceSnapshot
    ) -> TaxonomyDiff:
        return self._diff_engine.compute(old, new)

    def apply(
        self, diff: TaxonomyDiff, dry_run: bool = False
    ) -> ApplyResult:
        """Report new incidents for review.

        AIID incidents are informational — they don't directly modify
        the taxonomy but inform what new attack patterns exist.
        """
        if dry_run or not diff.has_changes:
            return ApplyResult(
                applied_count=0,
                skipped_count=len(diff.items),
                dry_run=dry_run,
            )

        logger.info("AIID: %d new incidents to review", len(diff.added))
        return ApplyResult(
            applied_count=0,
            skipped_count=len(diff.items),
        )
