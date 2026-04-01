"""Endpoint health checker for Layer 15 threat intel sources.

Lightweight utility to verify reachability and latency of all upstream
APIs used by Layer 15 sync modules. Designed for use in CI health checks,
CLI diagnostics, and pre-sync validation.

Thresholds:
- OK: response in <2000ms
- SLOW: response in 2000-{timeout}ms
- UNREACHABLE: connection failed or timed out
- RATE_LIMITED: HTTP 403/429 with rate-limit headers
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from na0s.layer15.config import (
    AIID_GRAPHQL_URL,
    ATLAS_API_URL,
    GARAK_API_URL,
    GITHUB_API_BASE,
    JAILBREAKBENCH_GITHUB_OWNER,
    JAILBREAKBENCH_GITHUB_REPO,
    OWASP_GITHUB_OWNER,
    OWASP_GITHUB_REPO,
    SAFETYPROMPTS_GITHUB_OWNER,
    SAFETYPROMPTS_GITHUB_REPO,
)

SLOW_THRESHOLD_MS = 2000


@dataclass
class EndpointStatus:
    """Result of checking a single endpoint."""

    name: str
    url: str
    status: str  # "ok", "slow", "unreachable", "rate_limited"
    response_time_ms: int
    status_code: int
    error: str = ""


@dataclass
class HealthReport:
    """Aggregated health report across all endpoints."""

    endpoints: List[EndpointStatus]
    timestamp: datetime

    def format_terminal(self) -> str:
        """Format as terminal-friendly output with status indicators."""
        lines = []
        for ep in self.endpoints:
            # Pad name + dots to align statuses
            label = ep.name
            dots = "." * max(1, 40 - len(label))
            if ep.status == "ok":
                indicator = f"OK ({ep.response_time_ms}ms)"
            elif ep.status == "slow":
                indicator = f"SLOW ({ep.response_time_ms}ms)"
            elif ep.status == "rate_limited":
                indicator = f"RATE LIMITED (HTTP {ep.status_code})"
            else:
                detail = ep.error or f"HTTP {ep.status_code}"
                indicator = f"UNREACHABLE ({detail})"
            lines.append(f"{label} {dots} {indicator}")

        header = f"Layer 15 Endpoint Health  [{self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}]"
        separator = "-" * max(len(header), max((len(l) for l in lines), default=0))
        return "\n".join([header, separator] + lines + [separator])

    def all_ok(self) -> bool:
        """True if every endpoint is ok or slow (reachable)."""
        return all(ep.status in ("ok", "slow") for ep in self.endpoints)

    def to_dict(self) -> Dict:
        """Serialize to a plain dict for JSON output."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "all_ok": self.all_ok(),
            "endpoints": [
                {
                    "name": ep.name,
                    "url": ep.url,
                    "status": ep.status,
                    "response_time_ms": ep.response_time_ms,
                    "status_code": ep.status_code,
                    "error": ep.error,
                }
                for ep in self.endpoints
            ],
        }


class EndpointHealthChecker:
    """Check reachability and latency of all Layer 15 upstream sources."""

    # Each entry: (name, url, method, headers, body)
    # Body is only used for GraphQL (POST) endpoints.
    ENDPOINTS: List[Tuple[str, str, str, Dict[str, str], Optional[bytes]]] = [
        (
            "ATLAS GitHub API",
            ATLAS_API_URL,
            "GET",
            {"Accept": "application/vnd.github.v3+json", "User-Agent": "Na0S-HealthCheck"},
            None,
        ),
        (
            "Garak GitHub API",
            GARAK_API_URL,
            "GET",
            {"Accept": "application/vnd.github.v3+json", "User-Agent": "Na0S-HealthCheck"},
            None,
        ),
        (
            "AIID GraphQL",
            AIID_GRAPHQL_URL,
            "POST",
            {
                "Content-Type": "application/json",
                "User-Agent": "Na0S-HealthCheck",
                "Origin": "https://incidentdatabase.ai",
            },
            json.dumps({"query": "{ __typename }"}).encode("utf-8"),
        ),
        (
            "JailbreakBench GitHub API",
            f"{GITHUB_API_BASE}/repos/{JAILBREAKBENCH_GITHUB_OWNER}/{JAILBREAKBENCH_GITHUB_REPO}",
            "GET",
            {"Accept": "application/vnd.github.v3+json", "User-Agent": "Na0S-HealthCheck"},
            None,
        ),
        (
            "OWASP LLM Top 10 GitHub API",
            f"{GITHUB_API_BASE}/repos/{OWASP_GITHUB_OWNER}/{OWASP_GITHUB_REPO}",
            "GET",
            {"Accept": "application/vnd.github.v3+json", "User-Agent": "Na0S-HealthCheck"},
            None,
        ),
        (
            "SafetyPrompts GitHub API",
            f"{GITHUB_API_BASE}/repos/{SAFETYPROMPTS_GITHUB_OWNER}/{SAFETYPROMPTS_GITHUB_REPO}",
            "GET",
            {"Accept": "application/vnd.github.v3+json", "User-Agent": "Na0S-HealthCheck"},
            None,
        ),
    ]

    def check_all(self, timeout: int = 10) -> HealthReport:
        """Check every registered endpoint and return a health report."""
        statuses = []
        for name, url, method, headers, body in self.ENDPOINTS:
            statuses.append(self.check_one(name, url, timeout, method, headers, body))
        return HealthReport(
            endpoints=statuses,
            timestamp=datetime.now(timezone.utc),
        )

    def check_one(
        self,
        name: str,
        url: str,
        timeout: int = 10,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        body: Optional[bytes] = None,
    ) -> EndpointStatus:
        """Check a single endpoint for reachability and latency."""
        headers = headers or {"User-Agent": "Na0S-HealthCheck"}
        start = time.monotonic()
        try:
            req = Request(url, data=body, headers=headers, method=method)
            with urlopen(req, timeout=timeout) as resp:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                # Consume the body so the connection can be reused
                resp.read()
                status_code = resp.status

                # Check for rate limiting even on 200 (GitHub sends headers)
                remaining = resp.headers.get("X-RateLimit-Remaining", "")
                if remaining and remaining.isdigit() and int(remaining) == 0:
                    return EndpointStatus(
                        name=name,
                        url=url,
                        status="rate_limited",
                        response_time_ms=elapsed_ms,
                        status_code=status_code,
                    )

                if elapsed_ms >= SLOW_THRESHOLD_MS:
                    return EndpointStatus(
                        name=name,
                        url=url,
                        status="slow",
                        response_time_ms=elapsed_ms,
                        status_code=status_code,
                    )

                return EndpointStatus(
                    name=name,
                    url=url,
                    status="ok",
                    response_time_ms=elapsed_ms,
                    status_code=status_code,
                )

        except HTTPError as e:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            # Detect rate limiting
            if e.code in (403, 429):
                remaining = e.headers.get("X-RateLimit-Remaining", "")
                if e.code == 429 or (remaining and remaining.isdigit() and int(remaining) == 0):
                    return EndpointStatus(
                        name=name,
                        url=url,
                        status="rate_limited",
                        response_time_ms=elapsed_ms,
                        status_code=e.code,
                    )
            return EndpointStatus(
                name=name,
                url=url,
                status="unreachable",
                response_time_ms=elapsed_ms,
                status_code=e.code,
                error=f"HTTP {e.code}",
            )

        except (URLError, OSError) as e:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return EndpointStatus(
                name=name,
                url=url,
                status="unreachable",
                response_time_ms=elapsed_ms,
                status_code=0,
                error=str(e),
            )
