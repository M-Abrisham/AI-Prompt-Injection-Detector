"""Shared HTTP utilities for Layer 15 sync modules.

Provides common functions for GitHub API requests, JSON fetching, and
text fetching with retry, timeout, and rate limit handling. Used by
all sync modules to avoid code duplication.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from na0s.layer15.base import RateLimitError, SourceUnavailableError
from na0s.layer15.config import (
    GITHUB_RATE_LIMIT_BUFFER,
    HTTP_BACKOFF_FACTOR,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

# Shared rate limit state — updated after each GitHub API response
_rate_limit_remaining: Optional[int] = None


def github_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Build GitHub API request headers.

    Parameters
    ----------
    token : str, optional
        GitHub API token. Falls back to GITHUB_TOKEN env var.
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Na0S-Layer15-ThreatIntelSync",
    }
    tok = token or os.environ.get("GITHUB_TOKEN", "")
    if tok:
        headers["Authorization"] = f"token {tok}"
    return headers


def _track_rate_limit(resp_headers: Dict[str, str]) -> None:
    """Update shared rate limit state from GitHub API response headers."""
    global _rate_limit_remaining
    remaining = resp_headers.get("x-ratelimit-remaining", "")
    if remaining.isdigit():
        _rate_limit_remaining = int(remaining)
        if _rate_limit_remaining <= GITHUB_RATE_LIMIT_BUFFER:
            logger.warning(
                "GitHub API rate limit low: %d remaining (buffer=%d)",
                _rate_limit_remaining,
                GITHUB_RATE_LIMIT_BUFFER,
            )


def check_rate_limit() -> bool:
    """Return True if we have enough GitHub API budget to continue.

    Returns True if rate limit is unknown (never checked) or above buffer.
    """
    if _rate_limit_remaining is None:
        return True
    return _rate_limit_remaining > GITHUB_RATE_LIMIT_BUFFER


def fetch_json(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = HTTP_TIMEOUT_SECONDS,
    etag: str = "",
) -> Tuple[Optional[Dict[str, Any]], Dict[str, str]]:
    """Fetch JSON from a URL with retry and error handling.

    Returns (parsed_json, response_headers). If ``etag`` is provided,
    sends an ``If-None-Match`` header; returns ``(None, headers)`` on
    HTTP 304 (Not Modified) — caller can skip processing.

    Raises SourceUnavailableError on network errors,
    RateLimitError on 403 with rate limit headers.
    """
    headers = dict(headers or {})
    if etag:
        headers["If-None-Match"] = etag
    last_error: Optional[Exception] = None

    for attempt in range(HTTP_MAX_RETRIES):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    body = resp.read().decode("utf-8", errors="replace")
                    logger.warning(
                        "HTTP %d from %s: %s", resp.status, url, body[:200]
                    )
                    raise SourceUnavailableError(
                        f"HTTP {resp.status} from {url}"
                    )
                resp_headers = {
                    k.lower(): v for k, v in resp.getheaders()
                }
                _track_rate_limit(resp_headers)
                data = json.loads(resp.read().decode("utf-8"))
                return data, resp_headers
        except HTTPError as e:
            if e.code == 304:
                # Not Modified — ETag matched, data unchanged
                resp_headers = {
                    k.lower(): v for k, v in (e.headers.items() if e.headers else [])
                }
                return None, resp_headers
            if e.code == 403:
                remaining = e.headers.get("X-RateLimit-Remaining", "")
                if remaining == "0":
                    reset_ts = e.headers.get("X-RateLimit-Reset", "")
                    raise RateLimitError(
                        f"GitHub rate limit exceeded. Resets at {reset_ts}"
                    )
            if e.code == 404:
                raise SourceUnavailableError(f"Not found: {url}")
            last_error = e
            logger.warning(
                "Fetch attempt %d/%d failed (HTTP %d): %s",
                attempt + 1,
                HTTP_MAX_RETRIES,
                e.code,
                str(e),
            )
        except (URLError, OSError) as e:
            last_error = e
            logger.warning(
                "Fetch attempt %d/%d failed: %s",
                attempt + 1,
                HTTP_MAX_RETRIES,
                str(e),
            )

        if attempt < HTTP_MAX_RETRIES - 1:
            backoff = HTTP_BACKOFF_FACTOR ** (attempt + 1)
            logger.info("Retrying in %ds...", backoff)
            time.sleep(backoff)

    raise SourceUnavailableError(
        f"Failed to fetch {url} after {HTTP_MAX_RETRIES} attempts: {last_error}"
    )


def fetch_text(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = HTTP_TIMEOUT_SECONDS,
) -> str:
    """Fetch raw text from a URL.

    Raises SourceUnavailableError on any failure.
    """
    headers = headers or {}
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise SourceUnavailableError(f"HTTP {resp.status} from {url}")
            return resp.read().decode("utf-8")
    except (HTTPError, URLError, OSError) as e:
        raise SourceUnavailableError(f"Failed to fetch {url}: {e}")


# Backward-compat aliases — modules that imported private names from atlas_sync
_github_headers = github_headers
_fetch_json = fetch_json
_fetch_text = fetch_text
