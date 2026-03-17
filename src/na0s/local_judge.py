"""Local LLM judge via Ollama — open-source fallback for prompt injection detection.

When both OpenAI and Groq API backends are unavailable, the cascade can fall
back to a locally-running Ollama instance.  This keeps the detection pipeline
functional even without cloud API keys or internet connectivity.

Environment variables
---------------------
NA0S_LOCAL_JUDGE_MODEL : str
    Ollama model to use (default: ``llama3.1:8b``).
NA0S_LOCAL_JUDGE_URL : str
    Ollama base URL (default: ``http://localhost:11434``).
"""

import json
import math
import os
import re
import secrets
import time
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from .llm_judge import (
    JUDGE_INPUT_MAX_CHARS,
    JUDGE_SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    JudgeVerdict,
    _CONTROL_RE,
    _patch_few_shot_nonce,
)

_DEFAULT_MODEL = "llama3.1:8b"
_DEFAULT_BASE_URL = "http://localhost:11434"


class LocalLLMJudge:
    """LLM judge backed by a local Ollama instance.

    Implements the same ``classify()`` interface as :class:`LLMJudge` so it
    can be used as a drop-in replacement or fallback.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        use_few_shot: bool = True,
        timeout: float = 30.0,
    ):
        self.model = (
            model
            or os.getenv("NA0S_LOCAL_JUDGE_MODEL")
            or _DEFAULT_MODEL
        )
        self.base_url = (
            base_url
            or os.getenv("NA0S_LOCAL_JUDGE_URL")
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        self.use_few_shot = use_few_shot
        self.timeout = timeout

    # ---- public API ----

    def is_available(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            req = Request(
                self.base_url + "/api/tags",
                method="GET",
            )
            resp = urlopen(req, timeout=min(self.timeout, 5.0))
            return resp.status == 200
        except Exception:
            return False

    def classify(self, user_input: str) -> JudgeVerdict:
        """Classify a single input.  Returns a JudgeVerdict."""
        nonce = secrets.token_hex(8)
        prompt_text = self._build_prompt(user_input, nonce=nonce)
        start = time.monotonic()

        try:
            payload = json.dumps({
                "model": self.model,
                "prompt": prompt_text,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                },
            }).encode("utf-8")

            req = Request(
                self.base_url + "/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urlopen(req, timeout=self.timeout)
            latency_ms = (time.monotonic() - start) * 1000
            body = json.loads(resp.read().decode("utf-8"))
            content = body.get("response", "")

            # Verify nonce
            if not self._verify_nonce(content, nonce):
                return JudgeVerdict(
                    verdict="UNKNOWN",
                    confidence=0.0,
                    reasoning="Nonce verification failed; local judge may be hijacked",
                    latency_ms=latency_ms,
                    model=self.model,
                    error="nonce_mismatch",
                )

            return self._parse_response(content, latency_ms)

        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            return JudgeVerdict(
                verdict="UNKNOWN",
                confidence=0.0,
                reasoning="Local LLM judge call failed",
                latency_ms=latency_ms,
                model=self.model,
                error=str(exc),
            )

    # ---- internal helpers ----

    def _build_prompt(self, user_input: str, nonce: Optional[str] = None) -> str:
        """Build a single text prompt for the Ollama /api/generate endpoint.

        Ollama's generate API takes a single ``prompt`` string rather than
        the chat-style ``messages`` array, so we flatten the system prompt,
        few-shot examples and user input into one block.
        """
        if len(user_input) > JUDGE_INPUT_MAX_CHARS:
            user_input = user_input[:JUDGE_INPUT_MAX_CHARS]

        parts = []

        # System prompt with nonce
        system_content = JUDGE_SYSTEM_PROMPT
        if nonce is not None:
            system_content = "NONCE: " + nonce + "\n\n" + system_content
        parts.append("### System:\n" + system_content)

        # Few-shot examples
        if self.use_few_shot:
            examples = _patch_few_shot_nonce(FEW_SHOT_EXAMPLES, nonce)
            for msg in examples:
                role = msg["role"].capitalize()
                parts.append("### {}:\n{}".format(role, msg["content"]))

        # User input
        wrapped = "<INPUT>\n" + user_input + "\n</INPUT>"
        parts.append("### User:\n" + wrapped)
        parts.append("### Assistant:\n")

        return "\n\n".join(parts)

    def _verify_nonce(self, content: str, expected_nonce: str) -> bool:
        """Return True if the nonce JSON field matches expected_nonce."""
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                return data.get("nonce", "") == expected_nonce
        except (json.JSONDecodeError, ValueError):
            pass
        return False

    def _parse_response(self, content: str, latency_ms: float) -> JudgeVerdict:
        start_idx = content.find("{")
        end_idx = content.rfind("}")

        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return JudgeVerdict(
                verdict="UNKNOWN",
                confidence=0.0,
                reasoning="Non-JSON response from local judge",
                latency_ms=latency_ms,
                model=self.model,
                error="parse_failure_no_json",
            )

        json_str = content[start_idx:end_idx + 1]
        try:
            data = json.loads(json_str)
            verdict = str(data.get("verdict", "UNKNOWN")).upper().strip()
            if verdict not in ("SAFE", "MALICIOUS"):
                verdict = "UNKNOWN"
            raw_conf = float(data.get("confidence", 0.5))
            if math.isnan(raw_conf) or math.isinf(raw_conf):
                raw_conf = 0.5
            confidence = max(0.0, min(1.0, raw_conf))
            reasoning = _CONTROL_RE.sub("", str(data.get("reasoning", ""))).strip()[:500]
            return JudgeVerdict(
                verdict=verdict,
                confidence=confidence,
                reasoning=reasoning,
                latency_ms=latency_ms,
                model=self.model,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            return JudgeVerdict(
                verdict="UNKNOWN",
                confidence=0.0,
                reasoning="JSON parse error",
                latency_ms=latency_ms,
                model=self.model,
                error="parse_failure: {}".format(type(exc).__name__),
            )
