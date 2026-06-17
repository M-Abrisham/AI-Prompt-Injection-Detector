"""Tests for Open Redirect bypass prevention in input_loader."""

import os
import sys
import socket
import unittest
from unittest.mock import patch, MagicMock
import urllib.request


from na0s.layer0.input_loader import (
    load_input,
    InputLoadError,
    _validate_redirect_url,
    _SafeRedirectHandler,
    _build_safe_opener,
)


class TestValidateRedirectUrl(unittest.TestCase):
    """Test _validate_redirect_url security checks."""

    @patch("na0s.layer0.input_loader.socket.getaddrinfo")
    def test_https_to_https_allowed(self, mock_dns):
        mock_dns.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
        _validate_redirect_url("https://example.com/page2")  # should not raise

    @patch("na0s.layer0.input_loader.HTTPS_ONLY", True)
    def test_https_to_http_blocked(self):
        with self.assertRaises(InputLoadError) as ctx:
            _validate_redirect_url("http://example.com/page2")
        self.assertIn("Redirect to HTTP URL blocked", str(ctx.exception))

    def test_file_scheme_blocked(self):
        with self.assertRaises(InputLoadError) as ctx:
            _validate_redirect_url("file:///etc/passwd")
        self.assertIn("disallowed scheme", str(ctx.exception))

    def test_ftp_scheme_blocked(self):
        with self.assertRaises(InputLoadError) as ctx:
            _validate_redirect_url("ftp://ftp.example.com/file")
        self.assertIn("disallowed scheme", str(ctx.exception))

    @patch("na0s.layer0.input_loader.socket.getaddrinfo")
    def test_redirect_to_private_ip_blocked(self, mock_dns):
        mock_dns.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", 443))
        ]
        with self.assertRaises(InputLoadError) as ctx:
            _validate_redirect_url("https://internal.corp/secret")
        self.assertIn("SSRF protection", str(ctx.exception))


class TestSafeRedirectHandler(unittest.TestCase):
    """Test the _SafeRedirectHandler class."""

    @patch("na0s.layer0.input_loader.socket.getaddrinfo")
    def test_redirect_count_enforced(self, mock_dns):
        mock_dns.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ]
        handler = _SafeRedirectHandler(max_redirects=2)
        mock_req = MagicMock()
        mock_fp = MagicMock()
        mock_headers = MagicMock()

        with patch.object(
            urllib.request.HTTPRedirectHandler, "redirect_request",
            return_value=mock_req,
        ):
            handler.redirect_request(
                mock_req, mock_fp, 301, "Moved",
                mock_headers, "https://example.com/r1"
            )
            handler.redirect_request(
                mock_req, mock_fp, 301, "Moved",
                mock_headers, "https://example.com/r2"
            )

        with self.assertRaises(InputLoadError) as ctx:
            handler.redirect_request(
                mock_req, mock_fp, 301, "Moved",
                mock_headers, "https://example.com/r3"
            )
        self.assertIn("Too many redirects", str(ctx.exception))

    @patch("na0s.layer0.input_loader.HTTPS_ONLY", True)
    def test_redirect_to_http_caught(self):
        handler = _SafeRedirectHandler()
        with self.assertRaises(InputLoadError) as ctx:
            handler.redirect_request(
                MagicMock(), MagicMock(), 302, "Found",
                MagicMock(), "http://evil.com/steal"
            )
        self.assertIn("Redirect to HTTP URL blocked", str(ctx.exception))

    @patch("na0s.layer0.input_loader.socket.getaddrinfo")
    def test_redirect_to_metadata_ip_caught(self, mock_dns):
        mock_dns.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443))
        ]
        handler = _SafeRedirectHandler()
        with self.assertRaises(InputLoadError) as ctx:
            handler.redirect_request(
                MagicMock(), MagicMock(), 302, "Found",
                MagicMock(), "https://169.254.169.254/latest/meta-data/"
            )
        self.assertIn("SSRF protection", str(ctx.exception))


class TestBuildSafeOpener(unittest.TestCase):

    def test_opener_has_redirect_handler(self):
        opener = _build_safe_opener()
        handler_types = [type(h) for h in opener.handlers]
        self.assertIn(_SafeRedirectHandler, handler_types)


if __name__ == "__main__":
    unittest.main()
