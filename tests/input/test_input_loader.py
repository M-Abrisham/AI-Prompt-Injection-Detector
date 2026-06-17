import os
import stat
import sys
import tempfile
import pathlib
import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO


from na0s.layer0.input_loader import (
    load_input,
    InputLoadError,
    _is_url,
    _is_file_path,
    _validate_file_path,
    _validate_path_string,
    _validate_fd,
    _safe_open_file,
    MAX_FILE_SIZE,
)


class TestTextPassthrough(unittest.TestCase):
    """Plain string inputs should pass through unchanged."""

    def test_simple_text(self):
        content, meta = load_input("Hello, world!")
        self.assertEqual(content, "Hello, world!")
        self.assertEqual(meta["source_type"], "text")
        self.assertIsNone(meta["source_path"])
        self.assertIsNone(meta["content_type"])
        self.assertIsNone(meta["file_size"])

    def test_empty_string(self):
        content, meta = load_input("")
        self.assertEqual(content, "")
        self.assertEqual(meta["source_type"], "text")

    def test_multiline_text(self):
        text = "Line 1\nLine 2\nLine 3"
        content, meta = load_input(text)
        self.assertEqual(content, text)
        self.assertEqual(meta["source_type"], "text")

    def test_unicode_text(self):
        text = "Hello \u4e16\u754c \U0001F600"
        content, meta = load_input(text)
        self.assertEqual(content, text)
        self.assertEqual(meta["source_type"], "text")

    def test_text_that_looks_like_path_but_doesnt_exist(self):
        content, meta = load_input("/nonexistent/path/to/file.txt")
        self.assertEqual(content, "/nonexistent/path/to/file.txt")
        self.assertEqual(meta["source_type"], "text")


class TestBytesPassthrough(unittest.TestCase):
    """Bytes inputs should pass through unchanged."""

    def test_simple_bytes(self):
        data = b"Hello, bytes!"
        content, meta = load_input(data)
        self.assertEqual(content, data)
        self.assertEqual(meta["source_type"], "bytes")
        self.assertIsNone(meta["source_path"])
        self.assertEqual(meta["file_size"], len(data))

    def test_empty_bytes(self):
        content, meta = load_input(b"")
        self.assertEqual(content, b"")
        self.assertEqual(meta["source_type"], "bytes")
        self.assertEqual(meta["file_size"], 0)

    def test_bytearray(self):
        data = bytearray(b"Hello, bytearray!")
        content, meta = load_input(data)
        self.assertEqual(content, data)
        self.assertEqual(meta["source_type"], "bytes")

    def test_binary_content(self):
        data = bytes(range(256))
        content, meta = load_input(data)
        self.assertEqual(content, data)
        self.assertEqual(meta["file_size"], 256)


class TestFileLoading(unittest.TestCase):
    """File path inputs should read file content."""

    def test_load_text_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("File content here")
            f.flush()
            filepath = f.name

        try:
            content, meta = load_input(filepath)
            self.assertEqual(content, b"File content here")
            self.assertEqual(meta["source_type"], "file")
            self.assertEqual(meta["source_path"], filepath)
            self.assertIn("text/plain", meta["content_type"] or "")
            self.assertEqual(meta["file_size"], 17)
        finally:
            os.unlink(filepath)

    def test_load_binary_file(self):
        data = bytes(range(256))
        with tempfile.NamedTemporaryFile(
            suffix=".bin", delete=False
        ) as f:
            f.write(data)
            f.flush()
            filepath = f.name

        try:
            content, meta = load_input(filepath)
            self.assertEqual(content, data)
            self.assertEqual(meta["source_type"], "file")
            self.assertEqual(meta["file_size"], 256)
        finally:
            os.unlink(filepath)

    def test_load_pathlib_path(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("pathlib content")
            f.flush()
            filepath = pathlib.Path(f.name)

        try:
            content, meta = load_input(filepath)
            self.assertEqual(content, b"pathlib content")
            self.assertEqual(meta["source_type"], "file")
        finally:
            os.unlink(str(filepath))

    def test_pathlib_nonexistent_raises(self):
        fake_path = pathlib.Path("/tmp/nonexistent_test_file_12345.txt")
        with self.assertRaises(InputLoadError) as ctx:
            load_input(fake_path)
        self.assertIn("File not found", str(ctx.exception))

    def test_load_empty_file(self):
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as f:
            filepath = f.name

        try:
            content, meta = load_input(filepath)
            self.assertEqual(content, b"")
            self.assertEqual(meta["source_type"], "file")
            self.assertEqual(meta["file_size"], 0)
        finally:
            os.unlink(filepath)


class TestFileSizeLimit(unittest.TestCase):
    """File size limit should be enforced."""

    def test_oversized_file_rejected(self):
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as f:
            filepath = f.name
            # Write a file slightly over the limit
            # We patch MAX_FILE_SIZE to a small value for testing
            f.write(b"x" * 1025)
            f.flush()

        try:
            import na0s.layer0.input_loader as loader
            orig = loader.MAX_FILE_SIZE
            loader.MAX_FILE_SIZE = 1024
            try:
                with self.assertRaises(InputLoadError) as ctx:
                    load_input(filepath)
                self.assertIn("exceeds limit", str(ctx.exception))
            finally:
                loader.MAX_FILE_SIZE = orig
        finally:
            os.unlink(filepath)

    def test_file_at_limit_passes(self):
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as f:
            filepath = f.name
            f.write(b"x" * 1024)
            f.flush()

        try:
            import na0s.layer0.input_loader as loader
            orig = loader.MAX_FILE_SIZE
            loader.MAX_FILE_SIZE = 1024
            try:
                content, meta = load_input(filepath)
                self.assertEqual(len(content), 1024)
            finally:
                loader.MAX_FILE_SIZE = orig
        finally:
            os.unlink(filepath)


class TestDirectoryTraversalRejection(unittest.TestCase):
    """Directory traversal attempts should be rejected."""

    def test_dotdot_in_path_direct(self):
        """Test _validate_file_path directly to verify .. rejection."""
        # _validate_file_path should reject any path containing ".."
        # regardless of whether it resolves to a real file.
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as f:
            f.write(b"test")
            f.flush()
            filepath = f.name

        try:
            dirpart = os.path.dirname(filepath)
            basename = os.path.basename(filepath)
            # Create the subdir so the path actually exists
            subdir = os.path.join(dirpart, "test_subdir_l0")
            os.makedirs(subdir, exist_ok=True)
            try:
                traversal_path = os.path.join(
                    dirpart, "test_subdir_l0", "..", basename
                )
                with self.assertRaises(InputLoadError) as ctx:
                    _validate_file_path(traversal_path)
                self.assertIn("Directory traversal", str(ctx.exception))
            finally:
                os.rmdir(subdir)
        finally:
            os.unlink(filepath)

    def test_dotdot_via_load_input(self):
        """Traversal path that exists on disk should still be rejected."""
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as f:
            f.write(b"test")
            f.flush()
            filepath = f.name

        try:
            dirpart = os.path.dirname(filepath)
            basename = os.path.basename(filepath)
            subdir = os.path.join(dirpart, "test_subdir_l0_2")
            os.makedirs(subdir, exist_ok=True)
            try:
                traversal_path = os.path.join(
                    dirpart, "test_subdir_l0_2", "..", basename
                )
                # Path with .. should be rejected even if it resolves to a real file
                with self.assertRaises(InputLoadError) as ctx:
                    load_input(traversal_path)
                self.assertIn("Directory traversal", str(ctx.exception))
            finally:
                os.rmdir(subdir)
        finally:
            os.unlink(filepath)

    def test_directory_rejected(self):
        tmpdir = tempfile.mkdtemp()
        try:
            with self.assertRaises(InputLoadError) as ctx:
                load_input(tmpdir)
            self.assertIn("directory", str(ctx.exception).lower())
        finally:
            os.rmdir(tmpdir)


class TestSymlinkRejection(unittest.TestCase):
    """Symlinks should be rejected."""

    def test_symlink_rejected(self):
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as f:
            f.write(b"real content")
            f.flush()
            real_path = f.name

        link_path = real_path + ".link"
        try:
            os.symlink(real_path, link_path)
            with self.assertRaises(InputLoadError) as ctx:
                load_input(link_path)
            self.assertIn("Symlink", str(ctx.exception))
        finally:
            os.unlink(link_path)
            os.unlink(real_path)


def _mock_opener(mock_resp):
    """Create a mock opener whose open() returns mock_resp as a context manager."""
    mock_opener = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_opener.open.return_value = mock_resp
    return mock_opener


class TestURLLoading(unittest.TestCase):
    """URL inputs should be fetched with urllib."""

    @patch("na0s.layer0.input_loader._validate_url_target")
    @patch("na0s.layer0.input_loader._build_safe_opener")
    def test_https_url_success(self, mock_build, mock_ssrf):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<html>Hello</html>"
        mock_resp.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_build.return_value = _mock_opener(mock_resp)

        content, meta = load_input("https://example.com/page")
        self.assertEqual(content, b"<html>Hello</html>")
        self.assertEqual(meta["source_type"], "url")
        self.assertEqual(meta["source_path"], "https://example.com/page")
        self.assertEqual(meta["content_type"], "text/html; charset=utf-8")
        self.assertEqual(meta["file_size"], 18)

    def test_http_rejected_by_default(self):
        with self.assertRaises(InputLoadError) as ctx:
            load_input("http://example.com/page")
        self.assertIn("HTTPS only", str(ctx.exception))

    @patch("na0s.layer0.input_loader.HTTPS_ONLY", False)
    @patch("na0s.layer0.input_loader._validate_url_target")
    @patch("na0s.layer0.input_loader._build_safe_opener")
    def test_http_allowed_when_configured(self, mock_build, mock_ssrf):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"content"
        mock_resp.headers = {"Content-Type": "text/plain"}
        mock_build.return_value = _mock_opener(mock_resp)

        content, meta = load_input("http://example.com/page")
        self.assertEqual(content, b"content")
        self.assertEqual(meta["source_type"], "url")

    @patch("na0s.layer0.input_loader._validate_url_target")
    @patch("na0s.layer0.input_loader._build_safe_opener")
    def test_url_fetch_failure(self, mock_build, mock_ssrf):
        import urllib.error
        mock_opener = MagicMock()
        mock_opener.open.side_effect = urllib.error.URLError("Connection refused")
        mock_build.return_value = mock_opener

        with self.assertRaises(InputLoadError) as ctx:
            load_input("https://example.com/fail")
        self.assertIn("URL fetch failed", str(ctx.exception))

    @patch("na0s.layer0.input_loader._validate_url_target")
    @patch("na0s.layer0.input_loader._build_safe_opener")
    def test_url_oversized_response(self, mock_build, mock_ssrf):
        import na0s.layer0.input_loader as loader
        orig = loader.MAX_URL_RESPONSE_SIZE
        loader.MAX_URL_RESPONSE_SIZE = 100

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"x" * 101
        mock_resp.headers = {"Content-Type": "text/plain"}
        mock_build.return_value = _mock_opener(mock_resp)

        try:
            with self.assertRaises(InputLoadError) as ctx:
                load_input("https://example.com/big")
            self.assertIn("byte limit", str(ctx.exception))
        finally:
            loader.MAX_URL_RESPONSE_SIZE = orig


class TestHelperFunctions(unittest.TestCase):
    """Test internal helper functions."""

    def test_is_url_https(self):
        self.assertTrue(_is_url("https://example.com"))

    def test_is_url_http(self):
        self.assertTrue(_is_url("http://example.com"))

    def test_is_url_not_url(self):
        self.assertFalse(_is_url("just some text"))
        self.assertFalse(_is_url("/path/to/file"))
        self.assertFalse(_is_url(123))

    def test_is_file_path_real_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filepath = f.name
        try:
            self.assertTrue(_is_file_path(filepath))
        finally:
            os.unlink(filepath)

    def test_is_file_path_nonexistent(self):
        self.assertFalse(_is_file_path("/nonexistent/file.txt"))

    def test_is_file_path_url_not_treated_as_path(self):
        self.assertFalse(_is_file_path("https://example.com"))


class TestUnsupportedTypes(unittest.TestCase):
    """Unsupported types should pass through with 'unknown' metadata."""

    def test_integer_passthrough(self):
        content, meta = load_input(12345)
        self.assertEqual(content, 12345)
        self.assertEqual(meta["source_type"], "unknown")

    def test_none_passthrough(self):
        content, meta = load_input(None)
        self.assertIsNone(content)
        self.assertEqual(meta["source_type"], "unknown")

    def test_list_passthrough(self):
        content, meta = load_input([1, 2, 3])
        self.assertEqual(content, [1, 2, 3])
        self.assertEqual(meta["source_type"], "unknown")


class TestIntegrationWithSanitizer(unittest.TestCase):
    """End-to-end: file path -> layer0_sanitize -> Layer0Result."""

    def test_file_through_sanitizer(self):
        from na0s.layer0 import layer0_sanitize

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("This is a normal test input for sanitization.")
            f.flush()
            filepath = f.name

        try:
            result = layer0_sanitize(filepath)
            self.assertFalse(result.rejected)
            self.assertIn("sanitiz", result.sanitized_text.lower())
            self.assertEqual(result.source_metadata["source_type"], "file")
            self.assertIn("input_loaded_from_file", result.anomaly_flags)
        finally:
            os.unlink(filepath)

    def test_plain_text_no_source_metadata(self):
        from na0s.layer0 import layer0_sanitize

        result = layer0_sanitize("Hello world, this is a test.")
        self.assertFalse(result.rejected)
        # Plain text should not have loading-related source_metadata
        # (language detection metadata is expected from Step 5)
        self.assertNotIn("source_type", result.source_metadata)

    @patch("na0s.layer0.input_loader._validate_url_target")
    @patch("na0s.layer0.input_loader._build_safe_opener")
    def test_url_through_sanitizer(self, mock_build, mock_ssrf):
        from na0s.layer0 import layer0_sanitize

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"URL content for testing purposes."
        mock_resp.headers = {"Content-Type": "text/plain"}
        mock_build.return_value = _mock_opener(mock_resp)

        result = layer0_sanitize("https://example.com/test")
        self.assertFalse(result.rejected)
        self.assertEqual(result.source_metadata["source_type"], "url")
        self.assertIn("input_loaded_from_url", result.anomaly_flags)

    def test_bytes_still_works(self):
        from na0s.layer0 import layer0_sanitize

        result = layer0_sanitize(b"Bytes input for testing.")
        self.assertFalse(result.rejected)
        self.assertIn("testing", result.sanitized_text.lower())


class TestTOCTOUSafeFileOpen(unittest.TestCase):
    """TOCTOU prevention: atomic open with O_NOFOLLOW + fd-based validation."""

    def test_normal_file_loads(self):
        """Regular file still loads correctly via the atomic fd path."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("toctou safe content")
            f.flush()
            filepath = f.name

        try:
            content, meta = load_input(filepath)
            self.assertEqual(content, b"toctou safe content")
            self.assertEqual(meta["source_type"], "file")
        finally:
            os.unlink(filepath)

    def test_symlink_rejected_at_open_time(self):
        """Symlink rejected via O_NOFOLLOW in a single syscall (no race window)."""
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False
        ) as f:
            f.write(b"secret data")
            f.flush()
            real_path = f.name

        link_path = real_path + ".toctou_link"
        try:
            os.symlink(real_path, link_path)
            with self.assertRaises(InputLoadError) as ctx:
                _safe_open_file(link_path)
            self.assertIn("Symlink", str(ctx.exception))
        finally:
            if os.path.islink(link_path):
                os.unlink(link_path)
            os.unlink(real_path)

    def test_pathlib_nonexistent_no_precheck(self):
        """pathlib.Path to nonexistent file raises InputLoadError via EAFP."""
        fake_path = pathlib.Path("/tmp/nonexistent_toctou_test_12345.txt")
        with self.assertRaises(InputLoadError) as ctx:
            load_input(fake_path)
        self.assertIn("File not found", str(ctx.exception))

    def test_path_traversal_still_blocked(self):
        """Directory traversal is caught by string-only check (no FS call)."""
        with self.assertRaises(InputLoadError) as ctx:
            _validate_path_string("/tmp/safe/../../../etc/passwd")
        self.assertIn("Directory traversal", str(ctx.exception))

    def test_validate_fd_rejects_directory(self):
        """_validate_fd rejects an fd pointing to a directory."""
        tmpdir = tempfile.mkdtemp()
        try:
            fd = os.open(tmpdir, os.O_RDONLY)
            try:
                with self.assertRaises(InputLoadError) as ctx:
                    _validate_fd(fd, tmpdir)
                self.assertIn("directory", str(ctx.exception).lower())
            finally:
                os.close(fd)
        finally:
            os.rmdir(tmpdir)

    def test_validate_fd_accepts_regular_file(self):
        """_validate_fd returns stat result for a regular file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test")
            filepath = f.name

        try:
            fd = os.open(filepath, os.O_RDONLY)
            try:
                st = _validate_fd(fd, filepath)
                self.assertTrue(stat.S_ISREG(st.st_mode))
            finally:
                os.close(fd)
        finally:
            os.unlink(filepath)

    def test_safe_open_file_returns_fd(self):
        """_safe_open_file returns a valid fd for a regular file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("fd test content")
            filepath = f.name

        try:
            fd = _safe_open_file(filepath)
            try:
                content = os.read(fd, 1024)
                self.assertEqual(content, b"fd test content")
            finally:
                os.close(fd)
        finally:
            os.unlink(filepath)

    def test_safe_open_file_nonexistent(self):
        """_safe_open_file raises InputLoadError for nonexistent files."""
        with self.assertRaises(InputLoadError) as ctx:
            _safe_open_file("/tmp/nonexistent_safe_open_12345.txt")
        self.assertIn("File not found", str(ctx.exception))

    def test_directory_rejected_via_load_input(self):
        """Directories are rejected through the fd-based validation path."""
        tmpdir = tempfile.mkdtemp()
        try:
            with self.assertRaises(InputLoadError) as ctx:
                load_input(tmpdir)
            self.assertIn("directory", str(ctx.exception).lower())
        finally:
            os.rmdir(tmpdir)

    def test_single_fd_flow_no_reopen(self):
        """Verify the entire load path uses one fd (no path re-open)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write("single fd test")
            filepath = f.name

        try:
            # Patch os.open to count how many times it's called for our file
            original_open = os.open
            open_count = [0]

            def counting_open(path, flags, *args, **kwargs):
                if path == filepath:
                    open_count[0] += 1
                return original_open(path, flags, *args, **kwargs)

            with patch("na0s.layer0.input_loader.os.open", side_effect=counting_open):
                content, meta = load_input(filepath)

            self.assertEqual(content, b"single fd test")
            # _load_file should only call os.open ONCE (the atomic open)
            self.assertEqual(open_count[0], 1,
                             "Expected exactly 1 os.open call (atomic), got {}".format(open_count[0]))
        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    unittest.main()
