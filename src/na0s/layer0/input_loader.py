"""Unified input loader for Layer 0.

Accepts str, bytes, file paths, and URLs. Returns normalized content
with source metadata for downstream processing.

TOCTOU Prevention (CWE-367)
---------------------------
File operations use the EAFP (Easier to Ask Forgiveness than Permission)
pattern.  Instead of check-then-open (which has a race window), we:

1. Open the file atomically with ``os.open(O_NOFOLLOW)`` to reject
   symlinks in the same syscall that opens the file.
2. Validate the *opened fd* via ``os.fstat()`` so there is zero gap
   between the security check and the file use.
3. Read from the fd directly, never re-opening the path.
"""

import errno
import ipaddress
import os
import socket
import stat
import pathlib
import urllib.parse
import urllib.request
import urllib.error
import mimetypes

# Max file size: 10 MB by default, configurable via env
from na0s._env import safe_int_env

MAX_FILE_SIZE = safe_int_env("L0_MAX_FILE_SIZE", 10 * 1024 * 1024, lo=1)

# URL fetch settings
URL_TIMEOUT = safe_int_env("L0_URL_TIMEOUT", 10, lo=1)
MAX_URL_RESPONSE_SIZE = safe_int_env("L0_MAX_URL_RESPONSE_SIZE", 10 * 1024 * 1024, lo=1)
HTTPS_ONLY = os.getenv("L0_HTTPS_ONLY", "1") != "0"
MAX_REDIRECTS = safe_int_env("L0_MAX_REDIRECTS", 5, lo=0)


class InputLoadError(Exception):
    """Raised when input loading fails due to security or I/O issues."""


def _is_url(source):
    """Check if source string looks like a URL."""
    if not isinstance(source, str):
        return False
    return source.startswith("http://") or source.startswith("https://")


def _is_file_path(source):
    """Check if source string is an existing file path.

    Only returns True for strings that exist on disk. This avoids
    accidentally treating normal text input as a file path.
    """
    if not isinstance(source, str):
        return False
    if _is_url(source):
        return False
    return os.path.exists(source)


def _validate_path_string(filepath):
    """Pure string-based security check for file paths (no filesystem calls).

    Rejects directory traversal (.. components in the raw path).
    TOCTOU-safe because it only inspects the string, not the filesystem.
    """
    parts = pathlib.PurePath(filepath).parts
    if ".." in parts:
        raise InputLoadError(
            "Directory traversal detected in path: {}".format(filepath)
        )


def _validate_fd(fd, filepath):
    """Validate an already-opened file descriptor.

    Uses os.fstat on the *open fd* so the check is atomic -- no window
    between the check and the use because we already hold the fd.
    Rejects directories, device files, FIFOs, and sockets.
    """
    try:
        st = os.fstat(fd)
    except OSError as exc:
        raise InputLoadError("Cannot stat file: {}".format(exc))

    mode = st.st_mode
    if stat.S_ISDIR(mode):
        raise InputLoadError("Path is a directory: {}".format(filepath))
    if stat.S_ISCHR(mode) or stat.S_ISBLK(mode):
        raise InputLoadError("Path is a device file: {}".format(filepath))
    if stat.S_ISFIFO(mode):
        raise InputLoadError("Path is a FIFO: {}".format(filepath))
    if stat.S_ISSOCK(mode):
        raise InputLoadError("Path is a socket: {}".format(filepath))

    return st


def _safe_open_file(filepath):
    """Atomically open *filepath* for reading, rejecting symlinks.

    Uses os.open with O_NOFOLLOW (where available) so the symlink check
    and the file open happen in a single syscall -- eliminating the TOCTOU
    race (CWE-367).

    On platforms without O_NOFOLLOW (Windows), falls back to os.lstat
    immediately before os.open to minimise the window.

    Returns an open file descriptor (caller must close it).
    """
    flags = os.O_RDONLY
    o_nofollow = getattr(os, "O_NOFOLLOW", 0)

    if o_nofollow:
        flags |= o_nofollow
    else:
        # Fallback (Windows): lstat right before open to minimise gap
        try:
            pre_st = os.lstat(filepath)
        except OSError as exc:
            raise InputLoadError("Cannot stat file: {}".format(exc))
        if stat.S_ISLNK(pre_st.st_mode):
            target = os.path.realpath(filepath)
            raise InputLoadError(
                "Symlink detected: {} -> {}".format(filepath, target)
            )

    try:
        fd = os.open(filepath, flags)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, getattr(errno, "EMLINK", -1)):
            target = os.path.realpath(filepath)
            raise InputLoadError(
                "Symlink detected: {} -> {}".format(filepath, target)
            )
        if exc.errno == errno.ENOENT:
            raise InputLoadError("File not found: {}".format(filepath))
        if exc.errno == errno.EACCES and o_nofollow and os.path.islink(filepath):
            # macOS returns EACCES for symlink + O_NOFOLLOW
            target = os.path.realpath(filepath)
            raise InputLoadError(
                "Symlink detected: {} -> {}".format(filepath, target)
            )
        raise InputLoadError("Cannot open file: {}".format(exc))

    return fd


def _validate_file_path(filepath):
    """Security checks for file paths -- backward-compatible wrapper.

    Performs string-based traversal check, then atomically opens and
    validates the file descriptor to reject symlinks and special files.
    Returns the resolved path (closing the test fd immediately).
    """
    _validate_path_string(filepath)

    fd = _safe_open_file(filepath)
    try:
        _validate_fd(fd, filepath)
        resolved = os.path.realpath(filepath)
    finally:
        os.close(fd)

    return resolved


def _load_file(filepath):
    """Read file content with security and size checks.

    TOCTOU-safe: opens the file atomically with O_NOFOLLOW, validates
    the fd, and reads from the same fd -- no re-open, no race window.

    Returns (content_bytes, metadata_dict).
    """
    # String-based check (no filesystem call, no TOCTOU risk)
    _validate_path_string(filepath)

    # Atomic open: rejects symlinks in a single syscall
    fd = _safe_open_file(filepath)
    try:
        # Validate the *opened* fd (not the path) -- no TOCTOU gap
        st = _validate_fd(fd, filepath)

        # Size check on the opened fd
        file_size = st.st_size
        if file_size > MAX_FILE_SIZE:
            raise InputLoadError(
                "File size {} exceeds limit {} bytes".format(
                    file_size, MAX_FILE_SIZE
                )
            )

        # Read from the fd directly -- never re-open the path
        content = os.read(fd, MAX_FILE_SIZE + 1)
        if len(content) > MAX_FILE_SIZE:
            raise InputLoadError(
                "File size {} exceeds limit {} bytes".format(
                    len(content), MAX_FILE_SIZE
                )
            )
    finally:
        os.close(fd)

    # Detect content type from extension (best effort, stdlib only)
    content_type, _ = mimetypes.guess_type(filepath)

    metadata = {
        "source_type": "file",
        "source_path": filepath,
        "content_type": content_type,
        "file_size": len(content),
    }
    return content, metadata


def _is_private_ip(addr):
    """Check if an IP address is private, loopback, link-local, or reserved."""
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return True  # unparseable = reject (fail-closed)
    return (
        ip.is_unspecified or ip.is_loopback or ip.is_private
        or ip.is_link_local or ip.is_reserved or ip.is_multicast
    )


def _validate_url_target(url):
    """Resolve hostname and reject URLs targeting private/internal IPs (SSRF protection)."""
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise InputLoadError("SSRF protection: could not extract hostname from URL")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        addr_infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise InputLoadError(
            "SSRF protection: DNS resolution failed for {}: {}".format(hostname, exc)
        )

    if not addr_infos:
        raise InputLoadError("SSRF protection: no DNS results for {}".format(hostname))

    for _family, _socktype, _proto, _canonname, sockaddr in addr_infos:
        resolved_ip = sockaddr[0]
        if _is_private_ip(resolved_ip):
            raise InputLoadError(
                "SSRF protection: URL resolves to blocked IP {} "
                "(private/internal/link-local address)".format(resolved_ip)
            )


def _validate_redirect_url(url):
    """Validate a redirect target URL for security."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise InputLoadError("Redirect to disallowed scheme: {}".format(parsed.scheme))
    if HTTPS_ONLY and parsed.scheme == "http":
        raise InputLoadError("Redirect to HTTP URL blocked (HTTPS only): {}".format(url))
    hostname = parsed.hostname
    if not hostname:
        raise InputLoadError("Redirect URL has no hostname: {}".format(url))
    # Reuse SSRF validation on redirect target
    _validate_url_target(url)


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validates each redirect hop: scheme, HTTPS-only, private IP, max count."""

    def __init__(self, max_redirects=None):
        super().__init__()
        self._max_redirects = max_redirects if max_redirects is not None else MAX_REDIRECTS
        self._redirect_count = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self._redirect_count += 1
        if self._redirect_count > self._max_redirects:
            raise InputLoadError(
                "Too many redirects ({} exceeded limit of {})".format(
                    self._redirect_count, self._max_redirects
                )
            )
        _validate_redirect_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _build_safe_opener():
    """Build a urllib opener with the safe redirect handler installed."""
    return urllib.request.build_opener(_SafeRedirectHandler())


def _load_url(url):
    """Fetch content from URL with security and size checks.

    Returns (content_bytes, metadata_dict).
    """
    if HTTPS_ONLY and url.startswith("http://"):
        raise InputLoadError(
            "HTTP URLs are not allowed (HTTPS only). "
            "Set L0_HTTPS_ONLY=0 to allow HTTP."
        )

    # SSRF protection: validate that URL does not target internal IPs
    _validate_url_target(url)

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "AI-Prompt-Injection-Detector/1.0"},
        )
        opener = _build_safe_opener()
        with opener.open(req, timeout=URL_TIMEOUT) as resp:
            content = resp.read(MAX_URL_RESPONSE_SIZE + 1)
            if len(content) > MAX_URL_RESPONSE_SIZE:
                raise InputLoadError(
                    "URL response exceeds {} byte limit".format(
                        MAX_URL_RESPONSE_SIZE
                    )
                )

            content_type = resp.headers.get("Content-Type", None)

            metadata = {
                "source_type": "url",
                "source_path": url,
                "content_type": content_type,
                "file_size": len(content),
            }
            return content, metadata

    except InputLoadError:
        raise
    except urllib.error.HTTPError as exc:
        raise InputLoadError(
            "URL returned HTTP {}: {}".format(exc.code, exc.reason)
        )
    except urllib.error.URLError as exc:
        raise InputLoadError("URL fetch failed: {}".format(exc))
    except OSError as exc:
        raise InputLoadError("Network error: {}".format(exc))


def load_input(source):
    """Unified input loader.

    Accepts:
        - str: plain text (passed through)
        - bytes: raw bytes (passed through)
        - pathlib.Path: file path (read from disk)
        - str that is an existing file path: read from disk
        - str starting with http:// or https://: fetch from URL

    Returns:
        (content, metadata) tuple where:
        - content is str or bytes
        - metadata is a dict with keys: source_type, source_path,
          content_type, file_size
    """
    # pathlib.Path -> always treat as file (EAFP: let _load_file handle
    # "not found" instead of a pre-check that creates a TOCTOU window)
    if isinstance(source, pathlib.Path):
        filepath = str(source)
        return _load_file(filepath)

    # bytes -> pass through
    if isinstance(source, (bytes, bytearray)):
        metadata = {
            "source_type": "bytes",
            "source_path": None,
            "content_type": None,
            "file_size": len(source),
        }
        return source, metadata

    # str -> check if URL, then file path, then plain text
    if isinstance(source, str):
        if _is_url(source):
            return _load_url(source)

        if _is_file_path(source):
            return _load_file(source)

        # Plain text pass-through
        metadata = {
            "source_type": "text",
            "source_path": None,
            "content_type": None,
            "file_size": None,
        }
        return source, metadata

    # Unsupported type -- let downstream validation handle it
    metadata = {
        "source_type": "unknown",
        "source_path": None,
        "content_type": None,
        "file_size": None,
    }
    return source, metadata
