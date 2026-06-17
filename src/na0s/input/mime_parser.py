"""MIME message parser for Layer 0.

Uses stdlib email.parser.BytesParser to extract text body and
attachments from email-format (MIME) inputs. Each attachment can
be individually passed through layer0_sanitize() for recursive
scanning.
"""

import email
import email.parser
import email.policy
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Attachment:
    """A single MIME attachment extracted from a message."""

    filename: Optional[str] = None
    content_type: str = "application/octet-stream"
    content: bytes = b""
    size: int = 0


@dataclass
class MIMEParseResult:
    """Result of parsing a MIME message."""

    body_text: str = ""
    attachments: List[Attachment] = field(default_factory=list)
    content_type: str = ""
    is_multipart: bool = False


def _looks_like_mime(raw_bytes):
    """Heuristic check: does this look like a MIME/email message?

    Checks for common MIME headers at the start of the content.
    This avoids trying to parse arbitrary text as MIME.
    """
    if not raw_bytes:
        return False

    # Decode just the first 2KB to check for MIME headers
    try:
        header_region = raw_bytes[:2048].decode("utf-8", errors="replace").lower()
    except Exception:
        return False

    mime_indicators = [
        "content-type:",
        "mime-version:",
        "from:",
        "to:",
        "subject:",
        "content-transfer-encoding:",
    ]

    # Need at least 2 MIME-like headers to consider it a MIME message
    matches = sum(1 for ind in mime_indicators if ind in header_region)
    return matches >= 2


def parse_mime_input(raw_bytes):
    """Parse a MIME/email message from raw bytes.

    Args:
        raw_bytes: Raw bytes that may contain a MIME message.

    Returns:
        MIMEParseResult with extracted body text, attachment list,
        content type, and multipart flag.
    """
    if not isinstance(raw_bytes, (bytes, bytearray)):
        raise TypeError(
            "parse_mime_input requires bytes, got {}".format(
                type(raw_bytes).__name__
            )
        )

    if not raw_bytes:
        return MIMEParseResult()

    # Parse the message using stdlib BytesParser
    parser = email.parser.BytesParser(policy=email.policy.default)
    msg = parser.parsebytes(raw_bytes)

    content_type = msg.get_content_type() or ""
    is_multipart = msg.is_multipart()

    body_parts = []
    attachments = []

    if is_multipart:
        for part in msg.walk():
            part_ct = part.get_content_type()
            part_disposition = str(part.get("Content-Disposition", ""))

            # Skip the multipart container itself
            if part.get_content_maintype() == "multipart":
                continue

            # Check if this part is an attachment
            if "attachment" in part_disposition or part.get_filename():
                payload = part.get_payload(decode=True)
                if payload is None:
                    payload = b""
                attachments.append(
                    Attachment(
                        filename=part.get_filename(),
                        content_type=part_ct,
                        content=payload,
                        size=len(payload),
                    )
                )
            elif part_ct == "text/plain":
                # Extract plain text body
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        body_parts.append(payload.decode(charset, errors="replace"))
                    except (LookupError, UnicodeDecodeError):
                        body_parts.append(payload.decode("utf-8", errors="replace"))
            elif part_ct == "text/html":
                # Extract HTML body (will be sanitized downstream by L0)
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        body_parts.append(payload.decode(charset, errors="replace"))
                    except (LookupError, UnicodeDecodeError):
                        body_parts.append(payload.decode("utf-8", errors="replace"))
            else:
                # Non-text, non-attachment inline content -> treat as attachment
                payload = part.get_payload(decode=True)
                if payload:
                    attachments.append(
                        Attachment(
                            filename=part.get_filename(),
                            content_type=part_ct,
                            content=payload,
                            size=len(payload),
                        )
                    )
    else:
        # Single-part message
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            try:
                body_parts.append(payload.decode(charset, errors="replace"))
            except (LookupError, UnicodeDecodeError):
                body_parts.append(payload.decode("utf-8", errors="replace"))

    body_text = "\n".join(body_parts)

    return MIMEParseResult(
        body_text=body_text,
        attachments=attachments,
        content_type=content_type,
        is_multipart=is_multipart,
    )
