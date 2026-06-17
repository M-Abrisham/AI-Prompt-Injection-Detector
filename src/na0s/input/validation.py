from na0s._env import safe_int_env

from .result import Layer0Result

MAX_INPUT_LENGTH = safe_int_env("L0_MAX_INPUT_CHARS", 50_000, lo=1)
MAX_INPUT_BYTES = safe_int_env("L0_MAX_INPUT_BYTES", 200_000, lo=1)


def validate_input(raw_input):
    """Fail-fast checks: type, emptiness, size limits.

    Returns a rejected Layer0Result if input is invalid, None if it passes.
    """
    # Type guard
    if not isinstance(raw_input, str):
        return Layer0Result(
            rejected=True,
            rejection_reason="input is not a string (got {})".format(
                type(raw_input).__name__
            ),
        )

    # Empty / whitespace-only guard
    if not raw_input or not raw_input.strip():
        return Layer0Result(
            rejected=True,
            rejection_reason="empty input",
            original_length=len(raw_input) if raw_input else 0,
        )

    # Character count limit
    if len(raw_input) > MAX_INPUT_LENGTH:
        return Layer0Result(
            rejected=True,
            rejection_reason="input exceeds {} char limit (got {})".format(
                MAX_INPUT_LENGTH, len(raw_input)
            ),
            original_length=len(raw_input),
        )

    # Byte size limit (catches multi-byte inflation attacks)
    # Use surrogatepass to handle lone surrogates without crashing
    byte_len = len(raw_input.encode("utf-8", errors="surrogatepass"))
    if byte_len > MAX_INPUT_BYTES:
        return Layer0Result(
            rejected=True,
            rejection_reason="input exceeds {} byte limit (got {})".format(
                MAX_INPUT_BYTES, byte_len
            ),
            original_length=len(raw_input),
        )

    # All checks passed
    return None
