"""Layer 16 exceptions."""  # LAYER16


class SessionNotFoundError(Exception):
    """Raised when a session_id does not exist in the store."""


class SessionExpiredError(Exception):
    """Raised when a session_id exists but has exceeded its TTL."""


class MaxSessionsReachedError(Exception):
    """Raised when the session store has hit the configured cap."""


# Alias used by some consumers
MaxSessionsExceededError = MaxSessionsReachedError
