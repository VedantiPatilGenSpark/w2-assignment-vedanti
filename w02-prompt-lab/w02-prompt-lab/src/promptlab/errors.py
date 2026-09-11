"""Errors used by the Week 2 local model lab."""


class UnknownModelError(ValueError):
    """Raised when a model identifier is not present in the configured model table."""


class TransientProviderError(Exception):
    """Raised when a provider call failed for a temporary reason and may be retried."""


class PermanentProviderError(Exception):
    """Raised when a provider call failed in a way that should not be retried."""


class TruncatedResponseError(Exception):
    """Raised when the model stopped because it hit the output token limit."""
