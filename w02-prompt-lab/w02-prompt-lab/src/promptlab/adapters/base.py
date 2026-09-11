"""Shared completion types and adapter protocol."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel

from promptlab.usage import CallRecord


class CompletionRequest(BaseModel):
    """Inbound completion envelope from a caller."""

    task: Literal["triage", "summarization", "extraction"]
    case_id: str
    prompt_id: str
    prompt_version: str
    system: str
    user_content: str
    temperature: float
    max_output_tokens: int


class CompletionResult(BaseModel):
    """Outbound result of one complete() call, including every attempt."""

    succeeded: bool
    text: str | None
    error_type: str | None
    records: list[CallRecord]


class ModelAdapter(Protocol):
    """Structural interface for a completion adapter."""

    provider: str
    model_id: str

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        """Run one completion, including retries, and return every attempt."""
        ...
