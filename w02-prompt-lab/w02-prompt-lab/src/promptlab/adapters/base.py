"""Shared completion types and adapter protocol."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


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
