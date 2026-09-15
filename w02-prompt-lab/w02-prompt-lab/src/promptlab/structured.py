"""Bounded structured-output path on top of a ModelAdapter."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from promptlab.adapters.base import CompletionRequest, ModelAdapter


class StructuredOutputError(Exception):
    """Raised when model output still fails the schema after allowed repairs."""


def complete_structured[T: BaseModel](
    adapter: ModelAdapter,
    request: CompletionRequest,
    schema: type[T],
    run_id: str,
    max_repairs: int = 1,
) -> T:
    """Call the adapter, validate JSON against schema, and repair at most once."""
    current = request
    repairs_used = 0
    last_error = "model output did not validate"

    while True:
        result = adapter.complete(current, run_id)
        try:
            payload = _parse_json_payload(result.text)
            return schema.model_validate(payload)
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
            if repairs_used >= max_repairs:
                raise StructuredOutputError(last_error) from exc
            repairs_used += 1
            current = request.model_copy(
                update={
                    "user_content": _repair_user_content(
                        original=request.user_content,
                        previous_text=result.text,
                        error=last_error,
                    )
                }
            )


def _parse_json_payload(text: str | None) -> Any:
    if text is None or not text.strip():
        raise ValueError("model returned empty text")
    raw = _strip_markdown_fence(text.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start : end + 1])
        raise


def _strip_markdown_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    without_open = text.split("\n", 1)[-1]
    if without_open.endswith("```"):
        without_open = without_open[: without_open.rfind("```")]
    return without_open.strip()


def _repair_user_content(*, original: str, previous_text: str | None, error: str) -> str:
    previous = previous_text if previous_text is not None else ""
    return (
        "The previous JSON did not pass validation.\n\n"
        f"Validation error:\n{error}\n\n"
        f"Previous output:\n{previous}\n\n"
        "Correct only what the validation error concerns. "
        "Return only a JSON object. Do not use Markdown.\n\n"
        f"Original instructions:\n{original}"
    )
