"""Ollama completion adapter."""

from __future__ import annotations

import random
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.config import Settings
from promptlab.errors import (
    PermanentProviderError,
    TransientProviderError,
    TruncatedResponseError,
    UnknownModelError,
)
from promptlab.usage import CallRecord, compute_cost

REQUEST_TIMEOUT_SECONDS = 180.0
_PERMANENT_STATUS_CODES = {400, 404, 422}


class OllamaAdapter:
    """One adapter class; Mistral vs Qwen is chosen by model_id at construction.

    think is Ollama-specific: None omits the field (provider default), True/False sends it.
    """

    provider = "ollama"

    def __init__(self, model_id: str, *, think: bool | None = None) -> None:
        known_ids = {config.model_id for config in Settings.from_env().models.values()}
        if model_id not in known_ids:
            raise UnknownModelError(model_id)
        self.model_id = model_id
        self.think = think
        self._settings = Settings.from_env()

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        records: list[CallRecord] = []
        max_attempts = min(1 + self._settings.max_retries, 3)
        last_text: str | None = None
        last_error: str | None = None
        succeeded = False
        url = f"{self._settings.ollama_base_url}/api/generate"

        for attempt in range(1, max_attempts + 1):
            started = time.perf_counter()
            text: str | None = None
            input_tokens = 0
            output_tokens = 0
            stop_reason: str | None = None
            try:
                try:
                    response = httpx.post(
                        url,
                        json=self._generate_body(request),
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    raise TransientProviderError(str(exc)) from exc

                status_code = getattr(response, "status_code", None)
                if status_code in _PERMANENT_STATUS_CODES:
                    raise PermanentProviderError(f"Ollama returned HTTP {status_code}")
                if isinstance(status_code, int) and status_code >= 500:
                    raise TransientProviderError(f"Ollama returned HTTP {status_code}")
                if isinstance(status_code, int) and status_code >= 400:
                    raise PermanentProviderError(f"Ollama returned HTTP {status_code}")

                try:
                    payload = response.json()
                except Exception as exc:
                    raise PermanentProviderError("Ollama returned an unparseable body") from exc
                if not isinstance(payload, dict):
                    raise PermanentProviderError("Ollama returned an unparseable body")

                text = _response_text(payload)
                try:
                    input_tokens = _require_int(payload, "prompt_eval_count")
                    output_tokens = _require_int(payload, "eval_count")
                    stop_reason = _optional_str(payload, "done_reason")
                except (TypeError, KeyError) as exc:
                    raise PermanentProviderError("Ollama usage fields were malformed") from exc

                if stop_reason == "length":
                    raise TruncatedResponseError("Response truncated by output-token ceiling")

                latency_ms = _elapsed_ms(started)
                succeeded = True
                last_error = None
                last_text = text
                records.append(
                    self._record(
                        request=request,
                        run_id=run_id,
                        attempt=attempt,
                        latency_ms=latency_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        stop_reason=stop_reason,
                        error_type=None,
                        response_text=text,
                    )
                )
                break

            except TruncatedResponseError as exc:
                latency_ms = _elapsed_ms(started)
                last_error = type(exc).__name__
                last_text = text
                records.append(
                    self._record(
                        request=request,
                        run_id=run_id,
                        attempt=attempt,
                        latency_ms=latency_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        stop_reason=stop_reason,
                        error_type=last_error,
                        response_text=last_text,
                    )
                )
                break

            except PermanentProviderError:
                latency_ms = _elapsed_ms(started)
                last_error = PermanentProviderError.__name__
                records.append(
                    self._record(
                        request=request,
                        run_id=run_id,
                        attempt=attempt,
                        latency_ms=latency_ms,
                        input_tokens=0,
                        output_tokens=0,
                        stop_reason=None,
                        error_type=last_error,
                        response_text=None,
                    )
                )
                break

            except TransientProviderError:
                latency_ms = _elapsed_ms(started)
                last_error = TransientProviderError.__name__
                records.append(
                    self._record(
                        request=request,
                        run_id=run_id,
                        attempt=attempt,
                        latency_ms=latency_ms,
                        input_tokens=0,
                        output_tokens=0,
                        stop_reason=None,
                        error_type=last_error,
                        response_text=None,
                    )
                )
                if attempt < max_attempts:
                    _sleep_before_retry(attempt)
                    continue
                break

        return CompletionResult(
            succeeded=succeeded,
            text=last_text,
            error_type=last_error,
            records=records,
        )

    def _record(
        self,
        *,
        request: CompletionRequest,
        run_id: str,
        attempt: int,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
        stop_reason: str | None,
        error_type: str | None,
        response_text: str | None,
    ) -> CallRecord:
        return CallRecord(
            record_id=str(uuid.uuid4()),
            run_id=run_id,
            timestamp=datetime.now(UTC),
            provider="ollama",
            model_id=self.model_id,
            task=request.task,
            case_id=request.case_id,
            prompt_id=request.prompt_id,
            prompt_version=request.prompt_version,
            attempt=attempt,
            temperature=request.temperature,
            max_output_tokens=request.max_output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=None,
            latency_ms=latency_ms,
            cost_usd=compute_cost(self.model_id, input_tokens, output_tokens),
            stop_reason=stop_reason,
            error_type=error_type,
            response_text=response_text,
        )

    def _generate_body(self, request: CompletionRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_id,
            "prompt": _prompt_text(request),
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }
        if self.think is not None:
            body["think"] = self.think
        return body


def _prompt_text(request: CompletionRequest) -> str:
    if request.system:
        return f"{request.system}\n\n{request.user_content}"
    return request.user_content


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _sleep_before_retry(attempt: int) -> None:
    delay = (0.5 * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
    time.sleep(delay)


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload[key]
    if not isinstance(value, int):
        raise TypeError(f"Ollama field {key!r} must be an int")
    return value


def _optional_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Ollama field {key!r} must be a string")
    return value


def _response_text(payload: dict[str, Any]) -> str | None:
    response = payload.get("response")
    if isinstance(response, str):
        return response
    message = payload.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return None
