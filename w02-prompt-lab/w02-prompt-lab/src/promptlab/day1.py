"""Day 1 experiment: instrument Mistral extraction calls."""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from promptlab.config import PROJECT_ROOT, Settings
from promptlab.usage import CallRecord, append_record, compute_cost

CASE_IDS = ("E12", "E07", "E11")
PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "baseline.v0.md"
CASES_PATH = PROJECT_ROOT / "cases" / "extraction.jsonl"
NORMAL_NUM_PREDICT = 512
TRUNCATION_NUM_PREDICT = 8
TEMPERATURE = 0.0
REQUEST_TIMEOUT_SECONDS = 180.0


def load_extraction_cases(path: Path, case_ids: tuple[str, ...]) -> dict[str, str]:
    wanted = set(case_ids)
    found: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row: Any = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("each extraction case must be a JSON object")
            case_id = row.get("id")
            source = row.get("source")
            if case_id in wanted:
                if not isinstance(case_id, str) or not isinstance(source, str):
                    raise TypeError("case id and source must be strings")
                found[case_id] = source
    missing = [case_id for case_id in case_ids if case_id not in found]
    if missing:
        raise KeyError(f"missing extraction cases: {missing}")
    return found


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require_int(payload: dict[str, Any], key: str) -> int:
    value = payload[key]
    if not isinstance(value, int):
        raise TypeError(f"Ollama field {key!r} must be an int")
    return value


def optional_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"Ollama field {key!r} must be a string")
    return value


def call_ollama(
    settings: Settings,
    model_id: str,
    prompt: str,
    num_predict: int,
) -> tuple[dict[str, Any], int, str | None]:
    started = time.perf_counter()
    try:
        response = httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": model_id,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": num_predict,
                },
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return {}, latency_ms, type(exc).__name__

    latency_ms = int((time.perf_counter() - started) * 1000)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        return {}, latency_ms, type(exc).__name__

    try:
        payload: Any = response.json()
    except ValueError:
        return {}, latency_ms, "JSONDecodeError"
    if not isinstance(payload, dict):
        return {}, latency_ms, "TypeError"
    return payload, latency_ms, None


def build_record(
    *,
    run_id: str,
    model_id: str,
    case_id: str,
    attempt: int,
    num_predict: int,
    payload: dict[str, Any],
    latency_ms: int,
    error_type: str | None,
) -> CallRecord:
    if error_type is not None and not payload:
        input_tokens = 0
        output_tokens = 0
        stop_reason = None
        response_text = None
    else:
        input_tokens = require_int(payload, "prompt_eval_count")
        output_tokens = require_int(payload, "eval_count")
        stop_reason = optional_str(payload, "done_reason")
        response_text = optional_str(payload, "response")
    return CallRecord(
        record_id=str(uuid.uuid4()),
        run_id=run_id,
        timestamp=datetime.now(UTC),
        provider="ollama",
        model_id=model_id,
        task="extraction",
        case_id=case_id,
        prompt_id="baseline",
        prompt_version="v0",
        attempt=attempt,
        temperature=TEMPERATURE,
        max_output_tokens=num_predict,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=None,
        latency_ms=latency_ms,
        cost_usd=compute_cost(model_id, input_tokens, output_tokens),
        stop_reason=stop_reason,
        error_type=error_type,
        response_text=response_text,
    )


def record_attempt(
    *,
    settings: Settings,
    model_id: str,
    run_id: str,
    case_id: str,
    prompt: str,
    attempt: int,
    num_predict: int,
) -> CallRecord:
    payload, latency_ms, transport_error = call_ollama(
        settings, model_id, prompt, num_predict
    )
    error_type = transport_error
    if error_type is None:
        stop_reason = optional_str(payload, "done_reason")
        if stop_reason == "length":
            error_type = "TruncatedResponseError"
    record = build_record(
        run_id=run_id,
        model_id=model_id,
        case_id=case_id,
        attempt=attempt,
        num_predict=num_predict,
        payload=payload,
        latency_ms=latency_ms,
        error_type=error_type,
    )
    append_record(record, run_id)
    return record


def main() -> None:
    settings = Settings.from_env()
    model_id = settings.models["mistral"].model_id
    template = load_prompt(PROMPT_PATH)
    cases = load_extraction_cases(CASES_PATH, CASE_IDS)
    run_id = str(uuid.uuid4())

    for case_id in CASE_IDS:
        prompt = template.replace("{document_text}", cases[case_id])
        record = record_attempt(
            settings=settings,
            model_id=model_id,
            run_id=run_id,
            case_id=case_id,
            prompt=prompt,
            attempt=1,
            num_predict=NORMAL_NUM_PREDICT,
        )
        print(
            f"{case_id} stop={record.stop_reason} "
            f"input={record.input_tokens} output={record.output_tokens} "
            f"latency_ms={record.latency_ms} error_type={record.error_type}"
        )

    truncated_prompt = template.replace("{document_text}", cases["E11"])
    truncated_record = record_attempt(
        settings=settings,
        model_id=model_id,
        run_id=run_id,
        case_id="E11",
        prompt=truncated_prompt,
        attempt=2,
        num_predict=TRUNCATION_NUM_PREDICT,
    )
    print(
        f"E11 truncation demo stop={truncated_record.stop_reason} "
        f"error_type={truncated_record.error_type}"
    )
    print(f"run_id={run_id}")
    print(f"log=runs/{run_id}.jsonl")


if __name__ == "__main__":
    main()
