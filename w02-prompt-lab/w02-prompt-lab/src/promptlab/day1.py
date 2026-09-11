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
) -> tuple[dict[str, Any], int]:
    started = time.perf_counter()
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
    latency_ms = int((time.perf_counter() - started) * 1000)
    response.raise_for_status()
    payload: Any = response.json()
    if not isinstance(payload, dict):
        raise TypeError("Ollama must return a JSON object")
    return payload, latency_ms


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
    input_tokens = require_int(payload, "prompt_eval_count")
    output_tokens = require_int(payload, "eval_count")
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
        stop_reason=optional_str(payload, "done_reason"),
        error_type=error_type,
        response_text=optional_str(payload, "response"),
    )


def main() -> None:
    settings = Settings.from_env()
    model_id = settings.models["mistral"].model_id
    template = load_prompt(PROMPT_PATH)
    cases = load_extraction_cases(CASES_PATH, CASE_IDS)
    run_id = str(uuid.uuid4())

    for case_id in CASE_IDS:
        prompt = template.replace("{document_text}", cases[case_id])
        payload, latency_ms = call_ollama(settings, model_id, prompt, NORMAL_NUM_PREDICT)
        record = build_record(
            run_id=run_id,
            model_id=model_id,
            case_id=case_id,
            attempt=1,
            num_predict=NORMAL_NUM_PREDICT,
            payload=payload,
            latency_ms=latency_ms,
            error_type=None,
        )
        append_record(record, run_id)
        print(
            f"{case_id} stop={record.stop_reason} "
            f"input={record.input_tokens} output={record.output_tokens} "
            f"latency_ms={record.latency_ms}"
        )

    truncated_prompt = template.replace("{document_text}", cases["E11"])
    truncated_payload, truncated_latency_ms = call_ollama(
        settings, model_id, truncated_prompt, TRUNCATION_NUM_PREDICT
    )
    truncated_stop = optional_str(truncated_payload, "done_reason")
    truncated_error = "TruncatedResponseError" if truncated_stop == "length" else None
    truncated_record = build_record(
        run_id=run_id,
        model_id=model_id,
        case_id="E11",
        attempt=2,
        num_predict=TRUNCATION_NUM_PREDICT,
        payload=truncated_payload,
        latency_ms=truncated_latency_ms,
        error_type=truncated_error,
    )
    append_record(truncated_record, run_id)
    print(
        f"E11 truncation demo stop={truncated_record.stop_reason} "
        f"error_type={truncated_record.error_type}"
    )
    print(f"run_id={run_id}")
    print(f"log=runs/{run_id}.jsonl")


if __name__ == "__main__":
    main()
