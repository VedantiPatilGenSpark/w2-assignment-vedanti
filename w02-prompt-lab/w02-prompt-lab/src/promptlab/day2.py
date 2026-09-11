"""Day 2 experiment: same summarization job on Mistral and Qwen via OllamaAdapter."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from promptlab.adapters.base import CompletionRequest
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, Settings
from promptlab.usage import append_record

CASE_IDS = (
    "S01",
    "S02",
    "S03",
    "S04",
    "S05",
    "S06",
    "S07",
    "S08",
    "S09",
    "S10",
    "S11",
    "S12",
)
PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "baseline.v0.md"
CASES_PATH = PROJECT_ROOT / "cases" / "summarization.jsonl"
MAX_OUTPUT_TOKENS = 512


def load_summarization_cases(path: Path, case_ids: tuple[str, ...]) -> dict[str, str]:
    wanted = set(case_ids)
    found: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row: Any = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("each summarization case must be a JSON object")
            case_id = row.get("id")
            source = row.get("source")
            if case_id in wanted:
                if not isinstance(case_id, str) or not isinstance(source, str):
                    raise TypeError("case id and source must be strings")
                found[case_id] = source
    missing = [case_id for case_id in case_ids if case_id not in found]
    if missing:
        raise KeyError(f"missing summarization cases: {missing}")
    return found


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _think_from_args(value: str | None) -> bool | None:
    if value is None:
        return None
    return value == "true"


def _think_label(think: bool | None) -> str:
    if think is None:
        return "omitted (Ollama model default)"
    return str(think).lower()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Day 2 Mistral vs Qwen summarization run")
    parser.add_argument(
        "--think",
        choices=("true", "false"),
        default=None,
        help="Send Ollama think=true/false for both models. If omitted, the field is not sent.",
    )
    args = parser.parse_args(argv)
    think = _think_from_args(args.think)

    settings = Settings.from_env()
    template = load_prompt(PROMPT_PATH)
    cases = load_summarization_cases(CASES_PATH, CASE_IDS)
    run_id = str(uuid.uuid4())
    adapters = (
        OllamaAdapter(model_id=settings.models["mistral"].model_id, think=think),
        OllamaAdapter(model_id=settings.models["qwen"].model_id, think=think),
    )

    print(f"think={_think_label(think)}")
    for adapter in adapters:
        for case_id in CASE_IDS:
            request = CompletionRequest(
                task="summarization",
                case_id=case_id,
                prompt_id="baseline",
                prompt_version="v0",
                system="",
                user_content=template.replace("{document_text}", cases[case_id]),
                temperature=settings.temperature,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            result = adapter.complete(request, run_id)
            for record in result.records:
                append_record(record, run_id)
            last = result.records[-1]
            print(
                f"{adapter.model_id} {case_id} succeeded={result.succeeded} "
                f"attempts={len(result.records)} "
                f"input={last.input_tokens} output={last.output_tokens} "
                f"latency_ms={last.latency_ms} error_type={result.error_type}"
            )

    print(f"run_id={run_id}")
    print(f"log=runs/{run_id}.jsonl")


if __name__ == "__main__":
    main()
