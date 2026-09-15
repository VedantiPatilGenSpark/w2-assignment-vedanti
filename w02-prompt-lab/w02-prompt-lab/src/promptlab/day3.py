"""Day 3 experiment: structured summarization and extraction on one Ollama model."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, Settings
from promptlab.records import OutputRecord
from promptlab.records import append_record as append_output_record
from promptlab.schemas import PolicyExtraction, SummarizationOutput, TaskName, schema_description
from promptlab.structured import StructuredOutputError, complete_structured
from promptlab.usage import append_record as append_call_record

SUMMARIZATION_IDS = (
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
EXTRACTION_IDS = (
    "E01",
    "E02",
    "E03",
    "E04",
    "E05",
    "E06",
    "E07",
    "E08",
    "E09",
    "E10",
    "E11",
    "E12",
)
SUMMARIZE_PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "summarize.v1.md"
EXTRACT_PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "extract.v2.md"
SUMMARIZATION_CASES_PATH = PROJECT_ROOT / "cases" / "summarization.jsonl"
EXTRACTION_CASES_PATH = PROJECT_ROOT / "cases" / "extraction.jsonl"
MAX_OUTPUT_TOKENS = 1024


class LoggingAdapter:
    """Forwards to OllamaAdapter, logs CallRecords, and counts complete() calls."""

    def __init__(self, inner: OllamaAdapter) -> None:
        self.provider = inner.provider
        self.model_id = inner.model_id
        self._inner = inner
        self.complete_calls = 0

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        self.complete_calls += 1
        result = self._inner.complete(request, run_id)
        for record in result.records:
            append_call_record(record, run_id)
        return result


def load_cases(path: Path, case_ids: tuple[str, ...]) -> dict[str, str]:
    wanted = set(case_ids)
    found: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row: Any = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("each case must be a JSON object")
            case_id = row.get("id")
            source = row.get("source")
            if case_id in wanted:
                if not isinstance(case_id, str) or not isinstance(source, str):
                    raise TypeError("case id and source must be strings")
                found[case_id] = source
    missing = [case_id for case_id in case_ids if case_id not in found]
    if missing:
        raise KeyError(f"missing cases: {missing}")
    return found


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fill_prompt(template: str, document_text: str, schema: type[BaseModel]) -> str:
    return template.replace("{schema_description}", schema_description(schema)).replace(
        "{document_text}", document_text
    )


def run_case(
    *,
    adapter: LoggingAdapter,
    request: CompletionRequest,
    schema: type[BaseModel],
    run_id: str,
    max_repairs: int,
    model_name: str,
    prompt_version: str,
    task: TaskName,
    outputs_path: Path,
) -> None:
    adapter.complete_calls = 0
    output: dict[str, Any] | None = None
    error: str | None = None
    succeeded = False
    try:
        parsed = complete_structured(adapter, request, schema, run_id, max_repairs=max_repairs)
        output = parsed.model_dump()
        succeeded = True
    except StructuredOutputError as exc:
        error = str(exc)
    repairs = max(adapter.complete_calls - 1, 0)
    append_output_record(
        outputs_path,
        OutputRecord(
            run_id=run_id,
            task=task,
            case_id=request.case_id,
            model_name=model_name,
            model_id=adapter.model_id,
            prompt_version=prompt_version,
            succeeded=succeeded,
            repairs=repairs,
            output=output,
            error=error,
        ),
    )
    print(
        f"{task} {request.case_id} succeeded={succeeded} "
        f"repairs={repairs} complete_calls={adapter.complete_calls} "
        f"error={error}"
    )


def main() -> None:
    settings = Settings.from_env()
    model_name = "mistral"
    model = settings.models[model_name]
    adapter = LoggingAdapter(OllamaAdapter(model_id=model.model_id))
    run_id = str(uuid.uuid4())
    outputs_path = Path("runs") / f"{run_id}-outputs.jsonl"
    max_repairs = settings.max_schema_repairs

    summarize_template = load_prompt(SUMMARIZE_PROMPT_PATH)
    extract_template = load_prompt(EXTRACT_PROMPT_PATH)
    summarization_cases = load_cases(SUMMARIZATION_CASES_PATH, SUMMARIZATION_IDS)
    extraction_cases = load_cases(EXTRACTION_CASES_PATH, EXTRACTION_IDS)

    print(f"model={model.model_id} temperature={settings.temperature} max_repairs={max_repairs}")

    for case_id in SUMMARIZATION_IDS:
        request = CompletionRequest(
            task="summarization",
            case_id=case_id,
            prompt_id="summarize",
            prompt_version="v1",
            system="",
            user_content=fill_prompt(
                summarize_template, summarization_cases[case_id], SummarizationOutput
            ),
            temperature=settings.temperature,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        run_case(
            adapter=adapter,
            request=request,
            schema=SummarizationOutput,
            run_id=run_id,
            max_repairs=max_repairs,
            model_name=model_name,
            prompt_version="v1",
            task="summarization",
            outputs_path=outputs_path,
        )

    for case_id in EXTRACTION_IDS:
        request = CompletionRequest(
            task="extraction",
            case_id=case_id,
            prompt_id="extract",
            prompt_version="v2",
            system="",
            user_content=fill_prompt(
                extract_template, extraction_cases[case_id], PolicyExtraction
            ),
            temperature=settings.temperature,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        run_case(
            adapter=adapter,
            request=request,
            schema=PolicyExtraction,
            run_id=run_id,
            max_repairs=max_repairs,
            model_name=model_name,
            prompt_version="v2",
            task="extraction",
            outputs_path=outputs_path,
        )

    print(f"run_id={run_id}")
    print(f"calls=runs/{run_id}.jsonl")
    print(f"outputs=runs/{run_id}-outputs.jsonl")


if __name__ == "__main__":
    main()
