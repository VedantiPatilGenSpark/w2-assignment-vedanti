"""Day 3 experiment: structured summarization and extraction on one Ollama model."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from promptlab.adapters.base import CompletionRequest, CompletionResult, ModelAdapter
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, Settings
from promptlab.records import OutputRecord
from promptlab.records import append_record as append_output_record
from promptlab.schemas import (
    PolicyExtraction,
    SummarizationOutput,
    TaskName,
    schema_description,
)
from promptlab.structured import (
    StructuredCallTrace,
    StructuredCompletionError,
    complete_structured,
)
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
RUNS_DIR = PROJECT_ROOT / "runs"
EVIDENCE_PATH = PROJECT_ROOT / "docs" / "day3-run.jsonl"
NOTES_PATH = PROJECT_ROOT / "docs" / "day3-notes.md"
MAX_OUTPUT_TOKENS = 1024
DOCUMENT_OPEN = "<document>"
LOGICAL_MODEL_KEY = "mistral"
_HEADING_RE = re.compile(r"^\s*\d+\.\s+(.+?)\s*$")


@dataclass(frozen=True)
class CaseOutcome:
    record: OutputRecord
    first_error: str | None


class LoggingAdapter:
    """Forwards to OllamaAdapter, logs CallRecords, and counts complete() calls."""

    def __init__(self, inner: ModelAdapter) -> None:
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


def split_prompt(template: str, source: str, schema_text: str) -> tuple[str, str]:
    """Fill placeholders and split instructions from the marked document.

    Everything from the first <document> marker onward stays in user_content so
    sections that follow the document (constraints, examples) are preserved.
    """
    filled = template.replace("{schema_description}", schema_text)
    filled = filled.replace("{document_text}", source)
    marker_at = filled.find(DOCUMENT_OPEN)
    if marker_at < 0:
        raise ValueError("prompt template is missing document markers")
    return filled[:marker_at].rstrip(), filled[marker_at:]


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
) -> CaseOutcome:
    adapter.complete_calls = 0
    trace = StructuredCallTrace()
    output: dict[str, Any] | None = None
    error: str | None = None
    succeeded = False
    try:
        parsed = complete_structured(
            adapter,
            request,
            schema,
            run_id,
            max_repairs=max_repairs,
            trace=trace,
        )
        output = parsed.model_dump()
        succeeded = True
    except StructuredCompletionError as exc:
        error = str(exc)
        trace = exc.trace
    repairs = trace.repairs
    record = OutputRecord(
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
        case_latency_ms=trace.elapsed_ms,
    )
    append_output_record(outputs_path, record)
    print(
        f"{task} {request.case_id} succeeded={succeeded} "
        f"repairs={repairs} complete_calls={adapter.complete_calls} "
        f"error={error}"
    )
    return CaseOutcome(record=record, first_error=trace.first_error)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: Sequence[str], size: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def _example_document_texts(prompt_text: str) -> list[str]:
    texts: list[str] = []
    for part in prompt_text.split("<document>")[1:]:
        body, _, _ = part.partition("</document>")
        if "{document_text}" not in body:
            texts.append(body)
    return texts


def distinctive_example_ngrams(
    corpus_texts: Sequence[str],
    example_texts: Sequence[str],
) -> set[tuple[str, ...]]:
    """Return example n-grams absent from every corpus text, at the largest useful size."""
    for size in (5, 4, 3):
        corpus: set[tuple[str, ...]] = set()
        for text in corpus_texts:
            corpus |= _ngrams(_tokens(text), size)
        example_grams: set[tuple[str, ...]] = set()
        for text in example_texts:
            example_grams |= _ngrams(_tokens(text), size)
        distinctive = example_grams - corpus
        if distinctive:
            return distinctive
    return set()


def contains_any_gram(text: str, grams: set[tuple[str, ...]]) -> bool:
    if not grams:
        return False
    tokens = _tokens(text)
    size = len(next(iter(grams)))
    return any(tuple(tokens[i : i + size]) in grams for i in range(len(tokens) - size + 1))


def leakage_case_ids(outcomes: Sequence[CaseOutcome], grams: set[tuple[str, ...]]) -> list[str]:
    if not grams:
        return []
    hits: list[str] = []
    for outcome in outcomes:
        if outcome.record.succeeded and outcome.record.output is not None:
            text = json.dumps(outcome.record.output)
            if contains_any_gram(text, grams):
                hits.append(outcome.record.case_id)
    return hits


def _section_headings(source: str) -> set[str]:
    headings: set[str] = set()
    for line in source.splitlines():
        match = _HEADING_RE.match(line)
        if match is not None:
            headings.add(" ".join(line.split()).casefold())
            headings.add(" ".join(match.group(1).split()).casefold())
        stripped = " ".join(line.split()).casefold()
        if stripped.startswith("## "):
            headings.add(stripped.removeprefix("## ").strip())
            headings.add(stripped)
        elif stripped.startswith("# "):
            headings.add(stripped.removeprefix("# ").strip())
            headings.add(stripped)
    return headings


def citation_exists(citation: str, headings: set[str]) -> bool:
    normalized = " ".join(citation.split()).casefold()
    normalized = re.sub(r"^section\s+", "", normalized)
    return normalized in headings


def citation_failures(
    outcomes: Sequence[CaseOutcome],
    cases: Sequence[dict[str, Any]],
    schema: type[SummarizationOutput] | type[PolicyExtraction],
) -> list[str]:
    source_by_id = {str(case["id"]): str(case["source"]) for case in cases}
    failures: list[str] = []
    for outcome in outcomes:
        record = outcome.record
        if not record.succeeded or record.output is None:
            continue
        validated = schema.model_validate(record.output)
        headings = _section_headings(source_by_id[record.case_id])
        for field_name, field in validated.evidence_fields().items():
            if field.status != "present":
                continue
            if field.citation is None or not citation_exists(field.citation, headings):
                failures.append(f"{record.case_id} {field_name}: citation={field.citation!r}")
    return failures


def _error_signature(first_error: str) -> str:
    match = re.search(r"\[type=([a-z_]+)", first_error)
    if match is not None:
        return match.group(1)
    return " ".join(first_error.split())[:200]


def _repair_rate(outcomes: Sequence[CaseOutcome]) -> tuple[int, int, str]:
    repaired = sum(1 for outcome in outcomes if outcome.record.repairs >= 1)
    total = len(outcomes)
    percent = f"{(repaired / total * 100):.1f}" if total else "0.0"
    return repaired, total, percent


def render_notes(
    settings: Settings,
    model_id: str,
    summarization: Sequence[CaseOutcome],
    extraction: Sequence[CaseOutcome],
    leakage: Sequence[str],
    citation_failure_lines: Sequence[str],
) -> str:
    sum_repaired, sum_total, sum_pct = _repair_rate(summarization)
    ext_repaired, ext_total, ext_pct = _repair_rate(extraction)
    first_failures = [
        (outcome, error)
        for outcome in [*summarization, *extraction]
        if (error := outcome.first_error) is not None
    ]
    recovered = sum(1 for outcome, _ in first_failures if outcome.record.succeeded)
    signature_counts = Counter(_error_signature(error) for _, error in first_failures)

    lines = [
        "# Day 3 notes: prompts that return validated objects",
        "",
        (
            f"Model: {model_id} (logical name {LOGICAL_MODEL_KEY}), temperature "
            f"{settings.temperature}, one run_id. Summarization ran "
            f"prompts/summarize.v1.md over {sum_total} summarization cases; "
            f"extraction ran prompts/extract.v2.md over {ext_total} extraction "
            "cases. A repair attempt means one bounded semantic repair request "
            "carrying the validation error; transport retries stay inside the "
            "adapter and are not counted. CallRecords are written by LoggingAdapter "
            "as each complete() returns; StructuredCallTrace records first-pass "
            "validation errors for these notes."
        ),
        "",
        "## Metrics",
        "",
        f"- Summarization repair rate: {sum_repaired}/{sum_total} ({sum_pct}%)",
        f"- Extraction repair rate: {ext_repaired}/{ext_total} ({ext_pct}%)",
        (
            f"- Example leakage count: {len(leakage)} of {ext_total} extraction "
            "outputs contained n-grams distinctive to the few-shot example "
            "documents" + (f" ({', '.join(leakage)})" if leakage else "")
        ),
        (
            f"- Citation-existence failures: {len(citation_failure_lines)} evidence "
            'fields with status "present" whose citation does not match a real '
            "section heading in the case source"
        ),
        "",
        "## Most common validation error",
        "",
    ]
    if first_failures:
        signature, count = signature_counts.most_common(1)[0]
        lines.append(
            f"The most common first-pass validation failure was the "
            f"{signature!r} error class, seen in {count} of "
            f"{len(first_failures)} first attempts that failed validation."
        )
        lines.append(
            f"The repair request repeated the delimited document together with the "
            f"previous response and the validation error and asked the model to "
            f"correct only that concern; this recovered {recovered} of "
            f"{len(first_failures)} first-pass failures."
        )
    else:
        lines.append(
            "No first-pass validation failures occurred, so no repair was needed."
        )
    if citation_failure_lines:
        lines.extend(["", "## Citation failures", ""])
        lines.extend(f"- {line}" for line in citation_failure_lines)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    settings = Settings.from_env()
    model_name = LOGICAL_MODEL_KEY
    model = settings.models[model_name]
    adapter = LoggingAdapter(OllamaAdapter(model_id=model.model_id))
    run_id = str(uuid.uuid4())
    outputs_path = RUNS_DIR / f"{run_id}-outputs.jsonl"
    outputs_path.parent.mkdir(parents=True, exist_ok=True)
    max_repairs = settings.max_schema_repairs

    summarize_template = load_prompt(SUMMARIZE_PROMPT_PATH)
    extract_template = load_prompt(EXTRACT_PROMPT_PATH)
    summarization_cases = load_cases(SUMMARIZATION_CASES_PATH, SUMMARIZATION_IDS)
    extraction_cases = load_cases(EXTRACTION_CASES_PATH, EXTRACTION_IDS)

    print(f"model={model.model_id} temperature={settings.temperature} max_repairs={max_repairs}")

    summarization: list[CaseOutcome] = []
    for case_id in SUMMARIZATION_IDS:
        system, user_content = split_prompt(
            summarize_template,
            summarization_cases[case_id],
            schema_description(SummarizationOutput),
        )
        request = CompletionRequest(
            task="summarization",
            case_id=case_id,
            prompt_id="summarize",
            prompt_version="v1",
            system=system,
            user_content=user_content,
            temperature=settings.temperature,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        summarization.append(
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
        )

    extraction: list[CaseOutcome] = []
    for case_id in EXTRACTION_IDS:
        system, user_content = split_prompt(
            extract_template,
            extraction_cases[case_id],
            schema_description(PolicyExtraction),
        )
        request = CompletionRequest(
            task="extraction",
            case_id=case_id,
            prompt_id="extract",
            prompt_version="v2",
            system=system,
            user_content=user_content,
            temperature=settings.temperature,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        extraction.append(
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
        )

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(outputs_path, EVIDENCE_PATH)

    summarization_rows = [
        {"id": case_id, "source": summarization_cases[case_id]} for case_id in SUMMARIZATION_IDS
    ]
    extraction_rows = [
        {"id": case_id, "source": extraction_cases[case_id]} for case_id in EXTRACTION_IDS
    ]
    corpus = [row["source"] for row in [*summarization_rows, *extraction_rows]]
    corpus.append(summarize_template)
    example_texts = _example_document_texts(extract_template)
    grams = distinctive_example_ngrams(corpus, example_texts)
    leakage = leakage_case_ids(extraction, grams)
    failures = [
        *citation_failures(summarization, summarization_rows, SummarizationOutput),
        *citation_failures(extraction, extraction_rows, PolicyExtraction),
    ]
    NOTES_PATH.write_text(
        render_notes(settings, model.model_id, summarization, extraction, leakage, failures),
        encoding="utf-8",
    )

    print(f"run_id={run_id}")
    print(f"calls=runs/{run_id}.jsonl")
    print(f"outputs={outputs_path}")
    print(f"evidence={EVIDENCE_PATH}")
    print(f"notes={NOTES_PATH}")


if __name__ == "__main__":
    main()
