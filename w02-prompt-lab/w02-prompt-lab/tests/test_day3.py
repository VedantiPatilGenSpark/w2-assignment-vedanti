"""Offline tests for the day-3 runner checks."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from promptlab.adapters.base import CompletionRequest, CompletionResult
from promptlab.day3 import CaseOutcome, LoggingAdapter, citation_failures, split_prompt
from promptlab.records import OutputRecord
from promptlab.schemas import PolicyExtraction, SummarizationOutput
from promptlab.structured import StructuredCallTrace, complete_structured
from promptlab.usage import CallRecord

CASE = {
    "id": "S01",
    "task": "summarization",
    "source": (
        "1. Document Control\nTitle: Card Dispute Intake Procedure. Version: 1.0.\n"
        "2. Purpose\nStandardize intake.\n"
        "3. Required Steps\nRecord the merchant and amount.\n"
        "4. Exceptions\nEscalate fraud."
    ),
}


def _outcome(case_id: str, output: dict[str, Any]) -> CaseOutcome:
    record = OutputRecord(
        run_id="test-run",
        task="summarization",
        case_id=case_id,
        model_name="mistral",
        model_id="mistral:7b",
        prompt_version="v1",
        succeeded=True,
        repairs=0,
        output=output,
        error=None,
    )
    return CaseOutcome(record=record, first_error=None)


def test_citation_failures_flag_bare_numbers_and_accept_headings() -> None:
    heading_citation = {
        "document_status": "valid",
        "title": {"value": "v", "status": "present", "citation": "1. Document Control"},
        "version": {"value": None, "status": "absent", "citation": None},
        "effective_date": {"value": None, "status": "absent", "citation": None},
        "purpose": {"value": "p", "status": "present", "citation": "2. Purpose"},
        "required_steps": {"value": None, "status": "absent", "citation": None},
        "exceptions": {"value": None, "status": "absent", "citation": None},
    }
    bare_number = {
        "document_status": "valid",
        "title": {"value": "v", "status": "present", "citation": "1"},
        "version": {"value": None, "status": "absent", "citation": None},
        "effective_date": {"value": None, "status": "absent", "citation": None},
        "purpose": {"value": "p", "status": "present", "citation": "2"},
        "required_steps": {"value": None, "status": "absent", "citation": None},
        "exceptions": {"value": None, "status": "absent", "citation": None},
    }

    assert citation_failures([_outcome("S01", heading_citation)], [CASE], SummarizationOutput) == []
    failures = citation_failures([_outcome("S01", bare_number)], [CASE], SummarizationOutput)
    assert len(failures) == 2
    assert all("S01 " in failure for failure in failures)


def test_citation_check_generalizes_to_extraction_schema() -> None:
    extraction_case = {
        "id": "E01",
        "task": "extraction",
        "source": CASE["source"],
    }
    output = {
        "document_status": "valid",
        "policy_name": {"value": "p", "status": "present", "citation": "Document Control"},
        "version": {"value": None, "status": "absent", "citation": None},
        "effective_date": {"value": None, "status": "absent", "citation": None},
        "jurisdictions": {"value": None, "status": "absent", "citation": None},
        "beneficial_ownership_threshold": {
            "value": None,
            "status": "absent",
            "citation": None,
        },
        "review_frequency": {"value": None, "status": "absent", "citation": None},
        "required_documents": {"value": None, "status": "absent", "citation": None},
    }

    assert citation_failures([_outcome("E01", output)], [extraction_case], PolicyExtraction) == []


def test_split_prompt_keeps_sections_after_the_document() -> None:
    template = (
        "Task\n\nDo the thing.\n\nInput\n\n<document>\n{document_text}\n</document>\n\n"
        "Examples\n\nTwo examples here.\n\nWhen the task cannot be completed\n\nSay so."
    )
    system, user_content = split_prompt(template, "SOURCE TEXT", "SCHEMA TEXT")

    assert "<document>" not in system
    assert "SOURCE TEXT" in user_content
    assert "SCHEMA TEXT" not in system.split("<document>")[0]
    assert "Examples" in user_content
    assert "When the task cannot be completed" in user_content
    assert user_content.startswith("<document>")


class _Answer(BaseModel):
    value: str


class _ScriptedAdapter:
    provider = "ollama"
    model_id = "fixture-model"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request: CompletionRequest, run_id: str) -> CompletionResult:
        self.calls += 1
        text = '{"wrong":"shape"}' if self.calls == 1 else '{"value":"fixed"}'
        return CompletionResult(
            succeeded=True,
            text=text,
            error_type=None,
            records=[
                CallRecord(
                    record_id=f"record-{self.calls}",
                    run_id=run_id,
                    timestamp=datetime.now(UTC),
                    provider="ollama",
                    model_id=self.model_id,
                    task=request.task,
                    case_id=request.case_id,
                    prompt_id=request.prompt_id,
                    prompt_version=request.prompt_version,
                    attempt=1,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                    input_tokens=1,
                    output_tokens=2,
                    cached_input_tokens=None,
                    latency_ms=3,
                    cost_usd=0.0,
                    stop_reason="stop",
                    error_type=None,
                    response_text=text,
                )
            ],
        )


def test_logging_adapter_writes_each_complete_call_during_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    inner = _ScriptedAdapter()
    adapter = LoggingAdapter(inner)
    trace = StructuredCallTrace()
    request = CompletionRequest(
        task="summarization",
        case_id="S00",
        prompt_id="summarize",
        prompt_version="v1",
        system="",
        user_content="Summarize.",
        temperature=0.0,
        max_output_tokens=32,
    )

    result = complete_structured(
        adapter, request, _Answer, "combined-run", max_repairs=1, trace=trace
    )

    assert result == _Answer(value="fixed")
    assert adapter.complete_calls == 2
    assert trace.repairs == 1
    log_path = tmp_path / "runs" / "combined-run.jsonl"
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["response_text"] == '{"wrong":"shape"}'
    assert json.loads(lines[1])["response_text"] == '{"value":"fixed"}'
