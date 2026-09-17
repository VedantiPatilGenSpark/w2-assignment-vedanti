"""Offline tests for the Day 5 harness wiring."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from promptlab.corpus import validate_corpus
from promptlab.day5 import TASK_SPECS, build_request
from promptlab.prompts import load
from promptlab.rules import VersionCandidate, select_current_version
from promptlab.schemas import schema_description


def test_corpus_has_twelve_cases_per_task() -> None:
    assert validate_corpus() == {
        "triage": 12,
        "summarization": 12,
        "extraction": 12,
    }


def test_task_prompts_are_the_measured_day5_versions() -> None:
    assert TASK_SPECS["summarization"].prompt_id == "summarize"
    assert TASK_SPECS["summarization"].version == "v1"
    assert TASK_SPECS["extraction"].prompt_id == "extract"
    assert TASK_SPECS["extraction"].version == "v2"
    assert TASK_SPECS["triage"].prompt_id == "triage"
    assert TASK_SPECS["triage"].version == "v1"
    for spec in TASK_SPECS.values():
        load(spec.prompt_id, spec.version)


def test_build_request_goes_through_prompt_registry() -> None:
    spec = TASK_SPECS["extraction"]
    request = build_request(
        task="extraction",
        spec=spec,
        case_id="E01",
        source="1. Document Control\nPolicy name: Test.",
        temperature=0.0,
    )
    assert request.prompt_id == "extract"
    assert request.prompt_version == "v2"
    assert "<document>" in request.user_content
    assert "Policy name: Test." in request.user_content
    assert schema_description(spec.schema) in request.user_content
    assert request.temperature == 0.0


def test_day5_module_has_no_model_id_literals() -> None:
    text = Path(__file__).resolve().parents[1].joinpath(
        "src", "promptlab", "day5.py"
    ).read_text(encoding="utf-8")
    assert "mistral:7b" not in text
    assert "qwen3:8b" not in text
    assert "anthropic" not in text.lower()
    assert "azure" not in text.lower()


def test_select_current_version_equal_effective_dates_are_ambiguous() -> None:
    selected = select_current_version(
        [
            VersionCandidate("a", "1.0", date(2025, 1, 1)),
            VersionCandidate("b", "2.0", date(2025, 1, 1)),
        ],
        date(2025, 6, 1),
    )
    assert selected is None


def test_prompts_do_not_ask_the_model_which_document_is_current() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "prompts"
    for name in ("extract.v2.md", "summarize.v1.md", "triage.v1.md"):
        text = (root / name).read_text(encoding="utf-8").lower()
        assert "select_current_version" not in text
        assert "which document is current" not in text
        assert "which version is current" not in text
