from __future__ import annotations

from pydantic import BaseModel

from promptlab.schemas import (
    EvidenceField,
    PolicyExtraction,
    SummarizationOutput,
    schema_description,
)


class TinyOutput(BaseModel):
    label: str
    count: int


def test_schema_description_is_derived_from_supplied_model() -> None:
    text = schema_description(TinyOutput)

    assert "label" in text
    assert "count" in text
    assert "policy_name" not in text
    assert "required_steps" not in text


def test_summarization_description_includes_nested_evidence_field() -> None:
    text = schema_description(SummarizationOutput)

    for name in SummarizationOutput.model_fields:
        assert name in text
    assert "EvidenceField" in text
    assert "value" in text
    assert "status" in text
    assert "citation" in text
    assert "policy_name" not in text


def test_extraction_description_does_not_use_summarization_fields() -> None:
    text = schema_description(PolicyExtraction)

    for name in PolicyExtraction.model_fields:
        assert name in text
    assert "required_steps" not in text


def test_citation_is_marked_optional() -> None:
    text = schema_description(EvidenceField)

    assert "citation" in text
    assert "optional" in text.lower()


def test_schema_description_is_not_json_schema() -> None:
    text = schema_description(SummarizationOutput)

    assert "$schema" not in text
    assert '"properties"' not in text
