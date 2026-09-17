from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from promptlab.records import OutputRecord, ScoreRecord, UsageRecord
from promptlab.report import write_reports


def test_report_is_generated_from_records(tmp_path: Path) -> None:
    usage = [
        UsageRecord(
            run_id="demo",
            task="triage",
            case_id="T01",
            model_name="mistral",
            model_id="mistral:7b",
            prompt_version="triage-mistral-v1",
            attempt=1,
            kind="primary",
            status="success",
            prompt_tokens=100,
            completion_tokens=25,
            latency_ms=125.0,
            cost_usd=Decimal("0"),
        )
    ]
    outputs = [
        OutputRecord(
            run_id="demo",
            task="triage",
            case_id="T01",
            model_name="mistral",
            model_id="mistral:7b",
            prompt_version="triage-mistral-v1",
            succeeded=True,
            repairs=0,
            output={"queue": "card_dispute"},
        )
    ]
    scores = [
        ScoreRecord(
            run_id="demo",
            task="triage",
            case_id="T01",
            model_name="mistral",
            prompt_version="triage-mistral-v1",
            scorer_version="2.0.0",
            metric="queue_accuracy",
            numerator=1,
            denominator=1,
        )
    ]
    report = tmp_path / "comparison.md"
    decision = tmp_path / "model-decision.md"
    write_reports(
        run_id="demo",
        models=["mistral"],
        usage=usage,
        outputs=outputs,
        scores=scores,
        report_path=report,
        decision_path=decision,
    )
    text = report.read_text(encoding="utf-8")
    assert "1/1" in text
    assert "triage-mistral-v1" in text
    assert "125 ms | 125 ms | 1 | — | — | 0 |" in text
    assert "Median case latency" in text
    assert "mistral" in decision.read_text(encoding="utf-8")


def test_report_uses_case_latency_separate_from_attempt_latency(tmp_path: Path) -> None:
    usage = [
        UsageRecord(
            run_id="demo",
            task="triage",
            case_id="T01",
            model_name="mistral",
            model_id="mistral:7b",
            prompt_version="triage-mistral-v1",
            attempt=1,
            kind="primary",
            status="success",
            prompt_tokens=100,
            completion_tokens=25,
            latency_ms=125.0,
            cost_usd=Decimal("0"),
        ),
        UsageRecord(
            run_id="demo",
            task="triage",
            case_id="T01",
            model_name="mistral",
            model_id="mistral:7b",
            prompt_version="triage-mistral-v1",
            attempt=1,
            kind="repair",
            status="success",
            prompt_tokens=40,
            completion_tokens=10,
            latency_ms=80.0,
            cost_usd=Decimal("0"),
        ),
    ]
    outputs = [
        OutputRecord(
            run_id="demo",
            task="triage",
            case_id="T01",
            model_name="mistral",
            model_id="mistral:7b",
            prompt_version="triage-mistral-v1",
            succeeded=True,
            repairs=1,
            output={"queue": "card_dispute"},
            case_latency_ms=250,
        )
    ]
    scores = [
        ScoreRecord(
            run_id="demo",
            task="triage",
            case_id="T01",
            model_name="mistral",
            prompt_version="triage-mistral-v1",
            scorer_version="2.0.0",
            metric="queue_accuracy",
            numerator=1,
            denominator=1,
        )
    ]
    report = tmp_path / "comparison.md"
    write_reports(
        run_id="demo",
        models=["mistral"],
        usage=usage,
        outputs=outputs,
        scores=scores,
        report_path=report,
        decision_path=tmp_path / "model-decision.md",
    )
    text = report.read_text(encoding="utf-8")
    assert "102.5 ms | 125 ms | 2 | 250 ms | 250 ms | 1 |" in text
    assert "Attempt latency (`n`) is one HTTP POST" in text

