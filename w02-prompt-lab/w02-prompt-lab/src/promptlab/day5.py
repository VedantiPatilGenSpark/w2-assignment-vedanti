"""Day 5 evaluation harness: three tasks, two configured Ollama models.

Runs through the existing prompt registry, complete_structured, ModelAdapter,
and OllamaAdapter. Scoring is deterministic and does not call a model.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Literal
from uuid import uuid4

from promptlab.adapters.base import CompletionRequest
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, Settings
from promptlab.corpus import GoldLabel, load_cases, validate_corpus
from promptlab.prompts import load, render_user, substitute
from promptlab.records import OutputRecord, ScoreRecord, UsageRecord, load_records
from promptlab.records import append_record as append_jsonl
from promptlab.report import write_reports
from promptlab.rules import VersionCandidate, select_current_version
from promptlab.schemas import (
    PolicyExtraction,
    StrictModel,
    SummarizationOutput,
    TaskName,
    TriageOutput,
    schema_description,
)
from promptlab.scoring import score_output, score_version_selection
from promptlab.structured import (
    StructuredCallTrace,
    StructuredCompletionError,
    complete_structured,
)
from promptlab.usage import CallRecord
from promptlab.usage import append_record as append_call_record

MAX_OUTPUT_TOKENS = 1024
EXPECTED_CASE_COUNT = 12
RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$"

# Prompts were authored against the Day 1-4 local model. The other configured
# model runs the same prompt versions as a transfer test.
DEVELOPMENT_LOGICAL_NAME = "mistral"

CASES_TASKS: tuple[TaskName, ...] = ("summarization", "extraction", "triage")
EVIDENCE_PATH = PROJECT_ROOT / "docs" / "day5-run.jsonl"
SCORES_PATH = PROJECT_ROOT / "docs" / "day5-scores.jsonl"
REPORT_PATH = PROJECT_ROOT / "reports" / "comparison.md"
DECISION_PATH = PROJECT_ROOT / "docs" / "model-decision.md"


@dataclass(frozen=True)
class TaskSpec:
    prompt_id: str
    version: str
    schema: type[StrictModel]


TASK_SPECS: dict[TaskName, TaskSpec] = {
    "summarization": TaskSpec("summarize", "v1", SummarizationOutput),
    "extraction": TaskSpec("extract", "v2", PolicyExtraction),
    "triage": TaskSpec("triage", "v1", TriageOutput),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day 5 local two-model prompt comparison")
    parser.add_argument("--run-id", help="Stable identifier for this run")
    parser.add_argument("--task", choices=list(TASK_SPECS))
    parser.add_argument("--model", help="Logical model name from configuration")
    parser.add_argument("--limit", type=int, help="Limit cases per task for a smoke run")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration and corpus without calling Ollama",
    )
    return parser


def _selected_models(settings: Settings, requested: str | None) -> list[str]:
    available = list(settings.models)
    if requested is None:
        return available
    if requested not in settings.models:
        raise SystemExit(
            f"Unknown logical model {requested!r}. Configured names: {', '.join(available)}"
        )
    return [requested]


def _adapter_for(settings: Settings, logical_name: str) -> OllamaAdapter:
    model = settings.models[logical_name]
    # think=False keeps the Ollama request shape identical across models. Day 2
    # showed Qwen's default thinking consumes the output budget and truncates.
    return OllamaAdapter(model_id=model.model_id, think=False)


def build_request(
    *,
    task: TaskName,
    spec: TaskSpec,
    case_id: str,
    source: str,
    temperature: float,
) -> CompletionRequest:
    template = load(spec.prompt_id, spec.version)
    schema_text = schema_description(spec.schema)
    variables = {"schema_description": schema_text}
    system = substitute(template.system, variables) if template.system else ""
    user_content = render_user(template, variables, untrusted=source)
    return CompletionRequest(
        task=task,
        case_id=case_id,
        prompt_id=spec.prompt_id,
        prompt_version=spec.version,
        system=system,
        user_content=user_content,
        temperature=temperature,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


UsageKind = Literal["primary", "transport_retry", "repair", "repair_retry"]
UsageStatus = Literal["success", "schema_invalid", "transport_error"]


def _usage_kind(records: Sequence[CallRecord]) -> list[tuple[CallRecord, UsageKind]]:
    labeled: list[tuple[CallRecord, UsageKind]] = []
    in_repair = False
    for record in records:
        kind: UsageKind
        if record.attempt == 1:
            if not labeled:
                kind = "primary"
            else:
                kind = "repair"
                in_repair = True
        elif in_repair:
            kind = "repair_retry"
        else:
            kind = "transport_retry"
        labeled.append((record, kind))
    return labeled


def _usage_status(record: CallRecord) -> UsageStatus:
    if record.error_type:
        return "transport_error"
    return "success"


def _to_usage(
    *,
    call: CallRecord,
    kind: UsageKind,
    model_name: str,
) -> UsageRecord:
    return UsageRecord(
        run_id=call.run_id,
        task=call.task,
        case_id=call.case_id,
        model_name=model_name,
        model_id=call.model_id,
        prompt_id=call.prompt_id,
        prompt_version=call.prompt_version,
        attempt=call.attempt,
        kind=kind,
        status=_usage_status(call),
        prompt_tokens=call.input_tokens,
        completion_tokens=call.output_tokens,
        latency_ms=float(call.latency_ms),
        cost_usd=Decimal(str(call.cost_usd)),
        error=call.error_type,
    )


def _version_fields(output: StrictModel) -> tuple[str, str] | None:
    if not isinstance(output, SummarizationOutput | PolicyExtraction):
        return None
    version = output.version
    effective = output.effective_date
    if (
        version.status == "present"
        and effective.status == "present"
        and isinstance(version.value, str)
        and isinstance(effective.value, str)
    ):
        return version.value, effective.value
    return None


def add_version_scores(
    *,
    run_id: str,
    task: TaskName,
    model_name: str,
    model_id: str,
    prompt_id: str,
    prompt_version: str,
    labels: Sequence[GoldLabel],
    outputs: Mapping[str, StrictModel],
) -> list[ScoreRecord]:
    grouped: dict[str, list[GoldLabel]] = defaultdict(list)
    for label in labels:
        if label.version_group:
            grouped[label.version_group].append(label)

    scores: list[ScoreRecord] = []
    for group_name, group_labels in grouped.items():
        if len(group_labels) < 2:
            continue
        expected = next(
            (
                label.expected_current_case_id
                for label in group_labels
                if label.expected_current_case_id
            ),
            None,
        )
        as_of_raw = next((label.as_of for label in group_labels if label.as_of), None)
        if expected is None or as_of_raw is None:
            continue
        candidates: list[VersionCandidate] = []
        for label in group_labels:
            output = outputs.get(label.id)
            if output is None:
                continue
            extracted = _version_fields(output)
            if extracted is None:
                continue
            version, effective_raw = extracted
            try:
                effective = date.fromisoformat(effective_raw)
            except ValueError:
                continue
            candidates.append(
                VersionCandidate(case_id=label.id, version=version, effective_date=effective)
            )
        selected = select_current_version(candidates, date.fromisoformat(as_of_raw))
        scores.append(
            score_version_selection(
                run_id=run_id,
                task=task,
                case_id=f"version:{group_name}",
                model_name=model_name,
                prompt_version=prompt_version,
                selected_case_id=selected.case_id if selected is not None else None,
                expected_case_id=expected,
                prompt_id=prompt_id,
                model_id=model_id,
            )
        )
    return scores


def run_case(
    *,
    adapter: OllamaAdapter,
    settings: Settings,
    run_id: str,
    task: TaskName,
    spec: TaskSpec,
    case_id: str,
    source: str,
    gold: GoldLabel,
    model_name: str,
    usage_path: Path,
    outputs_path: Path,
    scores_path: Path,
) -> tuple[OutputRecord, list[ScoreRecord], StrictModel | None]:
    request = build_request(
        task=task,
        spec=spec,
        case_id=case_id,
        source=source,
        temperature=settings.temperature,
    )
    trace = StructuredCallTrace()
    validated: StrictModel | None = None
    try:
        parsed = complete_structured(
            adapter,
            request,
            spec.schema,
            run_id,
            max_repairs=settings.max_schema_repairs,
            trace=trace,
        )
        validated = parsed
        output_record = OutputRecord(
            run_id=run_id,
            task=task,
            case_id=case_id,
            model_name=model_name,
            model_id=adapter.model_id,
            prompt_id=spec.prompt_id,
            prompt_version=spec.version,
            succeeded=True,
            repairs=trace.repairs,
            output=parsed.model_dump(),
            error=None,
        )
        case_scores = score_output(
            run_id=run_id,
            task=task,
            case_id=case_id,
            model_name=model_name,
            prompt_version=spec.version,
            output=parsed,
            gold=gold,
            source=source,
            prompt_id=spec.prompt_id,
            model_id=adapter.model_id,
        )
    except StructuredCompletionError as exc:
        trace = exc.trace
        output_record = OutputRecord(
            run_id=run_id,
            task=task,
            case_id=case_id,
            model_name=model_name,
            model_id=adapter.model_id,
            prompt_id=spec.prompt_id,
            prompt_version=spec.version,
            succeeded=False,
            repairs=exc.trace.repairs,
            output=None,
            error=str(exc),
        )
        case_scores = score_output(
            run_id=run_id,
            task=task,
            case_id=case_id,
            model_name=model_name,
            prompt_version=spec.version,
            output=None,
            gold=gold,
            source=source,
            prompt_id=spec.prompt_id,
            model_id=adapter.model_id,
        )

    for call, kind in _usage_kind(trace.records):
        append_call_record(call, run_id)
        append_jsonl(usage_path, _to_usage(call=call, kind=kind, model_name=model_name))
    append_jsonl(outputs_path, output_record)
    for score in case_scores:
        append_jsonl(scores_path, score)
    print(
        f"{task:13} {model_name:8} {case_id:5} "
        f"{'ok' if output_record.succeeded else 'failed'} repairs={output_record.repairs}"
    )
    return output_record, case_scores, validated


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    os.chdir(PROJECT_ROOT)
    counts = validate_corpus()
    if args.validate_only:
        print("Corpus valid: " + ", ".join(f"{task}={count}" for task, count in counts.items()))
        return

    settings = Settings.from_env()
    run_id = args.run_id or str(uuid4())
    if not re.fullmatch(RUN_ID_PATTERN, run_id):
        raise SystemExit("--run-id must use letters, numbers, '.', '_' or '-'")
    selected_tasks: list[TaskName] = [args.task] if args.task else list(CASES_TASKS)
    selected_models = _selected_models(settings, args.model)
    limit = args.limit
    if limit is not None and limit < 1:
        raise SystemExit("--limit must be at least 1")

    run_dir = PROJECT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    usage_path = run_dir / "usage.jsonl"
    outputs_path = run_dir / "outputs.jsonl"
    scores_path = run_dir / "scores.jsonl"

    all_usage: list[UsageRecord] = []
    all_outputs: list[OutputRecord] = []
    all_scores: list[ScoreRecord] = []
    validated_by_task_model: dict[tuple[TaskName, str], dict[str, StrictModel]] = defaultdict(dict)
    labels_by_task: dict[TaskName, list[GoldLabel]] = {}

    for task in selected_tasks:
        spec = TASK_SPECS[task]
        pairs = load_cases(task)
        if len(pairs) != EXPECTED_CASE_COUNT:
            raise ValueError(f"expected {EXPECTED_CASE_COUNT} {task} cases, got {len(pairs)}")
        if limit is not None:
            pairs = pairs[:limit]
        labels_by_task[task] = [gold for _case, gold in pairs]
        for logical_name in selected_models:
            adapter = _adapter_for(settings, logical_name)
            for case, gold in pairs:
                output_record, case_scores, validated = run_case(
                    adapter=adapter,
                    settings=settings,
                    run_id=run_id,
                    task=task,
                    spec=spec,
                    case_id=case.id,
                    source=case.document_text,
                    gold=gold,
                    model_name=logical_name,
                    usage_path=usage_path,
                    outputs_path=outputs_path,
                    scores_path=scores_path,
                )
                all_outputs.append(output_record)
                all_scores.extend(case_scores)
                if validated is not None:
                    validated_by_task_model[(task, logical_name)][case.id] = validated

    all_usage = load_records(usage_path, UsageRecord)

    for task in selected_tasks:
        if task == "triage":
            continue
        spec = TASK_SPECS[task]
        for logical_name in selected_models:
            model = settings.models[logical_name]
            version_scores = add_version_scores(
                run_id=run_id,
                task=task,
                model_name=logical_name,
                model_id=model.model_id,
                prompt_id=spec.prompt_id,
                prompt_version=spec.version,
                labels=labels_by_task[task],
                outputs=validated_by_task_model[(task, logical_name)],
            )
            for score in version_scores:
                append_jsonl(scores_path, score)
                all_scores.append(score)

    call_log = Path("runs") / f"{run_id}.jsonl"
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if call_log.exists():
        shutil.copyfile(call_log, EVIDENCE_PATH)
    else:
        EVIDENCE_PATH.write_text("", encoding="utf-8")
    if SCORES_PATH.exists():
        SCORES_PATH.unlink()
    for score in all_scores:
        append_jsonl(SCORES_PATH, score)

    write_reports(
        run_id=run_id,
        models=selected_models,
        usage=all_usage,
        outputs=all_outputs,
        scores=all_scores,
        report_path=REPORT_PATH,
        decision_path=DECISION_PATH,
        development_model=DEVELOPMENT_LOGICAL_NAME,
    )
    print(f"run_id={run_id}")
    print(f"calls={call_log}")
    print(f"evidence={EVIDENCE_PATH}")
    print(f"scores={SCORES_PATH}")
    print(f"report={REPORT_PATH}")


if __name__ == "__main__":
    main()
