"""Reporting for the Week 2 model-comparison lab.

The reporting layer consumes the existing UsageRecord, OutputRecord, and
ScoreRecord objects.  It does not rescore model output and it does not call an
LLM.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import median
from typing import Any

from promptlab.records import OutputRecord, ScoreRecord, UsageRecord

_ConfigKey = tuple[str, str, str]  # task, model_name, prompt_version


def _key(record: Any) -> _ConfigKey:
    return (
        str(record.task),
        str(record.model_name),
        str(record.prompt_version),
    )


def _for_run(records: Sequence[Any], run_id: str) -> list[Any]:
    return [record for record in records if str(record.run_id) == run_id]


def _fmt_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _aggregate_scores(
    records: Sequence[ScoreRecord],
) -> dict[str, tuple[int, int, bool | None]]:
    """Aggregate compatible score counts without averaging percentages."""

    grouped: dict[str, list[ScoreRecord]] = defaultdict(list)
    for record in records:
        grouped[str(record.metric)].append(record)

    result: dict[str, tuple[int, int, bool | None]] = {}

    for metric, rows in sorted(grouped.items()):
        numerator = sum(int(row.numerator) for row in rows)
        denominator = sum(int(row.denominator) for row in rows)

        directions = {
            bool(value)
            for value in (getattr(row, "lower_is_better", None) for row in rows)
            if value is not None
        }
        lower_is_better = next(iter(directions)) if len(directions) == 1 else None

        result[metric] = (numerator, denominator, lower_is_better)

    return result


def _metric_text(records: Sequence[ScoreRecord]) -> str:
    metrics = _aggregate_scores(records)
    if not metrics:
        return "—"

    rendered: list[str] = []
    for metric, (numerator, denominator, lower_is_better) in metrics.items():
        suffix = " ↓" if lower_is_better else ""
        rendered.append(f"{metric}: {numerator}/{denominator}{suffix}")

    return "<br>".join(rendered)


def _usage_summary(
    records: Sequence[UsageRecord],
) -> tuple[str, str, str, str, str, str]:
    """Return token, latency, observation, and retry summaries."""

    if not records:
        return "—", "—", "—", "—", "0", "0"

    prompt_tokens = sum(int(getattr(row, "prompt_tokens", 0) or 0) for row in records)
    completion_tokens = sum(
        int(getattr(row, "completion_tokens", 0) or 0) for row in records
    )

    latencies = [
        float(row.latency_ms)
        for row in records
        if getattr(row, "latency_ms", None) is not None
    ]

    if latencies:
        median_latency = f"{_fmt_number(float(median(latencies)))} ms"
        max_latency = f"{_fmt_number(float(max(latencies)))} ms"
    else:
        median_latency = "—"
        max_latency = "—"

    # A semantic repair is a separate model request and should not also be
    # reported as a transport retry merely because it has an attempt number.
    retry_attempts = sum(
        1
        for row in records
        if int(getattr(row, "attempt", 1) or 1) > 1
        and str(getattr(row, "kind", "")).lower() != "repair"
    )

    return (
        str(prompt_tokens),
        str(completion_tokens),
        median_latency,
        max_latency,
        str(len(latencies)),
        str(retry_attempts),
    )


def _output_summary(
    records: Sequence[OutputRecord],
) -> tuple[str, str, str]:
    if not records:
        return "0/0", "0/0", "0"

    total = len(records)
    succeeded = sum(1 for row in records if bool(row.succeeded))
    repairs_needed = sum(
        1 for row in records if int(getattr(row, "repairs", 0) or 0) > 0
    )
    failures = total - succeeded

    return (
        f"{succeeded}/{total}",
        f"{repairs_needed}/{total}",
        str(failures),
    )


def _all_config_keys(
    usage: Sequence[UsageRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
) -> list[_ConfigKey]:
    keys = {_key(row) for row in usage}
    keys.update(_key(row) for row in outputs)
    keys.update(_key(row) for row in scores)
    return sorted(keys)


def _prompt_label(
    *,
    model_name: str,
    prompt_version: str,
    prompt_id: str,
    development_model: str | None,
) -> str:
    label = f"{prompt_id}.{prompt_version}" if prompt_id else prompt_version
    if development_model and model_name != development_model:
        return f"{label} transfer"
    return label


def _write_report(
    *,
    run_id: str,
    usage: Sequence[UsageRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    report_path: Path,
    development_model: str | None = None,
) -> None:
    lines: list[str] = [
        "# Model Comparison",
        "",
        f"Run ID: `{run_id}`",
        "",
        "Counts are reported with their denominators. "
        "Latency uses median and maximum rather than mean.",
        "",
    ]

    keys = _all_config_keys(usage, outputs, scores)
    tasks = sorted({task for task, _model, _prompt in keys})

    if not tasks:
        lines.extend(
            [
                "No records were supplied for this run.",
                "",
            ]
        )

    for task in tasks:
        lines.extend(
            [
                f"## {task.title()}",
                "",
                "| Model | Prompt | Valid outputs | Metrics | Input tokens/case | "
                "Output tokens/case | Median latency | Max latency | n | "
                "Repairs | Retries | Final failures |",
                "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | "
                "---: | ---: | ---: |",
            ]
        )

        task_keys = [key for key in keys if key[0] == task]

        for key in task_keys:
            _task, model_name, prompt_version = key

            u = [row for row in usage if _key(row) == key]
            o = [row for row in outputs if _key(row) == key]
            s = [row for row in scores if _key(row) == key]

            (
                input_tokens,
                output_tokens,
                median_latency,
                max_latency,
                n,
                retries,
            ) = _usage_summary(u)

            valid_outputs, repairs, failures = _output_summary(o)
            metric_text = _metric_text(s)
            prompt_id = ""
            sample = o[0] if o else (u[0] if u else None)
            if sample is not None:
                prompt_id = str(getattr(sample, "prompt_id", "") or "")
            case_count = len(o)
            if case_count and input_tokens.isdigit() and output_tokens.isdigit():
                input_cell = _fmt_number(int(input_tokens) / case_count)
                output_cell = _fmt_number(int(output_tokens) / case_count)
            else:
                input_cell = input_tokens
                output_cell = output_tokens
            prompt_cell = _prompt_label(
                model_name=model_name,
                prompt_version=prompt_version,
                prompt_id=prompt_id,
                development_model=development_model,
            )

            lines.append(
                "| "
                f"{model_name} | {prompt_cell} | {valid_outputs} | "
                f"{metric_text} | {input_cell} | {output_cell} | "
                f"{median_latency} | {max_latency} | {n} | {repairs} | "
                f"{retries} | {failures} |"
            )

        lines.append("")

    lines.extend(
        [
            "## Limits",
            "",
            "- Each task has 12 cases. Counts are directional, not production-scale estimates.",
            "- A one-case gap such as 11/12 versus 10/12 is not a universal model ranking.",
            "- A row measures the model together with the prompt version shown in that row.",
            "- Prompt-transfer rows are labeled `transfer`. They are evidence about that "
            "prompt on the second model, not proof of the model's best adapted performance.",
            "- Untested combinations (other prompt versions, other decoding settings) are "
            "not claimed.",
            "- Local Ollama latency depends on lab hardware and is not a production SLO.",
            "- Local Ollama provider/API charge is `$0.00`; token usage and latency still "
            "represent real operational work.",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_decision_scaffold(
    *,
    run_id: str,
    models: Sequence[str],
    usage: Sequence[UsageRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    decision_path: Path,
) -> None:
    """Write an evidence scaffold, not an invented model recommendation."""

    keys = _all_config_keys(usage, outputs, scores)

    lines: list[str] = [
        "# Model Decision Record",
        "",
        f"Run ID: `{run_id}`",
        "",
        "Use this file to record the task-level decision after reviewing the measured "
        "comparison. Do not select one universal model solely because it leads on a "
        "different task.",
        "",
        "## Evaluated models",
        "",
    ]

    evaluated_models = sorted(
        {model for _task, model, _prompt in keys} | {str(model) for model in models}
    )
    if evaluated_models:
        for model in evaluated_models:
            lines.append(f"- {model}")
    else:
        lines.append("- None")

    lines.extend(["", "## Evaluated configurations", ""])

    if keys:
        for task, model, prompt in keys:
            lines.append(f"- `{task}` — {model} — `{prompt}`")
    else:
        lines.append("- No configurations supplied.")

    lines.extend(
        [
            "",
            "## Task decisions",
            "",
            "For each task, complete:",
            "",
            "- selected model",
            "- prompt version",
            "- measured reason",
            "- rejected alternative(s)",
            "- condition that would reopen the decision",
            "",
        ]
    )

    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text("\n".join(lines), encoding="utf-8")


def write_reports(
    *,
    run_id: str,
    models: Sequence[str],
    usage: Sequence[UsageRecord],
    outputs: Sequence[OutputRecord],
    scores: Sequence[ScoreRecord],
    report_path: Path,
    decision_path: Path,
    development_model: str | None = None,
) -> None:
    """Generate the comparison report and decision scaffold for one run.

    Only records whose ``run_id`` matches the requested run are included.
    """

    run_usage = _for_run(usage, run_id)
    run_outputs = _for_run(outputs, run_id)
    run_scores = _for_run(scores, run_id)

    _write_report(
        run_id=run_id,
        usage=run_usage,
        outputs=run_outputs,
        scores=run_scores,
        report_path=Path(report_path),
        development_model=development_model,
    )

    _write_decision_scaffold(
        run_id=run_id,
        models=models,
        usage=run_usage,
        outputs=run_outputs,
        scores=run_scores,
        decision_path=Path(decision_path),
    )
