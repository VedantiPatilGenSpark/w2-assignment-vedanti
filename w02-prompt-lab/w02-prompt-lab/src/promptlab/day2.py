"""Day 2 experiment: same summarization job on Mistral and Qwen via OllamaAdapter."""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from promptlab.adapters.base import CompletionRequest
from promptlab.adapters.ollama import OllamaAdapter
from promptlab.config import PROJECT_ROOT, Settings
from promptlab.errors import TruncatedResponseError
from promptlab.usage import CallRecord, append_record

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
LOGICAL_MODEL_KEYS: tuple[str, ...] = ("mistral", "qwen")
PROMPT_PATH = PROJECT_ROOT / "src" / "prompts" / "baseline.v0.md"
CASES_PATH = PROJECT_ROOT / "cases" / "summarization.jsonl"
EVIDENCE_PATH = PROJECT_ROOT / "docs" / "day2-run.jsonl"
COMPARISON_PATH = PROJECT_ROOT / "docs" / "day2-comparison.md"
DOCUMENT_OPEN = "<document>"
DOCUMENT_CLOSE = "</document>"
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


def split_baseline_prompt(template: str, source: str) -> tuple[str, str]:
    marker_at = template.find(DOCUMENT_OPEN)
    if marker_at < 0:
        raise ValueError("baseline prompt is missing document tags")
    system = template[:marker_at].rstrip()
    user_content = f"{DOCUMENT_OPEN}\n{source}\n{DOCUMENT_CLOSE}"
    return system, user_content


def _think_from_args(value: str | None) -> bool | None:
    if value is None:
        return None
    return value == "true"


def _think_label(think: bool | None) -> str:
    if think is None:
        return "omitted (Ollama model default)"
    return str(think).lower()


def load_evidence(path: Path) -> list[CallRecord]:
    records: list[CallRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(CallRecord.model_validate_json(line))
    return records


@dataclass(frozen=True)
class ModelStats:
    logical_name: str
    model_id: str
    attempts: int
    successes: int
    truncations: int
    other_errors: int
    input_tokens_sum: int
    input_tokens_mean: float
    output_tokens_sum: int
    output_tokens_mean: float
    latency_ms_mean: float
    latency_ms_median: float
    latency_ms_max: int
    cost_usd: float


def _mean(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _median(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def stats_for_model(
    logical_name: str,
    model_id: str,
    records: Sequence[CallRecord],
) -> ModelStats:
    rows = [record for record in records if record.model_id == model_id]
    successes = sum(1 for record in rows if record.error_type is None)
    truncations = sum(
        1 for record in rows if record.error_type == TruncatedResponseError.__name__
    )
    other_errors = len(rows) - successes - truncations
    input_tokens = [record.input_tokens for record in rows]
    output_tokens = [record.output_tokens for record in rows]
    latencies = [record.latency_ms for record in rows]
    costs = [record.cost_usd for record in rows]
    return ModelStats(
        logical_name=logical_name,
        model_id=model_id,
        attempts=len(rows),
        successes=successes,
        truncations=truncations,
        other_errors=other_errors,
        input_tokens_sum=sum(input_tokens),
        input_tokens_mean=_mean(input_tokens),
        output_tokens_sum=sum(output_tokens),
        output_tokens_mean=_mean(output_tokens),
        latency_ms_mean=_mean(latencies),
        latency_ms_median=_median(latencies),
        latency_ms_max=max(latencies) if latencies else 0,
        cost_usd=0.0 if not costs else max(costs),
    )


def render_comparison(records: Sequence[CallRecord], settings: Settings) -> str:
    stats_by_name = {
        logical_name: stats_for_model(
            logical_name,
            settings.models[logical_name].model_id,
            records,
        )
        for logical_name in LOGICAL_MODEL_KEYS
    }
    sections = [
        "# Day 2 model comparison",
        "",
        "Generated from `docs/day2-run.jsonl`. "
        "`cost_usd` is 0.0 for both configured models; this is not a dollar-cost race "
        "and no cost winner is named.",
        "",
        "`think` is not on `CompletionRequest`. Each experiment uses the same `think` "
        "setting for both models. The only intended difference inside an experiment is "
        "`model_id`.",
        "",
    ]
    for logical_name in LOGICAL_MODEL_KEYS:
        stats = stats_by_name[logical_name]
        sections.extend(
            [
                f"## {logical_name} (`{stats.model_id}`)",
                "",
                f"- Attempts: {stats.attempts}",
                f"- Successes: {stats.successes}",
                f"- Truncations: {stats.truncations}",
                f"- Other errors: {stats.other_errors}",
                f"- Input tokens: sum={stats.input_tokens_sum}, "
                f"mean={stats.input_tokens_mean:.2f}",
                f"- Output tokens: sum={stats.output_tokens_sum}, "
                f"mean={stats.output_tokens_mean:.2f}",
                f"- Latency (ms): mean={stats.latency_ms_mean:.2f}, "
                f"max={stats.latency_ms_max}",
                f"- Latency median (ms): {stats.latency_ms_median:.2f}",
                "- cost_usd: 0.0",
                "",
            ]
        )
    mistral = stats_by_name["mistral"]
    qwen = stats_by_name["qwen"]
    sections.extend(
        [
            "## Observation",
            "",
            (
                f"{mistral.logical_name} succeeded on {mistral.successes} of "
                f"{mistral.attempts} attempts (truncations={mistral.truncations}, "
                f"other errors={mistral.other_errors}) with {mistral.input_tokens_sum} "
                f"input tokens and {mistral.output_tokens_sum} output tokens; "
                f"median latency {mistral.latency_ms_median:.2f} ms, "
                f"max {mistral.latency_ms_max} ms. "
                f"{qwen.logical_name} succeeded on {qwen.successes} of "
                f"{qwen.attempts} attempts (truncations={qwen.truncations}, "
                f"other errors={qwen.other_errors}) with {qwen.input_tokens_sum} "
                f"input tokens and {qwen.output_tokens_sum} output tokens; "
                f"median latency {qwen.latency_ms_median:.2f} ms, "
                f"max {qwen.latency_ms_max} ms. "
                "These counts and latencies come from the JSONL records; both models "
                "recorded cost_usd=0.0, so this is a token and latency comparison, "
                "not a dollar-cost ranking. If Qwen truncates with think omitted, "
                "rerun with `--think false`; that overhead is thinking tokens, not "
                "the summarization task."
            ),
            "",
        ]
    )
    return "\n".join(sections).rstrip() + "\n"


def write_comparison(evidence_path: Path, markdown_path: Path, settings: Settings) -> None:
    records = load_evidence(evidence_path)
    markdown_path.write_text(render_comparison(records, settings), encoding="utf-8")


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
            system, user_content = split_baseline_prompt(template, cases[case_id])
            request = CompletionRequest(
                task="summarization",
                case_id=case_id,
                prompt_id="baseline",
                prompt_version="v0",
                system=system,
                user_content=user_content,
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

    run_path = Path("runs") / f"{run_id}.jsonl"
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if run_path.exists():
        shutil.copyfile(run_path, EVIDENCE_PATH)
        write_comparison(EVIDENCE_PATH, COMPARISON_PATH, settings)

    print(f"run_id={run_id}")
    print(f"log=runs/{run_id}.jsonl")
    print(f"evidence={EVIDENCE_PATH}")
    print(f"comparison={COMPARISON_PATH}")


if __name__ == "__main__":
    main()
