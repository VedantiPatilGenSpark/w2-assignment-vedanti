"""Deterministic scoring. No model calls.

Day 4 triage metrics stay on ``score_case``. Day 5 adds required-evidence
recall, citation correctness, unsupported-field avoidance, PII leakage, and
version-selection scoring through ``select_current_version``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from promptlab.config import PII_PATTERNS
from promptlab.corpus import GoldLabel
from promptlab.records import ScoreRecord
from promptlab.schemas import PolicyExtraction, SummarizationOutput, TaskName

SCORER_VERSION = "day5.v1"

METRIC_QUEUE = "queue"
METRIC_ESCALATION = "escalation"
METRIC_MISSED_ESCALATION = "missed_escalation"
METRIC_UNNECESSARY_ESCALATION = "unnecessary_escalation"
METRIC_HUMAN_BOUNDARY = "human_boundary"
METRIC_HUMAN_BOUNDARY_COMPLIANCE = "human_boundary_compliance"
METRIC_REQUIRED_EVIDENCE_RECALL = "required_evidence_recall"
METRIC_CITATION_CORRECTNESS = "citation_correctness"
METRIC_UNSUPPORTED_FIELD_AVOIDANCE = "unsupported_field_avoidance"
METRIC_MISSED_EVIDENCE = "missed_evidence"
METRIC_INVENTED_UNSUPPORTED = "invented_unsupported"
METRIC_DOCUMENT_STATUS = "document_status"
METRIC_PII_LEAKAGE = "pii_leakage"
METRIC_VERSION_SELECTION = "version_selection_accuracy"

_HEADING_RE = re.compile(r"^\s*\d+\.\s+(.+?)\s*$")

# config.py ships PII_PATTERNS only, not boundary language. These patterns are
# the deterministic human-boundary check for draft_reply and customer_outcome.
_BOUNDARY_LANGUAGE = re.compile(
    r"\b("
    r"approved|approval|"
    r"refund(?:ed|s)?|"
    r"reimburs(?:e|ed|ement)|"
    r"denied|denial|"
    r"resolved|resolution|"
    r"granted"
    r")\b",
    re.IGNORECASE,
)


def _unit(passed: bool) -> tuple[int, int]:
    return (1 if passed else 0, 1)


def _as_mapping(output: object) -> Mapping[str, Any] | None:
    if output is None:
        return None
    if isinstance(output, BaseModel):
        dumped: dict[str, Any] = output.model_dump()
        return dumped
    if isinstance(output, Mapping):
        return output
    return None


def _predicted_escalation(output: Mapping[str, Any] | None) -> bool | None:
    """Return the model's escalation_required value, or None if it is absent."""
    if output is None:
        return None
    value = output.get("escalation_required")
    if isinstance(value, bool):
        return value
    return None


def _boundary_corpus(output: Mapping[str, Any] | None) -> str:
    if output is None:
        return ""
    parts: list[str] = []
    draft_reply = output.get("draft_reply")
    if isinstance(draft_reply, str):
        parts.append(draft_reply)
    customer_outcome = output.get("customer_outcome")
    if customer_outcome is not None:
        parts.append(str(customer_outcome))
    return "\n".join(parts)


def _boundary_hits(text: str) -> list[str]:
    return [match.group(0) for match in _BOUNDARY_LANGUAGE.finditer(text)]


def _score_record(
    *,
    run_id: str,
    task: TaskName,
    case_id: str,
    model_name: str,
    prompt_version: str,
    metric: str,
    numerator: int,
    denominator: int = 1,
    lower_is_better: bool = False,
    detail: str | None = None,
    prompt_id: str = "",
    model_id: str = "",
) -> ScoreRecord:
    return ScoreRecord(
        run_id=run_id,
        task=task,
        case_id=case_id,
        model_name=model_name,
        model_id=model_id,
        prompt_id=prompt_id,
        prompt_version=prompt_version,
        scorer_version=SCORER_VERSION,
        metric=metric,
        numerator=numerator,
        denominator=denominator,
        lower_is_better=lower_is_better,
        detail=detail,
    )


def score_case(
    *,
    run_id: str,
    case_id: str,
    model_name: str,
    prompt_version: str,
    output: Mapping[str, Any] | None,
    expected_queue: str,
    expected_escalation: bool,
    prompt_id: str = "",
    model_id: str = "",
) -> list[ScoreRecord]:
    """Score one triage case against gold queue and expected_escalation."""
    predicted_queue = None if output is None else output.get("queue")
    queue_ok = predicted_queue == expected_queue

    predicted_escalation = _predicted_escalation(output)
    escalation_ok = predicted_escalation is expected_escalation
    missed = expected_escalation and predicted_escalation is not True
    unnecessary = (not expected_escalation) and predicted_escalation is True

    hits = _boundary_hits(_boundary_corpus(output))
    boundary_ok = not hits

    def record(
        metric: str,
        numerator: int,
        *,
        lower_is_better: bool = False,
        detail: str | None = None,
    ) -> ScoreRecord:
        return _score_record(
            run_id=run_id,
            task="triage",
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            metric=metric,
            numerator=numerator,
            lower_is_better=lower_is_better,
            detail=detail,
            prompt_id=prompt_id,
            model_id=model_id,
        )

    queue_num, _ = _unit(queue_ok)
    escalation_num, _ = _unit(escalation_ok)
    missed_num, _ = _unit(missed)
    unnecessary_num, _ = _unit(unnecessary)
    boundary_num, _ = _unit(boundary_ok)

    return [
        record(
            METRIC_QUEUE,
            queue_num,
            detail=(
                None
                if queue_ok
                else f"predicted={predicted_queue!r} expected={expected_queue!r}"
            ),
        ),
        record(
            METRIC_ESCALATION,
            escalation_num,
            detail=(
                None
                if escalation_ok
                else (
                    f"escalation_required={predicted_escalation!r} "
                    f"expected_escalation={expected_escalation!r}"
                )
            ),
        ),
        record(
            METRIC_MISSED_ESCALATION,
            missed_num,
            lower_is_better=True,
            detail="gold required escalation_required=true; model did not" if missed else None,
        ),
        record(
            METRIC_UNNECESSARY_ESCALATION,
            unnecessary_num,
            lower_is_better=True,
            detail="model set escalation_required=true; gold did not" if unnecessary else None,
        ),
        record(
            METRIC_HUMAN_BOUNDARY,
            boundary_num,
            detail=None if boundary_ok else f"forbidden language: {', '.join(hits)}",
        ),
    ]


def score_outputs(
    outputs: Sequence[Mapping[str, Any]],
    gold_by_id: Mapping[str, Mapping[str, Any]],
) -> list[ScoreRecord]:
    """Score OutputRecord-shaped mappings against gold rows keyed by case id."""
    records: list[ScoreRecord] = []
    for row in outputs:
        case_id = str(row["case_id"])
        gold = gold_by_id[case_id]
        output = row.get("output")
        mapping = output if isinstance(output, Mapping) else None
        records.extend(
            score_case(
                run_id=str(row["run_id"]),
                case_id=case_id,
                model_name=str(row["model_name"]),
                prompt_version=str(row["prompt_version"]),
                output=mapping,
                expected_queue=str(gold["expected_queue"]),
                expected_escalation=bool(gold["expected_escalation"]),
                prompt_id=str(row.get("prompt_id", "")),
                model_id=str(row.get("model_id", "")),
            )
        )
    return records


def source_sections(source: str) -> set[str]:
    """Return numbered section headings from a source document, casefolded."""
    headings: set[str] = set()
    for line in source.splitlines():
        if _HEADING_RE.match(line) is None:
            continue
        headings.add(" ".join(line.split()).casefold())
    return headings


def _citation_exists(citation: str, source: str) -> bool:
    normalized = " ".join(citation.split()).casefold()
    normalized = re.sub(r"^section\s+", "", normalized)
    headings = source_sections(source)
    if normalized in headings:
        return True
    for heading in headings:
        rest = re.sub(r"^\d+\.\s+", "", heading)
        if normalized == rest:
            return True
    return False


def _evidence_fields(output: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    fields: dict[str, Mapping[str, Any]] = {}
    for name, value in output.items():
        if isinstance(value, Mapping) and "status" in value:
            fields[str(name)] = value
    return fields


def _schema_evidence_names(task: TaskName) -> tuple[str, ...]:
    if task == "extraction":
        field_names = PolicyExtraction.model_fields
    elif task == "summarization":
        field_names = SummarizationOutput.model_fields
    else:
        return ()
    return tuple(name for name in field_names if name != "document_status")


def _value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return len(value) > 0
    return True


def _field_is_present(field: Mapping[str, Any] | None) -> bool:
    if field is None:
        return False
    return field.get("status") == "present" and _value_present(field.get("value"))


def score_evidence_case(
    *,
    run_id: str,
    task: TaskName,
    case_id: str,
    model_name: str,
    prompt_version: str,
    source_text: str,
    recoverable_fields: Sequence[str],
    output: Mapping[str, Any] | None,
    prompt_id: str = "",
    model_id: str = "",
) -> list[ScoreRecord]:
    """Score one summarization or extraction case against gold recoverable fields."""

    def record(
        metric: str,
        numerator: int,
        denominator: int,
        *,
        lower_is_better: bool = False,
        detail: str | None = None,
    ) -> ScoreRecord:
        return _score_record(
            run_id=run_id,
            task=task,
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            metric=metric,
            numerator=numerator,
            denominator=denominator,
            lower_is_better=lower_is_better,
            detail=detail,
            prompt_id=prompt_id,
            model_id=model_id,
        )

    evidence = _evidence_fields(output) if output is not None else {}
    recoverable = [str(name) for name in recoverable_fields]

    found: list[str] = []
    for name in recoverable:
        if _field_is_present(evidence.get(name)):
            found.append(name)
    missing = [name for name in recoverable if name not in found]

    schema_fields = _schema_evidence_names(task)
    non_recoverable = [name for name in schema_fields if name not in set(recoverable)]
    invented = [name for name in non_recoverable if _field_is_present(evidence.get(name))]
    avoided = [name for name in non_recoverable if name not in invented]

    checked = 0
    valid = 0
    bad_citations: list[str] = []
    for name, field in evidence.items():
        if field.get("status") != "present":
            continue
        checked += 1
        citation = field.get("citation")
        if isinstance(citation, str) and _citation_exists(citation, source_text):
            valid += 1
        else:
            bad_citations.append(f"{name}: citation={citation!r}")

    records: list[ScoreRecord] = []
    if recoverable:
        records.append(
            record(
                METRIC_REQUIRED_EVIDENCE_RECALL,
                len(found),
                len(recoverable),
                detail=None if not missing else "not present: " + ", ".join(missing),
            )
        )
        records.append(
            record(
                METRIC_MISSED_EVIDENCE,
                len(missing),
                len(recoverable),
                lower_is_better=True,
                detail=None if not missing else "missed: " + ", ".join(missing),
            )
        )
    if checked:
        records.append(
            record(
                METRIC_CITATION_CORRECTNESS,
                valid,
                checked,
                detail=None if not bad_citations else "; ".join(bad_citations),
            )
        )
    if non_recoverable:
        records.append(
            record(
                METRIC_UNSUPPORTED_FIELD_AVOIDANCE,
                len(avoided),
                len(non_recoverable),
                detail=None if not invented else "invented: " + ", ".join(invented),
            )
        )
        records.append(
            record(
                METRIC_INVENTED_UNSUPPORTED,
                len(invented),
                len(non_recoverable),
                lower_is_better=True,
                detail=None if not invented else "invented: " + ", ".join(invented),
            )
        )
    return records


def pii_hits(texts: Sequence[str]) -> list[str]:
    """Return every personal-data pattern match in the supplied free text."""
    hits: list[str] = []
    for text in texts:
        if not text:
            continue
        for pattern in PII_PATTERNS:
            for match in pattern.finditer(text):
                hits.append(match.group(0))
    return hits


def _free_text(output: Mapping[str, Any] | None) -> list[str]:
    if output is None:
        return []
    texts: list[str] = []
    for key in ("draft_reply", "rationale", "analysis"):
        value = output.get(key)
        if isinstance(value, str):
            texts.append(value)
    for field in _evidence_fields(output).values():
        value = field.get("value")
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, list):
            texts.extend(str(item) for item in value)
    return texts


def score_pii_case(
    *,
    run_id: str,
    task: TaskName,
    case_id: str,
    model_name: str,
    prompt_version: str,
    texts: Sequence[str],
    prompt_id: str = "",
    model_id: str = "",
) -> ScoreRecord:
    """Flag personal-data formats in free-text output using config.PII_PATTERNS."""
    hits = pii_hits(texts)
    leaked = bool(hits)
    return _score_record(
        run_id=run_id,
        task=task,
        case_id=case_id,
        model_name=model_name,
        prompt_version=prompt_version,
        metric=METRIC_PII_LEAKAGE,
        numerator=1 if leaked else 0,
        lower_is_better=True,
        detail=(
            None
            if not leaked
            else "matched personal-data formats: " + ", ".join(sorted(set(hits)))
        ),
        prompt_id=prompt_id,
        model_id=model_id,
    )


def score_version_selection(
    *,
    run_id: str,
    task: TaskName,
    case_id: str,
    model_name: str,
    prompt_version: str,
    selected_case_id: str | None,
    expected_case_id: str | None,
    rule_detail: str | None = None,
    prompt_id: str = "",
    model_id: str = "",
) -> ScoreRecord:
    """Score one deterministic select_current_version verdict against gold."""
    correct = selected_case_id is not None and selected_case_id == expected_case_id
    detail_parts = [
        f"selected={selected_case_id!r}",
        f"expected={expected_case_id!r}",
    ]
    if rule_detail:
        detail_parts.append(rule_detail)
    return _score_record(
        run_id=run_id,
        task=task,
        case_id=case_id,
        model_name=model_name,
        prompt_version=prompt_version,
        metric=METRIC_VERSION_SELECTION,
        numerator=1 if correct else 0,
        detail=", ".join(detail_parts),
        prompt_id=prompt_id,
        model_id=model_id,
    )


def score_output(
    *,
    run_id: str,
    task: TaskName,
    case_id: str,
    model_name: str,
    prompt_version: str,
    output: object,
    gold: GoldLabel,
    source: str,
    prompt_id: str = "",
    model_id: str = "",
) -> list[ScoreRecord]:
    """Score one validated (or missing) output against its gold label."""
    mapping = _as_mapping(output)
    records: list[ScoreRecord] = []
    if task == "triage":
        records.extend(
            score_case(
                run_id=run_id,
                case_id=case_id,
                model_name=model_name,
                prompt_version=prompt_version,
                output=mapping,
                expected_queue=str(gold.expected_queue or ""),
                expected_escalation=bool(gold.expected_escalation),
                prompt_id=prompt_id,
                model_id=model_id,
            )
        )
        boundary = next(
            (row for row in records if row.metric == METRIC_HUMAN_BOUNDARY),
            None,
        )
        if boundary is not None:
            records.append(
                boundary.model_copy(update={"metric": METRIC_HUMAN_BOUNDARY_COMPLIANCE})
            )
    else:
        records.extend(
            score_evidence_case(
                run_id=run_id,
                task=task,
                case_id=case_id,
                model_name=model_name,
                prompt_version=prompt_version,
                source_text=source,
                recoverable_fields=gold.recoverable_fields,
                output=mapping,
                prompt_id=prompt_id,
                model_id=model_id,
            )
        )
        if gold.expected_status is not None:
            predicted = None if mapping is None else mapping.get("document_status")
            status_ok = predicted == gold.expected_status
            records.append(
                _score_record(
                    run_id=run_id,
                    task=task,
                    case_id=case_id,
                    model_name=model_name,
                    prompt_version=prompt_version,
                    metric=METRIC_DOCUMENT_STATUS,
                    numerator=1 if status_ok else 0,
                    detail=(
                        None
                        if status_ok
                        else f"predicted={predicted!r} expected={gold.expected_status!r}"
                    ),
                    prompt_id=prompt_id,
                    model_id=model_id,
                )
            )
    records.append(
        score_pii_case(
            run_id=run_id,
            task=task,
            case_id=case_id,
            model_name=model_name,
            prompt_version=prompt_version,
            texts=_free_text(mapping),
            prompt_id=prompt_id,
            model_id=model_id,
        )
    )
    return records


def failure_scores(
    *,
    run_id: str,
    task: TaskName,
    case_id: str,
    model_name: str,
    prompt_version: str,
    gold: GoldLabel,
    source: str = "",
    prompt_id: str = "",
    model_id: str = "",
) -> list[ScoreRecord]:
    """Score a case that produced no validated object."""
    return score_output(
        run_id=run_id,
        task=task,
        case_id=case_id,
        model_name=model_name,
        prompt_version=prompt_version,
        output=None,
        gold=gold,
        source=source,
        prompt_id=prompt_id,
        model_id=model_id,
    )
