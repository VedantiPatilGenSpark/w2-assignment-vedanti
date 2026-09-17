# Model Comparison

Run ID: `day5-local-01`

Counts are reported with their denominators. Latency uses median and maximum rather than mean.

## Extraction

| Model | Prompt | Valid outputs | Metrics | Input tokens/case | Output tokens/case | Median latency | Max latency | n | Repairs | Retries | Final failures |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | extract.v2 | 12/12 | citation_correctness: 73/73<br>document_status: 9/12<br>invented_unsupported: 2/12 ↓<br>missed_evidence: 1/72 ↓<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 71/72<br>unsupported_field_avoidance: 10/12<br>version_selection_accuracy: 1/1 | 1841.9 | 395.1 | 18275 ms | 20565 ms | 12 | 0/12 | 0 | 0 |
| qwen | extract.v2 transfer | 12/12 | citation_correctness: 72/72<br>document_status: 11/12<br>invented_unsupported: 1/12 ↓<br>missed_evidence: 1/72 ↓<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 71/72<br>unsupported_field_avoidance: 11/12<br>version_selection_accuracy: 1/1 | 1567.9 | 176.6 | 10253.5 ms | 13410 ms | 12 | 0/12 | 0 | 0 |

## Summarization

| Model | Prompt | Valid outputs | Metrics | Input tokens/case | Output tokens/case | Median latency | Max latency | n | Repairs | Retries | Final failures |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | summarize.v1 | 12/12 | citation_correctness: 57/62<br>document_status: 10/12<br>invented_unsupported: 3/12 ↓<br>missed_evidence: 1/60 ↓<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 59/60<br>unsupported_field_avoidance: 9/12<br>version_selection_accuracy: 1/1 | 1010.9 | 304.7 | 13312 ms | 23097 ms | 13 | 1/12 | 0 | 0 |
| qwen | summarize.v1 transfer | 12/12 | citation_correctness: 63/63<br>document_status: 10/12<br>invented_unsupported: 3/12 ↓<br>missed_evidence: 0/60 ↓<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 60/60<br>unsupported_field_avoidance: 9/12<br>version_selection_accuracy: 1/1 | 783.2 | 250.2 | 12128 ms | 15160 ms | 12 | 0/12 | 0 | 0 |

## Triage

| Model | Prompt | Valid outputs | Metrics | Input tokens/case | Output tokens/case | Median latency | Max latency | n | Repairs | Retries | Final failures |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | triage.v1 | 12/12 | escalation: 10/12<br>human_boundary: 12/12<br>human_boundary_compliance: 12/12<br>missed_escalation: 2/12 ↓<br>pii_leakage: 0/12 ↓<br>queue: 9/12<br>unnecessary_escalation: 0/12 ↓ | 682.8 | 122.8 | 4689 ms | 8857 ms | 12 | 0/12 | 0 | 0 |
| qwen | triage.v1 transfer | 12/12 | escalation: 12/12<br>human_boundary: 12/12<br>human_boundary_compliance: 12/12<br>missed_escalation: 0/12 ↓<br>pii_leakage: 0/12 ↓<br>queue: 12/12<br>unnecessary_escalation: 0/12 ↓ | 606.6 | 103.8 | 4589 ms | 7245 ms | 12 | 0/12 | 0 | 0 |

## Limits

- Each task has 12 cases. Counts are directional, not production-scale estimates.
- A one-case gap such as 11/12 versus 10/12 is not a universal model ranking.
- A row measures the model together with the prompt version shown in that row.
- Prompt-transfer rows are labeled `transfer`. They are evidence about that prompt on the second model, not proof of the model's best adapted performance.
- Untested combinations (other prompt versions, other decoding settings) are not claimed.
- Local Ollama latency depends on lab hardware and is not a production SLO.
- Local Ollama provider/API charge is `$0.00`; token usage and latency still represent real operational work.
