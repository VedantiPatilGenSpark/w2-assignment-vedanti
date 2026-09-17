# Model Comparison

Run ID: `day5-local-02`

Counts are reported with their denominators. Latency uses median and maximum rather than mean. Attempt latency is one HTTP POST; case latency is one `complete_structured` wall clock.

## Extraction

| Model | Prompt | Valid outputs | Metrics | Input tokens/case | Output tokens/case | Median latency | Max latency | n | Median case latency | Max case latency | n cases | Repairs | Retries | Final failures |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | extract.v2 | 12/12 | citation_correctness: 73/73<br>document_status: 9/12<br>invented_unsupported: 2/12 ↓<br>missed_evidence: 1/72 ↓<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 71/72<br>unsupported_field_avoidance: 10/12<br>version_selection_accuracy: 1/1 | 1841.9 | 395.1 | 18957 ms | 23434 ms | 12 | 18958.5 ms | 23435 ms | 12 | 0/12 | 0 | 0 |
| qwen | extract.v2 transfer | 12/12 | citation_correctness: 72/72<br>document_status: 11/12<br>invented_unsupported: 1/12 ↓<br>missed_evidence: 1/72 ↓<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 71/72<br>unsupported_field_avoidance: 11/12<br>version_selection_accuracy: 1/1 | 1567.9 | 176.6 | 11030 ms | 14283 ms | 12 | 11035 ms | 14284 ms | 12 | 0/12 | 0 | 0 |

## Summarization

| Model | Prompt | Valid outputs | Metrics | Input tokens/case | Output tokens/case | Median latency | Max latency | n | Median case latency | Max case latency | n cases | Repairs | Retries | Final failures |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | summarize.v1 | 12/12 | citation_correctness: 57/62<br>document_status: 10/12<br>invented_unsupported: 3/12 ↓<br>missed_evidence: 1/60 ↓<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 59/60<br>unsupported_field_avoidance: 9/12<br>version_selection_accuracy: 1/1 | 1010.9 | 304.7 | 13363 ms | 18126 ms | 13 | 13946.5 ms | 18130 ms | 12 | 1/12 | 0 | 0 |
| qwen | summarize.v1 transfer | 12/12 | citation_correctness: 63/63<br>document_status: 10/12<br>invented_unsupported: 3/12 ↓<br>missed_evidence: 0/60 ↓<br>pii_leakage: 0/12 ↓<br>required_evidence_recall: 60/60<br>unsupported_field_avoidance: 9/12<br>version_selection_accuracy: 1/1 | 783.2 | 250.2 | 13195 ms | 17669 ms | 12 | 13197 ms | 17669 ms | 12 | 0/12 | 0 | 0 |

## Triage

| Model | Prompt | Valid outputs | Metrics | Input tokens/case | Output tokens/case | Median latency | Max latency | n | Median case latency | Max case latency | n cases | Repairs | Retries | Final failures |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mistral | triage.v1 | 12/12 | escalation: 10/12<br>human_boundary: 12/12<br>human_boundary_compliance: 12/12<br>missed_escalation: 2/12 ↓<br>pii_leakage: 0/12 ↓<br>queue: 9/12<br>unnecessary_escalation: 0/12 ↓ | 682.8 | 122.8 | 5043.5 ms | 9455 ms | 12 | 5044 ms | 9455 ms | 12 | 0/12 | 0 | 0 |
| qwen | triage.v1 transfer | 12/12 | escalation: 12/12<br>human_boundary: 12/12<br>human_boundary_compliance: 12/12<br>missed_escalation: 0/12 ↓<br>pii_leakage: 0/12 ↓<br>queue: 12/12<br>unnecessary_escalation: 0/12 ↓ | 606.6 | 103.8 | 5172.5 ms | 9392 ms | 12 | 5173 ms | 9397 ms | 12 | 0/12 | 0 | 0 |

## Limits

- Each task has 12 cases. Counts are directional, not production-scale estimates.
- A one-case gap such as 11/12 versus 10/12 is not a universal model ranking.
- A row measures the model together with the prompt version shown in that row.
- Prompt-transfer rows are labeled `transfer`. They are evidence about that prompt on the second model, not proof of the model's best adapted performance.
- Untested combinations (other prompt versions, other decoding settings) are not claimed.
- Local Ollama latency depends on lab hardware and is not a production SLO.
- Attempt latency (`n`) is one HTTP POST. Case latency (`n cases`) is one `complete_structured` call, including transport retries, backoff, validation, and at most one repair. A repair raises attempt `n` without adding a case.
- Local Ollama provider/API charge is `$0.00`; token usage and latency still represent real operational work.
