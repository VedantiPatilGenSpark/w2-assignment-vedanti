# Day 4 notes: triage prompt comparison

Model: mistral:7b (logical name mistral), temperature 0.0, run_id d6d9557e-dfbe-4f3d-8984-a772afcbf075. Both prompt versions ran all 12 triage cases through complete_structured. provider/API cost = $0.00; token and latency overhead are measured from local Ollama call records.

triage.v1: queue 9/12, escalation 10/12, missed 2, unnecessary 0, boundary 12/12
triage.v2: queue 8/12, escalation 9/12, missed 2, unnecessary 1, boundary 12/12
changed queue between versions: 1
output tokens/case: v1 = 122.8, v2 = 184.8, delta = +62.0
median latency: v1 = 4.56s, v2 = 7.26s
maximum latency: v1 = 8.97s, v2 = 8.26s
observation count: v1 = 12, v2 = 12
conclusion: v2 spent extra output tokens and queue accuracy moved by one case; a one-case difference on 12 cases is a shrug, not a verdict, so the analysis field did not earn its overhead.
