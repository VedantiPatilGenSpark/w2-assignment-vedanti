# Day 2 comparison

Same 12 summarization cases (`S01`–`S12`), baseline prompt `v0`, temperature `0.0`, `max_output_tokens=512`, provider `ollama`. Cost is `$0.00` for every row; there is no dollar comparison.

`think` is not on `CompletionRequest`. Each experiment uses the same `think` setting for both models. The only intended difference inside an experiment is `model_id`.

## Experiment 1 (primary)

Ollama `think` field omitted (each model’s default). Log: `docs/day2-run.jsonl` (`run_id=1b032c0f-fc78-4cf1-9cab-3cfb212a990a`). 24 rows, all `attempt=1`.

| Model | Finished (no error) | Truncated | Mean input tokens | Mean output tokens | Mean latency (ms) |
|---|---|---|---|---|---|
| `mistral:7b` | 12 / 12 | 0 | 232 | 95 | 3948 |
| `qwen3:8b` | 10 / 12 | 2 (`S06`, `S10`, `done_reason=length`, 512 output tokens) | 202 | 401 | 16824 |

Mistral stayed well under the 512 cap (67–126 output tokens). Qwen3 used about 4× as many output tokens and about 4× the latency, and hit the cap twice. That matches Qwen3’s default thinking: omitted `think` usually means thinking is on, and those tokens count toward `num_predict`.

## Experiment 2 (follow-up)

`"think": false` sent for **both** models. Log: `docs/day2-run-think-off.jsonl` (`run_id=09e2aada-265c-43fa-b635-52f5e24ae786`). 24 rows, all `attempt=1`.

| Model | Finished (no error) | Truncated | Mean input tokens | Mean output tokens | Mean latency (ms) |
|---|---|---|---|---|---|
| `mistral:7b` | 12 / 12 | 0 | 232 | 95 | 3811 |
| `qwen3:8b` | 12 / 12 | 0 | 208 | 62 | 2926 |

Mistral is essentially unchanged (no thinking mode to turn off). Qwen3 finishes every case, drops to ~62 mean output tokens and ~3s mean latency, and no longer truncates. Experiment 1’s extra Qwen tokens, latency, and two failures were from default thinking, not from the summarization task needing a larger cap.

See `docs/README.md` for how the two runs were produced.
