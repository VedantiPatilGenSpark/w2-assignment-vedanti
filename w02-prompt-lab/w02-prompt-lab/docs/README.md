# Day 2 evidence

Two summarization experiments. Same 12 cases (`S01`–`S12`), same baseline prompt, same temperature, same `max_output_tokens=512`, both models on Ollama. The only intended difference **within** each experiment is `model_id` (`mistral:7b` vs `qwen3:8b`).

`think` is not part of `CompletionRequest` or `ModelAdapter`. It is an optional `OllamaAdapter` constructor flag. Both models always get the same `think` value in a given experiment.

## Experiment 1 (primary / assignment filenames)

**Question:** With Ollama left at each model’s default thinking behavior, how do Mistral and Qwen compare on this set?

**Request body:** do not send a `think` field. Mistral 7B has no thinking mode. Qwen3 typically thinks when the field is omitted.

**How to run:**

```bash
uv run python -m promptlab.day2
```

**Evidence (grader-facing names):**

- `docs/day2-run.jsonl` — full 12×2 log for this experiment (`run_id=1b032c0f-fc78-4cf1-9cab-3cfb212a990a`)
- `docs/day2-comparison.md` — counts, tokens, latency for this experiment, plus experiment 2

Working copy: `runs/{run_id}.jsonl` (gitignored). Copy into `docs/day2-run.jsonl` after the run.

## Experiment 2 (follow-up)

**Question:** Under the same prompt and 512-token cap, how do they compare when thinking is explicitly off for both?

**Request body:** send `"think": false` for **both** models so the POST shape stays identical except `model`.

**How to run:**

```bash
uv run python -m promptlab.day2 --think false
```

**Evidence (extra; does not replace experiment 1):**

- `docs/day2-run-think-off.jsonl` (`run_id=09e2aada-265c-43fa-b635-52f5e24ae786`)

This is not a substitute for `docs/day2-run.jsonl`. It isolates whether Qwen3’s default thinking (not the summarization task alone) drove extra tokens, latency, and truncation in experiment 1.

## What stays out of the comparison

No dollar comparison. Local Ollama cost is `$0.00` for both models.
