# w2-assignment

Student copy of the Week 2 local-model prompt lab (Day 1). Work lives on the `w2-day1` branch.

The course project is nested at [`w02-prompt-lab/w02-prompt-lab/`](w02-prompt-lab/w02-prompt-lab/). Full setup (devcontainer, Ollama on the host Mac, `uv`) is in that folder’s [README](w02-prompt-lab/w02-prompt-lab/README.md). Assignment text: [`assignments/W02_Day1_Assignment_LOCAL.md`](w02-prompt-lab/w02-prompt-lab/assignments/W02_Day1_Assignment_LOCAL.md).

## Day 1 deliverables

- [`src/promptlab/usage.py`](w02-prompt-lab/w02-prompt-lab/src/promptlab/usage.py) — `CallRecord`, `compute_cost`, `append_record`
- [`src/promptlab/day1.py`](w02-prompt-lab/w02-prompt-lab/src/promptlab/day1.py) — Mistral extraction run
- [`docs/day1-run.jsonl`](w02-prompt-lab/w02-prompt-lab/docs/day1-run.jsonl) — three successful records
- [`docs/day1-observations.md`](w02-prompt-lab/w02-prompt-lab/docs/day1-observations.md)

## Run from the inner project

Python in the devcontainer talks to Ollama on the Mac at `http://host.docker.internal:11434`. From `w02-prompt-lab/w02-prompt-lab/`:

```bash
export UV_PROJECT_ENVIRONMENT=.venv
uv sync --frozen
uv run pytest tests/test_usage_contract.py
uv run python -m promptlab.day1
```
