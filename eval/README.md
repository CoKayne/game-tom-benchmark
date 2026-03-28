# Evaluation pipeline (Hugging Face)

Pipeline: **load benchmark JSON → build prompt → local Transformers model → parse A/B/C/D → metrics → save results**.

## Setup

```bash
uv sync
```

Optional: copy `.env.example` to `.env` and set `HF_TOKEN` if you use a gated or private model.

## Run

From the project root:

```bash
uv run python -m eval.run_eval

# Another model
uv run python -m eval.run_eval --model Qwen/Qwen2.5-1.5B-Instruct

# Debug
uv run python -m eval.run_eval --limit 2
```

Without uv: `pip install .` then `python -m eval.run_eval`.

## Outputs

- **`results/run_hf_<model>_<timestamp>_results.json`**: metrics and run config.
- **`results/run_hf_<model>_<timestamp>_log.json`**: per-item `model_option`, `raw`, etc.

## Layout

| File | Role |
|------|------|
| `prompt.py` | Build prompt from each item. |
| `llm.py` | `get_answer` (HF), `parse_option`. |
| `metrics.py` | Accuracy and breakdowns. |
| `run_eval.py` | CLI entry point. |
