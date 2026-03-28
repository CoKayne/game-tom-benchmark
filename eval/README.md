# Evaluation pipeline

Pipeline: **load benchmark JSON → build prompt per item → call LLM → parse A/B/C/D → compute accuracy and breakdowns → save results and per-item log**.

## Layout

| File | Role |
|------|------|
| `prompt.py` | Build one prompt from `game_rules`, `opponent_strategy`, `scenario`, `question`, `options`. |
| `llm.py` | `get_answer(prompt, backend, model, temperature)` → raw text; `parse_option(raw)` → A/B/C/D. |
| `metrics.py` | `compute_metrics(records)` → overall accuracy + by `tom_type` and `difficulty`. |
| `run_eval.py` | CLI: load dataset, run model on each item, write `*_results.json` and `*_log.json`. |

## Setup (uv)

From the project root:

```bash
# Create venv and install dependencies (no global pip needed)
uv sync

# API keys: put them in a .env file (copy from .env.example)
#   cp .env.example .env   # then edit .env and add your keys
# Keys are loaded automatically when you run the pipeline.
# Or set in the shell: set GEMINI_API_KEY=...  (Windows) / export GEMINI_API_KEY=...  (macOS/Linux)
```

## Run (from project root)

```bash
# Default: Gemini (gemini-2.0-flash), avalon_testcases.json, results in ./results
uv run python -m eval.run_eval

# Explicit Gemini with another model
uv run python -m eval.run_eval --backend gemini --model gemini-1.5-flash

# Use OpenAI instead
uv run python -m eval.run_eval --backend openai --model gpt-4o-mini

# Local model (no API key): Ollama or Hugging Face
# See "Local models" below.
uv run python -m eval.run_eval --backend ollama --model qwen2.5:0.5b
uv run python -m eval.run_eval --backend hf --model Qwen/Qwen2.5-0.5B-Instruct

# Custom dataset and output
uv run python -m eval.run_eval --dataset avalon_testcases.json --output-dir ./results

# Different model / temperature
uv run python -m eval.run_eval --model gemini-2.0-flash --temperature 0

# Debug: run only first 2 items
uv run python -m eval.run_eval --limit 2
```

Without uv: from the project root run `pip install .` then `python -m eval.run_eval`.

## Local models (no API key)

### Option 1: Ollama (recommended)

[Ollama](https://ollama.com) runs models locally and exposes an OpenAI-compatible API. No API key needed.

1. Install Ollama and pull a small Qwen model:
   ```bash
   ollama pull qwen2.5:0.5b
   ```
2. Run the eval (Ollama must be running in the background):
   ```bash
   uv run python -m eval.run_eval --backend ollama --model qwen2.5:0.5b
   ```
3. Optional: set `OLLAMA_BASE_URL` in `.env` if Ollama is on another host/port (default: `http://localhost:11434/v1`).

### Option 2: Hugging Face (in-process)

Run a small model (e.g. Qwen) directly in Python with Transformers. No separate server; first run will download the model.

1. Install the optional `local` dependencies (torch, transformers, accelerate):
   ```bash
   uv sync --extra local
   ```
2. Run the eval (model loads on first item; needs enough RAM/VRAM):
   ```bash
   uv run python -m eval.run_eval --backend hf --model Qwen/Qwen2.5-0.5B-Instruct
   ```
   Other small models: `Qwen/Qwen2.5-1.5B-Instruct`, `HuggingFaceTB/SmolLM2-360M-Instruct`, etc.
3. **Gated or private models**: add your Hugging Face token to `.env` as `HF_TOKEN=your-token` (or set `HUGGING_FACE_HUB_TOKEN`). Get a token at https://huggingface.co/settings/tokens. The pipeline loads `.env` before running, so the token is used when downloading the model.

## Outputs

- **`results/<run>_results.json`**: `metrics` (accuracy, by_tom_type, by_difficulty) and `run` (backend, model, temperature, seed).
- **`results/<run>_log.json`**: One object per item: `id`, `correct_option`, `model_option`, `correct`, `raw`, `tom_type`, `difficulty`. Use for error analysis.

## Extending the pipeline

1. **Another LLM backend**  
   In `llm.py`, add e.g. `get_answer_anthropic(...)` and in `get_answer()` branch on `backend == "anthropic"`. Same contract: prompt in, raw string out.

2. **Different prompt format**  
   Edit `prompt.py` (e.g. few-shot, or different section order). Keep the instruction “Answer with only the letter (A/B/C/D)” so `parse_option()` still works.

3. **Stricter parsing**  
   If models often output “A.” or “Answer: A”, extend the regex in `parse_option()` in `llm.py`; fallback to first/last A–D if needed.

4. **Seeds**  
   `--seed` is stored in results; applying it is backend-specific (e.g. OpenAI does not expose a seed; for local models, set `torch.manual_seed(seed)` before each call).

5. **Multiple runs**  
   Run `python -m eval.run_eval` several times with different `--model` or `--output-dir`; then aggregate or compare the `*_results.json` files.
