#!/usr/bin/env python3
"""
Evaluation pipeline: load benchmark JSON → prompt each item → call LLM → parse answer → compute metrics → save.

Usage:
  python -m eval.run_eval --dataset ../avalon_testcases.json --output-dir ./results
  python -m eval.run_eval --dataset ../avalon_testcases.json --model gpt-4o --seed 42

Loads API keys from the environment or from a .env file in the project root (see .env.example).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from eval.llm import get_answer, parse_option, parse_reasoning
from eval.metrics import compute_metrics
from eval.prompt import build_prompt


def load_dataset(path: str | Path) -> list[dict]:
    """Load JSON dataset; return list of items (expects .items or single list)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "items" in data:
        return data["items"]
    if isinstance(data, list):
        return data
    raise ValueError("Dataset must have an 'items' key or be a list of items.")


def run_eval(
    items: list[dict],
    backend: str = "openai",
    model: str | None = None,
    temperature: float = 0.0,
    reasoning: str = "brief",
) -> list[dict]:
    """Run model on each item; return list of records (id, correct_option, model_option, correct, raw, reasoning, ...)."""
    records = []
    for i, item in enumerate(items):
        item_id = item.get("id", f"item_{i}")
        correct_option = item.get("correct_option", "").strip().upper()
        if correct_option not in ("A", "B", "C", "D"):
            print(f"Warning: {item_id} has invalid correct_option '{correct_option}', skipping.", file=sys.stderr)
            continue

        prompt = build_prompt(item, reasoning=reasoning)
        max_tokens = 32 if reasoning == "none" else None  # cap output so generation stops early
        try:
            raw = get_answer(prompt, backend=backend, model=model, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            print(f"Error on {item_id}: {e}", file=sys.stderr)
            records.append({
                "id": item_id,
                "correct_option": correct_option,
                "model_option": None,
                "correct": False,
                "raw": None,
                "reasoning": None,
                "error": str(e),
                **{k: item.get(k) for k in ("tom_type", "difficulty")},
            })
            continue

        model_option = parse_option(raw)
        reasoning = parse_reasoning(raw)
        correct = model_option == correct_option if model_option else False

        records.append({
            "id": item_id,
            "correct_option": correct_option,
            "model_option": model_option,
            "correct": correct,
            "reasoning": reasoning,
            "raw": raw,
            "tom_type": item.get("tom_type"),
            "difficulty": item.get("difficulty"),
        })
    return records


def main():
    # Load .env from project root so GEMINI_API_KEY / OPENAI_API_KEY are available
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")

    parser = argparse.ArgumentParser(description="Run ToM benchmark evaluation")
    parser.add_argument("--dataset", type=str, default="avalon_testcases.json", help="Path to benchmark JSON")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for results and logs")
    parser.add_argument("--backend", type=str, default="gemini", choices=["openai", "gemini", "ollama", "hf"], help="LLM backend (ollama/hf = local, no API key)")
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g. gpt-4o-mini, gpt-4o)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--reasoning", type=str, default="brief", choices=["none", "brief", "detailed"], help="Ask model for reasoning: none, brief, or detailed (default: brief)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (for reproducibility; backend-dependent)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of items to evaluate (for debugging)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        # Resolve relative to project root (parent of eval/)
        root = Path(__file__).resolve().parent.parent
        dataset_path = root / args.dataset
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    items = load_dataset(dataset_path)
    if args.limit:
        items = items[: args.limit]

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path(__file__).resolve().parent.parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    default_models = {"openai": "gpt-4o-mini", "gemini": "gemini-2.0-flash", "ollama": "qwen2.5:0.5b", "hf": "Qwen/Qwen2.5-0.5B-Instruct"}
    model = args.model or default_models.get(args.backend)
    print(f"Evaluating {len(items)} items with backend={args.backend} model={model} temperature={args.temperature}")

    records = run_eval(items, backend=args.backend, model=model, temperature=args.temperature, reasoning=args.reasoning)
    metrics = compute_metrics(records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{args.backend}_{(model or 'default').replace('/', '_')}_{timestamp}"
    results_file = output_dir / f"{run_name}_results.json"
    log_file = output_dir / f"{run_name}_log.json"

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "run": {"backend": args.backend, "model": model, "temperature": args.temperature, "reasoning": args.reasoning, "seed": args.seed}}, f, indent=2)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"Results: {results_file}")
    print(f"Per-item log: {log_file}")


if __name__ == "__main__":
    main()
