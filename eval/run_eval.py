#!/usr/bin/env python3
"""
Evaluation: load benchmark JSON → prompt each item → Hugging Face generate → parse answer → metrics → save.

Usage:
  uv run python -m eval.run_eval
  uv run python -m eval.run_eval --model Qwen/Qwen2.5-1.5B-Instruct --limit 2

Loads HF_TOKEN from .env (optional) for gated models.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from eval.llm import get_answer, parse_option
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
    model: str | None = None,
    temperature: float = 0.0,
) -> list[dict]:
    """Run Hugging Face model on each item; return list of records."""
    records = []
    default_model = "Qwen/Qwen2.5-0.5B-Instruct"
    model = model or default_model

    for i, item in enumerate(items):
        item_id = item.get("id", f"item_{i}")
        correct_option = item.get("correct_option", "").strip().upper()
        if correct_option not in ("A", "B", "C", "D"):
            print(f"Warning: {item_id} has invalid correct_option '{correct_option}', skipping.", file=sys.stderr)
            continue

        prompt = build_prompt(item)
        try:
            raw = get_answer(prompt, model=model, temperature=temperature, max_tokens=32)
        except Exception as e:
            print(f"Error on {item_id}: {e}", file=sys.stderr)
            records.append({
                "id": item_id,
                "correct_option": correct_option,
                "model_option": None,
                "correct": False,
                "raw": None,
                "error": str(e),
                **{k: item.get(k) for k in ("tom_type", "difficulty")},
            })
            continue

        model_option = parse_option(raw)
        correct = model_option == correct_option if model_option else False

        records.append({
            "id": item_id,
            "correct_option": correct_option,
            "model_option": model_option,
            "correct": correct,
            "raw": raw,
            "tom_type": item.get("tom_type"),
            "difficulty": item.get("difficulty"),
        })
    return records


def main():
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")

    parser = argparse.ArgumentParser(description="Run ToM benchmark (Hugging Face)")
    parser.add_argument("--dataset", type=str, default="avalon_testcases.json", help="Path to benchmark JSON")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for results and logs")
    parser.add_argument("--model", type=str, default=None, help="Hugging Face model id (default: Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    parser.add_argument("--seed", type=int, default=None, help="Stored in results JSON (for your notes)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of items (debug)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = root / args.dataset
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    items = load_dataset(dataset_path)
    if args.limit:
        items = items[: args.limit]

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model = args.model or "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Evaluating {len(items)} items with model={model} temperature={args.temperature}")

    records = run_eval(items, model=model, temperature=args.temperature)
    metrics = compute_metrics(records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = model.replace("/", "_")
    run_name = f"run_hf_{safe_model}_{timestamp}"
    results_file = output_dir / f"{run_name}_results.json"
    log_file = output_dir / f"{run_name}_log.json"

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": metrics,
            "run": {
                "backend": "hf",
                "model": model,
                "temperature": args.temperature,
                "seed": args.seed,
            },
        }, f, indent=2)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    print(f"Results: {results_file}")
    print(f"Per-item log: {log_file}")


if __name__ == "__main__":
    main()
