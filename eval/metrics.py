"""
Compute accuracy and optional breakdowns (by tom_type, difficulty).
"""

from collections import defaultdict


def compute_metrics(records: list[dict]) -> dict:
    """
    records: list of {"id", "correct_option", "model_option", "tom_type", "difficulty", ...}
    Returns: overall accuracy, counts, and optional breakdowns.
    """
    total = len(records)
    correct = sum(1 for r in records if r.get("correct") is True)
    accuracy = correct / total if total else 0.0

    by_tom = defaultdict(lambda: {"correct": 0, "total": 0})
    by_difficulty = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in records:
        c = 1 if r.get("correct") else 0
        by_tom[r.get("tom_type", "unknown")]["total"] += 1
        by_tom[r.get("tom_type", "unknown")]["correct"] += c
        by_difficulty[r.get("difficulty", "unknown")]["total"] += 1
        by_difficulty[r.get("difficulty", "unknown")]["correct"] += c

    def acc(d):
        return d["correct"] / d["total"] if d["total"] else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "by_tom_type": {k: {"total": v["total"], "correct": v["correct"], "accuracy": acc(v)} for k, v in by_tom.items()},
        "by_difficulty": {k: {"total": v["total"], "correct": v["correct"], "accuracy": acc(v)} for k, v in by_difficulty.items()},
    }
