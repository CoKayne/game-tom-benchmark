"""
Build a single prompt from a test item (game_rules, opponent_strategy, scenario, question, options).
"""

OPTION_LETTERS = ("A", "B", "C", "D")


def build_prompt(item: dict) -> str:
    """Turn one benchmark item into the text sent to the model."""
    parts = [
        "## Game rules",
        item["game_rules"],
        "",
        "## Opponent / table strategy",
        item["opponent_strategy"],
        "",
        "## Scenario",
        item["scenario"],
        "",
        "## Question",
        item["question"],
        "",
        "## Options (choose exactly one)",
    ]
    for letter, text in zip(OPTION_LETTERS, item["options"]):
        parts.append(f"{letter}. {text}")
    parts.append("")
    parts.append("Answer with only the letter of your choice (A, B, C, or D).")
    return "\n".join(parts)
