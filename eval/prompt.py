"""
Build a single prompt from a test item (game_rules, opponent_strategy, scenario, question, options).
"""

OPTION_LETTERS = ("A", "B", "C", "D")

# Closing instructions by --reasoning style
REASONING_INSTRUCTIONS = {
    "none": "Answer with only the letter of your choice (A, B, C, or D).",
    "brief": "First give brief reasoning for your choice, then on the last line answer with only the letter (A, B, C, or D).",
    "detailed": "First explain your reasoning step by step, then on the last line answer with only the letter (A, B, C, or D).",
}


def build_prompt(item: dict, reasoning: str = "brief") -> str:
    """Turn one benchmark item into the text sent to the model. reasoning: 'none' | 'brief' | 'detailed'."""
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
    parts.append(REASONING_INSTRUCTIONS.get(reasoning, REASONING_INSTRUCTIONS["brief"]))
    return "\n".join(parts)
