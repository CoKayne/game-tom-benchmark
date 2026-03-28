# Testbench Specification

## 1. Research Goals and Motivation

**Research question**  
This testbench serves as a **Theory of Mind (ToM) benchmark in the board-game domain**. The main question is: *How well do LLMs perform on ToM reasoning when the context is game rules, opponent strategy, and in-game scenarios?*

**Expected contribution**
- Compare game-domain ToM performance with existing ToM benchmarks (e.g. social / narrative domains) to see if the game setting yields different behaviour.
- Evaluate different model families (e.g. general-purpose vs. math-reasoning-focused). If performance differs systematically, follow-up work can analyse *where* that difference comes from (e.g. layers, neurons, or attention patterns) to better understand how ToM is implemented in these models.

---

## 2. Scope and Boundaries

**In scope**
- LLMs (specify families and sizes when you pick them, e.g. GPT-4, Claude, Llama, Mistral, and/or math-specialised models).
- Single-turn, multiple-choice ToM questions in a board-game setting: model receives game rules, opponent strategy, scenario, and chooses among 4 options.
- Evaluation metrics and reproducibility setup for this benchmark.

**Out of scope**
- Multi-turn dialogue or interactive play; only single decision per item.
- Other modalities (e.g. images of the board); input is text (and structured JSON) only.
- Explaining *why* a model fails (e.g. interpretability) is follow-up; the testbench itself focuses on *whether* and *how well* models perform.

---

## 3. Inputs and Data

**Input format**  
Text, structured as JSON. Each item contains:

| Field | Description |
|-------|-------------|
| `game_rules` | Text description of the game rules. |
| `opponent_strategy` | Description of the opponent’s strategy or behaviour. |
| `scenario` | The in-game situation (board state, history, or context) where a decision is needed. |
| `question` or `prompt` | The actual task (e.g. “What should you do?” or “What does the opponent believe?”). |

**Response format**  
Single choice among **4 options** (A/B/C/D). Option keys and exact field names can be fixed when you lock the schema (e.g. `options` + `correct_index` or `correct_option`).

Example shape (adjust keys as needed):

```json
{
  "game_rules": "...",
  "opponent_strategy": "...",
  "scenario": "...",
  "question": "...",
  "options": ["A", "B", "C", "D"],
  "correct_option": "B"
}
```

**Data sources**  
Custom-designed. Items are created from:
- Official or clearly stated game rules.
- Common strategic situations and opponent behaviours that players encounter in practice.

**Scale**  
- **Suggested starting point**: e.g. 100–300 items (adjust by game complexity and labour). Can grow later.
- **Sampling**: Define a small set of games first; optionally stratify by game, difficulty, or ToM type (e.g. belief vs. intention) for balanced analysis.
- **Splits**: Reserve a fixed test set (and optionally dev/val) for reporting; do not train or tune on test.

---

## 4. Evaluation and Metrics

**Primary metric**  
- **Accuracy**: Fraction of items where the model’s chosen option matches the correct option (exact match on A/B/C/D).

**Secondary / reference metrics**  
- Accuracy by game, by difficulty (if labelled), or by ToM subtype.
- Consistency: same model, same item, multiple runs (with fixed seed) to check stability.

**Baselines**  
- **Random**: 25% (uniform over 4 options).
- **Human** (optional): Small pilot with human participants on a subset to establish an upper bound.
- **Existing ToM benchmarks**: Report results on a standard ToM benchmark for the same models so you can compare “game ToM” vs. “other ToM” in the same write-up.

---

## 5. Experiment Design

**Variables**  
- **Model**: Different LLMs (and optionally same model, different checkpoints or settings).
- **Temperature**: Use low (e.g. 0 or 0.2) for evaluation to reduce randomness; optionally vary for sensitivity.
- **Prompt format**: One primary prompt template; optionally compare 1–2 variants (e.g. few-shot vs. zero-shot).

**Control vs. treatment**  
- “Control”: baseline(s) above (random, and optionally a simple non-LLM baseline if you add one).
- “Treatment”: each LLM under your chosen setting. Compare accuracy (and secondary metrics) across models and, if applicable, across domains (game vs. other ToM).

**Repetition and randomness**  
- Fix random seed(s) for reproducibility.
- For accuracy: 1 run per item is typical; if you report consistency, run each item 2–3 times and report agreement or variance.
- No train/val/test in the “training” sense; only a fixed test (and optional dev) set for evaluation.

---

## 6. Environment and Implementation

**Environment**  
- Document: hardware (GPU type and count), OS, Python version, and key libraries (e.g. `transformers`, `torch`, `accelerate`).
- Document Hugging Face model IDs used for evaluation and optional `HF_TOKEN` for gated models.

**Interface**  
- Scripts or notebooks that: load the JSON dataset → run local Hugging Face inference → parse the chosen option → compute metrics. CLI entry point: `python -m eval.run_eval`.

**Reproducibility**  
- `pyproject.toml` / `uv.lock` (or equivalent) with pinned versions.
- README with: how to obtain data (or generate from your spec), how to run evaluation, and which seeds/settings were used.
- Optionally: Docker or a single script that runs the full evaluation.

---

## 7. Outputs and Reporting

**Output format**  
- **Results file**: Per-model (and per run, if multiple) accuracy and any secondary metrics; e.g. CSV or JSON.
- **Logs**: Model outputs (chosen option per item) so you can analyse errors later.
- **Figures**: e.g. bar chart of accuracy by model; optional: by game or by ToM type.

**Interpretation**  
- “Pass” or “strong performance”: define relative to baselines (e.g. clearly above random and, if available, close to or above human on the subset you measured).
- Report confidence intervals or standard errors if you run multiple seeds or subsamples.

---

## 8. Timeline and Milestones (optional)

- *Fill in with your target dates, e.g.:*
  - **Data**: Finalise game(s), design and review items, lock test set.
  - **Implementation**: Data loader, model integration, metrics script.
  - **Runs**: Complete evaluation for all models and baselines.
  - **Analysis**: Tables, figures, and draft write-up.

---

*Remove or shorten sections that don’t apply; adapt to your lab or advisor’s conventions.*
