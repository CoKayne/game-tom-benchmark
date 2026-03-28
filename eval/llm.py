"""
Hugging Face Transformers: load a causal LM, generate, parse A/B/C/D from output.
"""

import os
import re


def parse_option(raw: str) -> str | None:
    """
    Extract a single choice A/B/C/D from model output.
    Handles: "B", "Option B", "(B)", "answer: B", "The answer is B", etc.
    """
    if not raw or not isinstance(raw, str):
        return None
    text = raw.strip().upper()
    matches = list(re.finditer(r"\b([A-D])\b", text))
    if not matches:
        return None
    return matches[-1].group(1)


def get_answer(
    prompt: str,
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    """Run a Hugging Face causal LM locally. Set HF_TOKEN for gated models."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError("Install deps: uv sync") from e

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if get_answer._model is None or get_answer._model_name != model:
        get_answer._model_name = model
        get_answer._tokenizer = AutoTokenizer.from_pretrained(
            model, token=hf_token, trust_remote_code=True
        )
        get_answer._model = AutoModelForCausalLM.from_pretrained(
            model, token=hf_token, dtype=torch.float32, device_map="auto", trust_remote_code=True
        )
    tokenizer = get_answer._tokenizer
    llm = get_answer._model

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(llm.device)
    gen = llm.generate(
        **inputs,
        max_new_tokens=max_tokens if max_tokens is not None else 512,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else 1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    out = tokenizer.decode(gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return out.strip()


get_answer._model = None
get_answer._model_name = None
get_answer._tokenizer = None
