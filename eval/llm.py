"""
LLM client: send prompt, get raw response, parse to A/B/C/D.
Extend with more backends (e.g. Anthropic, local HuggingFace) by adding functions
and switching in run_eval.py.
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
    # Prefer last occurrence (model might reason then say "So the answer is B")
    matches = list(re.finditer(r"\b([A-D])\b", text))
    if not matches:
        return None
    return matches[-1].group(1)


def parse_reasoning(raw: str) -> str:
    """
    Extract the reasoning part: everything before the last A/B/C/D in the response.
    If no letter is found, returns the full response (treated as reasoning).
    """
    if not raw or not isinstance(raw, str):
        return ""
    text = raw.strip()
    matches = list(re.finditer(r"\b([A-D])\b", text, re.IGNORECASE))
    if not matches:
        return text
    # Everything before the last answer letter
    last_match = matches[-1]
    reasoning = text[: last_match.start()].strip()
    return reasoning


def get_answer_openai(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int | None = None) -> str:
    """Call OpenAI API; return raw response text. Requires OPENAI_API_KEY in env."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: pip install openai") from None

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY in the environment")

    client = OpenAI(api_key=api_key)
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def get_answer_gemini(prompt: str, model: str = "gemini-1.5-flash", temperature: float = 0.0, max_tokens: int | None = None) -> str:
    """Call Gemini API; return raw response text. Requires GEMINI_API_KEY in env."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("Install google-genai: uv add google-genai") from None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY in the environment")

    client = genai.Client(api_key=api_key)
    config_kwargs = {"temperature": temperature}
    if max_tokens is not None:
        config_kwargs["max_output_tokens"] = max_tokens
    config = types.GenerateContentConfig(**config_kwargs)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    text = getattr(response, "text", None)
    if text is None and response.candidates:
        text = response.candidates[0].content.parts[0].text
    return (text or "").strip()


def get_answer_ollama(prompt: str, model: str = "qwen2.5:0.5b", temperature: float = 0.0, max_tokens: int | None = None) -> str:
    """Call local Ollama (OpenAI-compatible). No API key. Start with: ollama run qwen2.5:0.5b"""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: uv add openai") from None

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    client = OpenAI(base_url=base_url, api_key="ollama")  # key ignored by Ollama
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def get_answer_hf(prompt: str, model: str = "Qwen/Qwen2.5-0.5B-Instruct", temperature: float = 0.0, max_tokens: int | None = None) -> str:
    """Run a Hugging Face model locally (e.g. small Qwen). Install with: uv sync --extra local"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError("Install local deps: uv sync --extra local") from None

    # Reuse model/tokenizer across calls (same process)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if get_answer_hf._model is None or get_answer_hf._model_name != model:
        get_answer_hf._model_name = model
        get_answer_hf._tokenizer = AutoTokenizer.from_pretrained(
            model, token=hf_token, trust_remote_code=True
        )
        get_answer_hf._model = AutoModelForCausalLM.from_pretrained(
            model, token=hf_token, dtype=torch.float32, device_map="auto", trust_remote_code=True
        )
    tokenizer = get_answer_hf._tokenizer
    llm = get_answer_hf._model

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


get_answer_hf._model = None
get_answer_hf._model_name = None
get_answer_hf._tokenizer = None


def get_answer(
    prompt: str,
    backend: str = "openai",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> str:
    """
    Single entry point: backend in ("openai", "gemini", "ollama", "hf"), returns raw response.
    When max_tokens is set (e.g. for reasoning=none), generation stops early and runs faster.
    """
    if backend == "openai":
        return get_answer_openai(prompt, model=model or "gpt-4o-mini", temperature=temperature, max_tokens=max_tokens)
    if backend == "gemini":
        return get_answer_gemini(prompt, model=model or "gemini-2.0-flash", temperature=temperature, max_tokens=max_tokens)
    if backend == "ollama":
        return get_answer_ollama(prompt, model=model or "qwen2.5:0.5b", temperature=temperature, max_tokens=max_tokens)
    if backend == "hf":
        return get_answer_hf(prompt, model=model or "Qwen/Qwen2.5-0.5B-Instruct", temperature=temperature, max_tokens=max_tokens)
    raise ValueError(f"Unknown backend: {backend}")
