"""
LLM factory. Priority: Groq (GROQ_API_KEY) > Hugging Face (HF_TOKEN) > None (rule-based mode).
Model can be forced with GROQ_MODEL, otherwise it is auto-selected from the account's available models.
"""
import os
from typing import Any, Optional

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
# Preference order when GROQ_MODEL is not set. Groq retires models regularly
# (llama-3.3-70b-versatile and llama-3.1-8b-instant were shut down on 2026-08-16),
# so the available list is queried from the API instead of hard-coding one name.
GROQ_PREFERRED = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "qwen/qwen3.6-27b"]


def _pick_groq_model(api_key: str) -> str:
    """GROQ_MODEL env var wins; otherwise pick the first preferred model the account can use."""
    forced = os.getenv("GROQ_MODEL")
    if forced:
        return forced
    try:
        import json, urllib.request
        req = urllib.request.Request("https://api.groq.com/openai/v1/models",
                                     headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            available = {m["id"] for m in json.load(r)["data"]}
        for m in GROQ_PREFERRED:
            if m in available:
                return m
        print(f"Warning: none of {GROQ_PREFERRED} available. Set GROQ_MODEL. Available: {sorted(available)}")
    except Exception as e:
        print(f"Warning: could not list Groq models ({e}); using {DEFAULT_GROQ_MODEL}")
    return DEFAULT_GROQ_MODEL


def get_llm(api_key: Optional[str] = None, temperature: float = 0.1, max_tokens: int = 1000) -> Any:
    """Return a LangChain LLM, or None if no key/provider is usable."""
    groq_key = api_key if (api_key and not api_key.startswith("hf_")) else os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=_pick_groq_model(groq_key),
                api_key=groq_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"Warning: could not initialize Groq LLM: {e}")
            return None

    hf_token = api_key if (api_key or "").startswith("hf_") else os.getenv("HF_TOKEN")
    if hf_token:
        try:
            from langchain_huggingface import HuggingFaceEndpoint
            return HuggingFaceEndpoint(
                repo_id=os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"),
                huggingfacehub_api_token=hf_token,
                temperature=temperature,
                max_new_tokens=max_tokens,
            )
        except Exception as e:
            print(f"Warning: could not initialize HF LLM: {e}")
    return None


def call_llm(llm: Any, prompt: str) -> str:
    """Invoke the LLM and always return plain text (ChatModels return a message object)."""
    out = llm.invoke(prompt)
    return out if isinstance(out, str) else str(getattr(out, "content", out))
