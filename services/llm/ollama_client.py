from __future__ import annotations

import os

import httpx

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")


class LLMError(Exception): ...


def generate(
    prompt: str, model: str | None = None, temperature: float = 0.2, timeout_s: int = 60
) -> str:
    """Call Ollama /api/generate (non-streaming)."""
    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {
        "model": model or DEFAULT_MODEL,
        "prompt": prompt,
        "options": {"temperature": temperature},
        "stream": False,
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        text = (data or {}).get("response", "")
        if not text:
            raise LLMError("empty response from LLM")
        return text.strip()
    except Exception as e:  # noqa: BLE001
        raise LLMError(str(e))
