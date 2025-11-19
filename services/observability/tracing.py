from __future__ import annotations

import os
from typing import Any

_langfuse = None
try:
    from langfuse import Langfuse

    if os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"):
        _langfuse = Langfuse()
except Exception:
    _langfuse = None


def trace_llm(
    event: str, input_payload: dict[str, Any], output_text: str, tags: list[str] | None = None
):
    """No-op if Langfuse not configured."""
    if not _langfuse:
        return
    try:
        obs = _langfuse.generation(
            name=event,
            input=input_payload,
            output=output_text,
            model=input_payload.get("model", ""),
            tags=tags or [],
        )
        obs.end()
    except Exception:
        pass
