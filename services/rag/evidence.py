from __future__ import annotations


def collect_evidence(parsed_docs: list[dict], max_chars: int = 1200) -> str:
    """Prefer table rows; fallback to first text chunks."""
    lines: list[str] = []
    for p in parsed_docs:
        if p.get("tables"):
            t = p["tables"][0]
            rows = t.get("rows", [])[:5]
            for row in rows:
                lines.append(" • " + ", ".join(f"{k}={v}" for k, v in list(row.items())[:6]))
    if not lines:
        for p in parsed_docs:
            chunks = p.get("chunks") or []
            if chunks:
                lines.append(chunks[0].get("text", "")[:400])
    text = "\n".join(lines)
    return text[:max_chars]
