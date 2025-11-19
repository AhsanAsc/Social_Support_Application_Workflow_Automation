from typing import List


def simple_text_chunk(text: str, max_chars: int = 800) -> List[str]:
    """
    Naive text chunker – splits text into roughly max_chars chunks.
    Later: make this doc-type aware (tables vs narrative etc.).
    """
    chunks: List[str] = []
    current = []

    for line in text.splitlines():
        if sum(len(x) for x in current) + len(line) > max_chars:
            chunks.append("\n".join(current))
            current = []
        current.append(line)

    if current:
        chunks.append("\n".join(current))

    return chunks
