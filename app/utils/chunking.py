import re


def chunk_text(text: str, chunk_size: int = 420, overlap: int = 60) -> list[str]:
    words = re.findall(r"\S+", text)
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks
