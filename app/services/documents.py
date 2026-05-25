import json
from pathlib import Path
from typing import Any

from app.config import ROOT_DIR
from app.utils.chunking import chunk_text


def load_documents(path: Path | None = None) -> list[dict[str, str]]:
    docs_path = path or ROOT_DIR / "docs.json"
    with docs_path.open("r", encoding="utf-8") as file:
        raw_docs: list[dict[str, Any]] = json.load(file)

    documents: list[dict[str, str]] = []
    for index, doc in enumerate(raw_docs):
        title = str(doc.get("title", f"Document {index + 1}")).strip()
        content = str(doc.get("content", "")).strip()
        if content:
            documents.append({"title": title, "content": content})
    return documents


def build_chunks(documents: list[dict[str, str]]) -> list[dict[str, str]]:
    chunks: list[dict[str, str]] = []
    for doc_index, doc in enumerate(documents):
        for chunk_index, chunk in enumerate(chunk_text(doc["content"])):
            chunks.append(
                {
                    "id": f"doc-{doc_index + 1}-chunk-{chunk_index + 1}",
                    "title": doc["title"],
                    "source": f"docs.json#{doc_index + 1}",
                    "text": chunk,
                }
            )
    return chunks
