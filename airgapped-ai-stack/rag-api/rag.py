"""Minimal local RAG: chunk -> embed with Ollama -> store in Chroma -> retrieve -> answer."""

import hashlib
import io
import os
from pathlib import Path

import chromadb
import httpx
from chromadb.config import Settings
from pypdf import PdfReader

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-oss:20b")
COLLECTION = os.getenv("COLLECTION", "documents")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))
EMBED_BATCH = 32

TEXT_SUFFIXES = {".txt", ".md", ".rst", ".csv", ".tsv", ".log", ".json", ".yaml", ".yml",
                 ".xml", ".html", ".py", ".sh", ".c", ".cpp", ".h", ".java", ".js", ".ts", ".go", ".rs", ".sql"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf"}

SYSTEM_PROMPT = (
    "You answer questions using only the provided context from internal documents. "
    "Cite sources as [n] using the numbers given. If the context does not contain the "
    "answer, say you don't know."
)

_ollama = httpx.Client(base_url=OLLAMA_URL, timeout=httpx.Timeout(600.0, connect=10.0))
_chroma = None


def chroma():
    global _chroma
    if _chroma is None:
        _chroma = chromadb.HttpClient(
            host=CHROMA_HOST, port=CHROMA_PORT, settings=Settings(anonymized_telemetry=False)
        )
    return _chroma


def collection():
    return chroma().get_or_create_collection(COLLECTION, configuration={"hnsw": {"space": "cosine"}})


def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix in TEXT_SUFFIXES:
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"unsupported file type: {suffix or filename}")


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split into ~size-character chunks with overlap, breaking on whitespace where possible."""
    text = " ".join(text.split())
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start + size // 2, end)
            if space != -1:
                end = space
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        r = _ollama.post("/api/embed", json={"model": EMBED_MODEL, "input": texts[i:i + EMBED_BATCH]})
        r.raise_for_status()
        vectors.extend(r.json()["embeddings"])
    return vectors


def ingest(source: str, data: bytes) -> int:
    """Index one document, replacing any earlier version with the same source name."""
    chunks = chunk_text(extract_text(source, data))
    col = collection()
    col.delete(where={"source": source})
    if not chunks:
        return 0
    col.add(
        ids=[hashlib.sha256(f"{source}:{i}".encode()).hexdigest() for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks),
        metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
    )
    return len(chunks)


def delete(source: str) -> None:
    collection().delete(where={"source": source})


def sources() -> dict[str, int]:
    counts: dict[str, int] = {}
    for meta in collection().get(include=["metadatas"])["metadatas"]:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return counts


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    res = collection().query(
        query_embeddings=embed([question]), n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"source": m["source"], "chunk": m["chunk"], "score": round(1 - d, 4), "text": doc}
        for doc, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def answer(question: str, top_k: int = TOP_K, model: str | None = None) -> dict:
    hits = retrieve(question, top_k)
    context = "\n\n".join(f"[{i}] ({h['source']})\n{h['text']}" for i, h in enumerate(hits, 1))
    r = _ollama.post("/api/chat", json={
        "model": model or CHAT_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    })
    r.raise_for_status()
    return {"answer": r.json()["message"]["content"], "model": model or CHAT_MODEL, "sources": hits}


def health() -> dict:
    status = {}
    try:
        tags = _ollama.get("/api/tags", timeout=5).json()
        status["ollama"] = {"ok": True, "models": sorted(m["name"] for m in tags.get("models", []))}
    except Exception as e:  # noqa: BLE001 - report any failure
        status["ollama"] = {"ok": False, "error": str(e)}
    try:
        chroma().heartbeat()
        status["chroma"] = {"ok": True}
    except Exception as e:  # noqa: BLE001
        status["chroma"] = {"ok": False, "error": str(e)}
    return status
