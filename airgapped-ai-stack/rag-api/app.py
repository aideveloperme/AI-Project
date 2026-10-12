"""HTTP API for the local RAG pipeline. Interactive docs at /docs (served offline)."""

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

import rag

# Swagger UI normally loads its JS/CSS from a CDN, which does not exist here,
# so only the raw OpenAPI schema is served; use curl or the CLI examples in the README.
app = FastAPI(title="Air-gapped RAG API", docs_url=None, redoc_url=None)


class Query(BaseModel):
    question: str
    top_k: int = rag.TOP_K
    model: str | None = None


@app.get("/health")
def health():
    return rag.health()


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    results = {}
    for f in files:
        try:
            results[f.filename] = {"chunks": rag.ingest(f.filename, await f.read())}
        except ValueError as e:
            results[f.filename] = {"error": str(e)}
    return results


@app.get("/documents")
def documents():
    return rag.sources()


@app.delete("/documents/{source:path}")
def delete_document(source: str):
    rag.delete(source)
    return {"deleted": source}


@app.post("/search")
def search(q: Query):
    return rag.retrieve(q.question, q.top_k)


@app.post("/query")
def query(q: Query):
    if not q.question.strip():
        raise HTTPException(400, "question is empty")
    return rag.answer(q.question, q.top_k, q.model)
