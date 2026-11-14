from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select

from .answer import answer_question
from .db import Chunk, Document, SessionLocal, init_db
from .ingestion import ingest_document


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Questionnaire AI", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        try:
            results.append(ingest_document(f.filename, f.file.read()))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return results


@app.post("/ask")
def ask(req: AskRequest):
    return answer_question(req.question)


@app.get("/documents")
def list_documents():
    with SessionLocal() as session:
        rows = session.execute(
            select(Document.name, func.count(Chunk.id))
            .outerjoin(Chunk, Chunk.document_id == Document.id)
            .group_by(Document.name)
            .order_by(Document.name)
        ).all()
    return [{"document": name, "chunks": count} for name, count in rows]


@app.delete("/documents/{name}")
def delete_document(name: str):
    with SessionLocal() as session:
        result = session.execute(delete(Document).where(Document.name == name))
        session.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail=f"No document named {name}")
    return {"deleted": name}