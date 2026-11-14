from contextlib import asynccontextmanager

from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select

from .answer import answer_question
from .db import Chunk, Document, SessionLocal, init_db
from .excel import process_workbook
from .ingestion import ingest_document, save_approved_answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Questionnaire AI", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3)


class ApproveRequest(BaseModel):
    question: str = Field(min_length=3)
    answer: str = Field(min_length=3)
    expert: str = Field(default="Security Expert")
    role: str = Field(default="Governance, Risk & Compliance (GRC)")


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


@app.post("/questionnaire/fill")
def fill_questionnaire(
    file: UploadFile = File(...),
    format: Literal["file", "json"] = Query(default="file", description="Return styled .xlsx file or JSON summary with escalations"),
):
    if not file.filename.endswith((".xlsx", ".xlsm")):
        raise HTTPException(status_code=400, detail="Only .xlsx or .xlsm files are supported")
    content = file.file.read()
    output_bytes, stats, escalations = process_workbook(content)

    filename_out = f"completed_{file.filename}"
    out_path = Path("/data") / filename_out
    out_path.write_bytes(output_bytes)

    if format == "json":
        return {
            "filename": filename_out,
            "download_url": f"/questionnaire/download/{filename_out}",
            "stats": stats,
            "escalations": escalations,
        }

    return Response(
        content=output_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{filename_out}"',
            "X-Total-Questions": str(stats["total_questions"]),
            "X-Drafted": str(stats["drafted"]),
            "X-Needs-Expert": str(stats["needs_expert"]),
            "X-Automation-Rate": str(stats["automation_rate"]),
        },
    )


@app.get("/questionnaire/download/{filename}")
def download_questionnaire(filename: str):
    file_path = Path("/data") / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Questionnaire file not found")
    return FileResponse(
        path=file_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
    )


@app.post("/answers/approve")
def approve_answer(req: ApproveRequest):
    result = save_approved_answer(
        question=req.question,
        answer=req.answer,
        expert=req.expert,
        role=req.role,
    )
    return {"status": "saved", "details": result}


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