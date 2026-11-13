import io
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sqlalchemy import delete

from .db import Chunk, Document, SessionLocal
from .embeddings import embedder

MIN_CHARS = 50

_md = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")], strip_headers=False
)
_rc = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def load_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    if ext in {".md", ".txt"}:
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {ext} (use .pdf, .md or .txt)")


def chunk_text(text: str) -> list[tuple[str, str]]:
    """Returns (section, content) pairs."""
    out = []
    for doc in _md.split_text(text):
        section = " > ".join(doc.metadata[k] for k in ("h1", "h2", "h3") if k in doc.metadata)
        for piece in _rc.split_text(doc.page_content):
            if len(piece.strip()) >= MIN_CHARS:
                out.append((section, piece))
    return out


def ingest_document(filename: str, data: bytes) -> dict:
    chunks = chunk_text(load_text(filename, data))
    if not chunks:
        raise ValueError("No text could be extracted from the file")

    vectors = embedder.embed_documents([f"{filename} | {s}\n{c}" for s, c in chunks])

    with SessionLocal() as session:
        session.execute(delete(Document).where(Document.name == filename))
        doc = Document(name=filename)
        doc.chunks = [
            Chunk(chunk_index=i, section=section, content=content, embedding=vec)
            for i, ((section, content), vec) in enumerate(zip(chunks, vectors))
        ]
        session.add(doc)
        session.commit()
    return {"document": filename, "chunks": len(chunks)}