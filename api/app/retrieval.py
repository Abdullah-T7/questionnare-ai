import re

from sqlalchemy import func, select

from .config import settings
from .db import Chunk, Document, SessionLocal
from .embeddings import embedder

RRF_K = 60


def _or_query(question: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9]+", question) if len(w) > 2]
    return " | ".join(words)


def hybrid_search(question: str, k: int | None = None) -> list[dict]:
    k = k or settings.top_k
    n = settings.candidates
    qvec = embedder.embed_query(question)
    tsq = _or_query(question)

    with SessionLocal() as session:
        vec_ids = session.scalars(
            select(Chunk.id).order_by(Chunk.embedding.cosine_distance(qvec)).limit(n)
        ).all()

        kw_ids = []
        if tsq:
            query = func.to_tsquery("english", tsq)
            kw_ids = session.scalars(
                select(Chunk.id)
                .where(Chunk.tsv.op("@@", is_comparison=True)(query))
                .order_by(func.ts_rank_cd(Chunk.tsv, query).desc())
                .limit(n)
            ).all()

        scores: dict[int, float] = {}
        for ranked in (vec_ids, kw_ids):
            for rank, cid in enumerate(ranked):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        top = sorted(scores, key=scores.get, reverse=True)[:k]
        if not top:
            return []

        rows = session.execute(
            select(Chunk.id, Document.name, Chunk.section, Chunk.content)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.id.in_(top))
        ).all()

    by_id = {r.id: r for r in rows}
    return [
        {
            "id": cid,
            "document": by_id[cid].name,
            "section": by_id[cid].section,
            "content": by_id[cid].content,
            "score": round(scores[cid], 4),
        }
        for cid in top
    ]