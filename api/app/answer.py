from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from .config import settings
from .retrieval import hybrid_search


class Draft(BaseModel):
    can_answer: bool = Field(description="True only if the sources contain the answer")
    answer: str = Field(description="Concise answer based only on the sources")
    citations: list[int] = Field(description="IDs of the source chunks used")
    confidence: float = Field(description="0.0 to 1.0, how well the sources support the answer")


SYSTEM = """You answer security questionnaire questions for a company.
Rules:
- Use ONLY the numbered sources provided. Never use outside knowledge.
- If the sources do not clearly answer the question, set can_answer to false.
- Do not invent numbers, dates, tools, or certifications that are not in the sources.
- List the IDs of every source you used in citations."""

_llm = ChatOllama(
    model=settings.llm_model, base_url=settings.ollama_base_url, temperature=0
).with_structured_output(Draft)


def answer_question(question: str) -> dict:
    chunks = hybrid_search(question)
    sources_by_id = {c["id"]: c for c in chunks}

    draft = None
    if chunks:
        sources = "\n\n".join(
            f"[{c['id']}] ({c['document']} / {c['section']})\n{c['content']}" for c in chunks
        )
        draft = _llm.invoke(
            [("system", SYSTEM), ("human", f"Sources:\n{sources}\n\nQuestion: {question}")]
        )

    citations = (
        [sources_by_id[i] for i in draft.citations if i in sources_by_id] if draft else []
    )
    ok = bool(draft and draft.can_answer and citations)
    return {
        "question": question,
        "status": "drafted" if ok else "needs_expert",
        "answer": draft.answer if ok else None,
        "confidence": draft.confidence if ok else 0.0,
        "citations": [
            {"id": c["id"], "document": c["document"], "section": c["section"],
             "excerpt": c["content"][:300]}
            for c in citations
        ],
        "retrieved": [{"id": c["id"], "document": c["document"], "score": c["score"]} for c in chunks],
    }