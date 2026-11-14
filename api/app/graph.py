import re
from typing import Literal, TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import settings
from .retrieval import hybrid_search


# --- Pydantic Schemas for Structured LLM Outputs ---

class DraftOutput(BaseModel):
    can_answer: bool = Field(
        description="True only if the sources clearly contain the facts to answer the question. False if missing, unmentioned, or uncertain."
    )
    answer: str = Field(
        description="Direct and complete answer based strictly on the sources with relevant technical details (algorithms, timeframes, requirements). Never reply with just 'Yes' or 'No'."
    )
    citations: list[int] = Field(description="List of integer IDs of the source chunks used")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0 based on how well sources support the answer")


class GroundednessCheck(BaseModel):
    is_grounded: bool = Field(
        description="True only if EVERY statement and claim in the answer is directly and explicitly substantiated by the cited sources. False if the answer assumes, infers, or claims something not explicitly written (e.g. denying a practice or certification when the text is simply silent)."
    )
    reason: str = Field(description="Concise explanation of whether and why the answer is substantiated by the sources.")
    calibrated_confidence: float = Field(description="Calibrated confidence score between 0.0 and 1.0 reflecting factual evidence.")


# --- LLM Setups ---

_drafter = ChatOllama(
    model=settings.llm_model, base_url=settings.ollama_base_url, temperature=0
).with_structured_output(DraftOutput)

_critique = ChatOllama(
    model=settings.llm_model, base_url=settings.ollama_base_url, temperature=0
).with_structured_output(GroundednessCheck)

DRAFT_SYSTEM = """You answer enterprise security questionnaires on behalf of a company.
Rules:
- Rely strictly on the numbered source excerpts provided. Never use outside knowledge or assumptions.
- If the sources do not directly address the question, set can_answer to false.
- Do not invent numbers, dates, tools, or certifications that are not in the sources.
- List the IDs of every source chunk you used in citations."""

CRITIQUE_SYSTEM = """You are a strict security compliance auditor.
Your job is to audit an AI-drafted answer against the cited source excerpts to prevent hallucinations.
Rules:
1. Every factual statement, number, tool, or timeline in the answer must appear explicitly in the sources.
2. In security compliance, silence is not confirmation. If a policy does not mention a practice (e.g. penetration testing, background checks, specific certifications), asserting 'No, we do not do X' is an UNSUPPORTED CLAIM. In such cases, is_grounded MUST be false.
3. If the answer is ungrounded or speculates beyond the source text, set is_grounded to false."""


# --- Graph State ---

class WorkflowState(TypedDict, total=False):
    question: str
    chunks: list[dict]
    retrieved: list[dict]
    can_answer: bool
    draft_answer: str | None
    citations: list[dict]
    confidence: float
    is_grounded: bool
    critique_reason: str
    status: Literal["drafted", "needs_expert"]
    answer: str | None
    expert_role: str | None
    expert_reason: str | None


# --- Helper: Expert Role Assignment ---

def determine_expert_role(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["access", "password", "mfa", "login", "sso", "privilege", "key"]):
        return "Identity & Access Management"
    if any(k in q for k in ["backup", "recovery", "rto", "rpo", "disaster", "retention", "laptop", "disk"]):
        return "IT Infrastructure & Ops"
    if any(k in q for k in ["encrypt", "tls", "aes", "kms", "cipher"]):
        return "Security Engineering"
    if any(k in q for k in ["incident", "breach", "pagerduty", "notify", "review"]):
        return "SecOps & Incident Response"
    if any(k in q for k in ["soc 2", "iso", "audit", "compliance", "certif", "insurance", "penetration", "background check"]):
        return "Governance, Risk & Compliance (GRC)"
    return "Information Security Team"


# --- Graph Nodes ---

def retrieve_node(state: WorkflowState) -> WorkflowState:
    chunks = hybrid_search(state["question"])
    return {
        "chunks": chunks,
        "retrieved": [{"id": c["id"], "document": c["document"], "score": c["score"]} for c in chunks],
    }


def draft_node(state: WorkflowState) -> WorkflowState:
    chunks = state.get("chunks", [])
    question = state["question"]

    if not chunks:
        return {
            "can_answer": False,
            "draft_answer": None,
            "citations": [],
            "confidence": 0.0,
        }

    sources_by_id = {c["id"]: c for c in chunks}
    sources_text = "\n\n".join(
        f"[{c['id']}] ({c['document']} / {c['section']})\n{c['content']}" for c in chunks
    )

    human_prompt = (
        f"Sources:\n{sources_text}\n\n"
        f"Question: {question}\n\n"
        "Provide a complete, explanatory answer with specific facts and details from the sources. "
        "Never reply with just 'Yes' or 'No'."
    )

    try:
        draft = _drafter.invoke([("system", DRAFT_SYSTEM), ("human", human_prompt)])
    except Exception:
        return {
            "can_answer": False,
            "draft_answer": None,
            "citations": [],
            "confidence": 0.0,
        }

    if not draft or not draft.can_answer:
        return {
            "can_answer": False,
            "draft_answer": None,
            "citations": [],
            "confidence": 0.0,
        }

    valid_citations = [sources_by_id[cid] for cid in draft.citations if cid in sources_by_id]
    if not valid_citations:
        return {
            "can_answer": False,
            "draft_answer": None,
            "citations": [],
            "confidence": 0.0,
        }

    return {
        "can_answer": True,
        "draft_answer": draft.answer,
        "citations": valid_citations,
        "confidence": draft.confidence,
    }


def self_check_node(state: WorkflowState) -> WorkflowState:
    if not state.get("can_answer") or not state.get("draft_answer"):
        return {
            "is_grounded": False,
            "critique_reason": "No answer drafted from sources.",
        }

    citations = state.get("citations", [])
    cited_text = "\n\n".join(
        f"[{c['id']}] ({c['document']} / {c['section']})\n{c['content']}" for c in citations
    )

    human_prompt = (
        f"Cited Sources:\n{cited_text}\n\n"
        f"Question: {state['question']}\n\n"
        f"Draft Answer: {state['draft_answer']}\n\n"
        "Verify whether every claim in the Draft Answer is explicitly substantiated by the Cited Sources."
    )

    try:
        critique = _critique.invoke([("system", CRITIQUE_SYSTEM), ("human", human_prompt)])
        return {
            "is_grounded": critique.is_grounded,
            "critique_reason": critique.reason,
            "confidence": min(state.get("confidence", 1.0), critique.calibrated_confidence),
        }
    except Exception as e:
        return {
            "is_grounded": True,
            "critique_reason": f"Self-check fallback: {e}",
        }


def route_node(state: WorkflowState) -> WorkflowState:
    question = state["question"]
    can_answer = state.get("can_answer", False)
    is_grounded = state.get("is_grounded", False)
    citations = state.get("citations", [])

    if can_answer and is_grounded and citations:
        return {
            "status": "drafted",
            "answer": state.get("draft_answer"),
            "expert_role": None,
            "expert_reason": None,
        }

    # Route to expert
    role = determine_expert_role(question)
    critique_reason = state.get("critique_reason", "")
    reason = critique_reason if critique_reason and not is_grounded else "Information not found in existing policies."

    return {
        "status": "needs_expert",
        "answer": None,
        "confidence": 0.0,
        "expert_role": role,
        "expert_reason": reason,
    }


# --- Build StateGraph ---

builder = StateGraph(WorkflowState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("draft", draft_node)
builder.add_node("self_check", self_check_node)
builder.add_node("route", route_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "draft")
builder.add_edge("draft", "self_check")
builder.add_edge("self_check", "route")
builder.add_edge("route", END)

workflow = builder.compile()


def run_questionnaire_workflow(question: str) -> dict:
    result = workflow.invoke({"question": question})
    citations = result.get("citations", [])
    return {
        "question": question,
        "status": result.get("status", "needs_expert"),
        "answer": result.get("answer"),
        "confidence": result.get("confidence", 0.0),
        "citations": [
            {
                "id": c["id"],
                "document": c["document"],
                "section": c["section"],
                "excerpt": c["content"][:300],
            }
            for c in citations
        ],
        "retrieved": result.get("retrieved", []),
        "expert_role": result.get("expert_role"),
        "expert_reason": result.get("expert_reason"),
        "is_grounded": result.get("is_grounded", False),
    }

