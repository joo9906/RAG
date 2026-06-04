"""LangGraph state for the advanced RAG pipeline."""

from __future__ import annotations

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AdvancedRAGState(TypedDict):
    # ── conversation ──────────────────────────────────────────────────────
    messages: Annotated[list, add_messages]

    # ── cache layer ───────────────────────────────────────────────────────
    original_query: str
    cache_hit: bool
    cache_similarity: float

    # ── routing ───────────────────────────────────────────────────────────
    route_decision: str           # "rag" | "stock"

    # ── multi-query expansion ─────────────────────────────────────────────
    expanded_queries: list[str]   # original + LLM-generated variants

    # ── retrieval & reranking ─────────────────────────────────────────────
    retrieved_docs: list          # list[list[dict]] — per-(query, store) ranked lists
    reranked_docs: list[dict]     # RRF top-k flattened docs

    # ── generation ────────────────────────────────────────────────────────
    draft_answer: str

    # ── hallucination verification ────────────────────────────────────────
    hallucination_verdict: str    # "grounded" | "hallucinated"
    hallucination_reason: str

    # ── self-correction loop ──────────────────────────────────────────────
    correction_count: int
    corrected_query: str          # refined query from self-correction pass

    # ── final output ──────────────────────────────────────────────────────
    final_answer: str
