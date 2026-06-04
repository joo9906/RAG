"""LangGraph StateGraph builder for the advanced RAG pipeline.

Graph topology:

  START
    └─► cache_check ──(hit)──────────────────────────────────────► END
              │
          (miss)
              ▼
           router ──(stock)──────────────────────────────────────► stock ──► END
              │
            (rag)
              ▼
        query_expander ◄─────────────────────────────────────────┐
              │                                                   │
              ▼                                                   │
       multi_retrieval                                   self_correction
              │                                                   ▲
              ▼                                                   │ (hallucinated
           reranker                                               │  & retries left)
              │                                                   │
              ▼                                                   │
      answer_generator                                            │
              │                                                   │
              ▼                                                   │
  hallucination_checker ──────────────────────────────────────────┘
              │
     (grounded OR max retries)
              ▼
         cache_store ──────────────────────────────────────────► END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.stock_agent import stock_node
from graph.nodes import (
    answer_generator_node,
    cache_check_node,
    cache_store_node,
    hallucination_checker_node,
    multi_retrieval_node,
    query_expander_node,
    reranker_node,
    router_node,
    self_correction_node,
)
from graph.state import AdvancedRAGState

_MAX_CORRECTIONS = 2


def _after_cache(state: AdvancedRAGState) -> str:
    return END if state.get("cache_hit") else "router"


def _after_router(state: AdvancedRAGState) -> str:
    return "stock" if state.get("route_decision") == "stock" else "query_expander"


def _after_hallucination(state: AdvancedRAGState) -> str:
    verdict = state.get("hallucination_verdict", "grounded")
    count = state.get("correction_count", 0)
    if verdict == "grounded" or count >= _MAX_CORRECTIONS:
        return "cache_store"
    return "self_correction"


def build_graph():
    graph = StateGraph(AdvancedRAGState)

    graph.add_node("cache_check", cache_check_node)
    graph.add_node("router", router_node)
    graph.add_node("query_expander", query_expander_node)
    graph.add_node("multi_retrieval", multi_retrieval_node)
    graph.add_node("reranker", reranker_node)
    graph.add_node("answer_generator", answer_generator_node)
    graph.add_node("hallucination_checker", hallucination_checker_node)
    graph.add_node("self_correction", self_correction_node)
    graph.add_node("cache_store", cache_store_node)
    graph.add_node("stock", stock_node)

    graph.add_edge(START, "cache_check")

    graph.add_conditional_edges(
        "cache_check",
        _after_cache,
        {"router": "router", END: END},
    )
    graph.add_conditional_edges(
        "router",
        _after_router,
        {"query_expander": "query_expander", "stock": "stock"},
    )

    graph.add_edge("query_expander", "multi_retrieval")
    graph.add_edge("multi_retrieval", "reranker")
    graph.add_edge("reranker", "answer_generator")
    graph.add_edge("answer_generator", "hallucination_checker")

    graph.add_conditional_edges(
        "hallucination_checker",
        _after_hallucination,
        {"cache_store": "cache_store", "self_correction": "self_correction"},
    )

    # Self-correction loops back to query_expander with refined query
    graph.add_edge("self_correction", "query_expander")

    graph.add_edge("cache_store", END)
    graph.add_edge("stock", END)

    return graph.compile()
