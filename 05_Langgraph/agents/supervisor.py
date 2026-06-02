"""LangGraph supervisor — routes user queries to RAG agent or Stock agent."""

from __future__ import annotations

from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langgraph.graph import END, MessagesState, StateGraph

from agents.rag_agent import rag_node
from agents.stock_agent import stock_node

_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

_ROUTER_SYSTEM = """당신은 금융 챗봇의 라우터 역할을 합니다.
사용자의 질문을 분석하여 다음 두 에이전트 중 어디로 보낼지 결정하세요.

에이전트 종류:
- "rag": 금융 지식, 개념, 투자 전략, 금리, 채권, 주식 기초 등 **일반 금융 지식** 질문
- "stock": 특정 종목의 현재 주가, 실시간 시세, 주식 비교, 과거 이력 등 **실시간 주식 데이터** 질문

반드시 "rag" 또는 "stock" 중 하나만 응답하세요. 다른 텍스트는 출력하지 마세요."""


def supervisor_node(state: MessagesState) -> dict:
    """Decides which sub-agent to route to. Stores decision in last message metadata."""
    last_user_msg = next(
        (m for m in reversed(state["messages"]) if m.type == "human"),
        None,
    )
    query = last_user_msg.content if last_user_msg else ""

    response = _llm.invoke(
        [
            SystemMessage(content=_ROUTER_SYSTEM),
            {"role": "user", "content": f"질문: {query}"},
        ]
    )

    route = response.content.strip().lower()
    if route not in ("rag", "stock"):
        route = "rag"

    # Store route decision as metadata on the last AI message (or as a plain dict key)
    return {"route_decision": route}


def route_decision(state: MessagesState) -> Literal["rag", "stock"]:
    return state.get("route_decision", "rag")


# ── Graph definition ──────────────────────────────────────────────────────────

class SupervisorState(MessagesState):
    route_decision: str


def build_graph() -> "CompiledGraph":  # noqa: F821
    graph = StateGraph(SupervisorState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag", rag_node)
    graph.add_node("stock", stock_node)

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_decision,
        {"rag": "rag", "stock": "stock"},
    )
    graph.add_edge("rag", END)
    graph.add_edge("stock", END)

    return graph.compile()
