"""Multi-agent financial chatbot entry point.

Usage:
    python main.py

Prerequisites:
    1. docker compose up -d           (Weaviate + ES)
    2. python ingest.py               (index documents)
    3. cp .env.example .env && edit   (set API keys)
    4. python main.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

_REQUIRED_KEYS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
_MISSING = [k for k in _REQUIRED_KEYS if not os.getenv(k)]
if _MISSING:
    print(f"[ERROR] Missing environment variables: {', '.join(_MISSING)}")
    print("  Copy .env.example to .env and fill in your API keys.")
    sys.exit(1)

# Deferred import so env vars are loaded first
from agents.supervisor import build_graph  # noqa: E402

_WELCOME = """
╔══════════════════════════════════════════════════════════╗
║       🏦 금융 멀티 에이전트 챗봇 (LangGraph + MCP)        ║
╠══════════════════════════════════════════════════════════╣
║  • 금융 지식 질문 → RAG 에이전트 (Weaviate + ES 비교)     ║
║  • 실시간 주가 질문 → 주식 에이전트 (MCP + yfinance)       ║
║                                                          ║
║  예시 질문:                                               ║
║    - PER이 무엇인가요?                                    ║
║    - ETF 투자 전략을 알려주세요                            ║
║    - 애플 현재 주가 알려줘                                 ║
║    - 삼성전자와 SK하이닉스 주가 비교해줘                   ║
║    - TSLA 최근 1개월 이력 보여줘                           ║
║                                                          ║
║  종료: 'exit' 또는 Ctrl+C                                 ║
╚══════════════════════════════════════════════════════════╝
"""


def _langsmith_status() -> str:
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    project = os.getenv("LANGCHAIN_PROJECT", "default")
    if tracing and os.getenv("LANGCHAIN_API_KEY"):
        return f"  LangSmith 추적 활성화 — 프로젝트: {project}"
    return "  LangSmith 추적 비활성화 (.env에서 LANGCHAIN_TRACING_V2=true 설정)"


async def chat_loop() -> None:
    print(_WELCOME)
    print(_langsmith_status())
    print()

    graph = build_graph()
    conversation_history: list = []

    while True:
        try:
            user_input = input("\n🧑 질문: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n챗봇을 종료합니다. 감사합니다!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "종료", "그만"):
            print("\n챗봇을 종료합니다. 감사합니다!")
            break

        conversation_history.append(HumanMessage(content=user_input))

        print("\n🤖 처리 중…\n")
        try:
            result = await graph.ainvoke({"messages": conversation_history})
        except Exception as e:
            print(f"[ERROR] {e}")
            print("  서비스가 실행 중인지 확인하세요: docker compose up -d && python ingest.py")
            continue

        # Extract last AI message
        from langchain_core.messages import AIMessage
        ai_msg = next(
            (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
            None,
        )

        if ai_msg:
            print(f"🤖 답변:\n{ai_msg.content}")
            conversation_history.append(ai_msg)
        else:
            print("[WARNING] 응답을 받지 못했습니다.")

        route = result.get("route_decision", "")
        agent_label = {"rag": "📚 RAG 에이전트", "stock": "📈 주식 에이전트"}.get(route, "")
        if agent_label:
            print(f"\n  [라우팅: {agent_label}]")


def main() -> None:
    asyncio.run(chat_loop())


if __name__ == "__main__":
    main()
