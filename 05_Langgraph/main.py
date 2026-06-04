"""Advanced multi-agent financial chatbot entry point.

Pipeline:
  사용자 쿼리 → 캐시 확인 → 동적 라우팅 → 멀티쿼리 생성
  → 문서 검색 및 RRF 리랭킹 → Self-correction loop
  → 답변 생성 → 할루시네이션 검증 → 캐시 저장

Prerequisites:
    1. docker compose up -d           (Weaviate + ES + Redis)
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

_REQUIRED_KEYS = ["OPENAI_API_KEY"]
_MISSING = [k for k in _REQUIRED_KEYS if not os.getenv(k)]
if _MISSING:
    print(f"[ERROR] Missing environment variables: {', '.join(_MISSING)}")
    print("  Copy .env.example to .env and fill in your API keys.")
    sys.exit(1)

from graph.builder import build_graph  # noqa: E402

_WELCOME = """
╔═══════════════════════════════════════════════════════════════════╗
║    🏦 고도화 금융 멀티 에이전트 RAG 챗봇 (LangGraph + Redis)      ║
╠═══════════════════════════════════════════════════════════════════╣
║  파이프라인:                                                       ║
║    쿼리 입력 → 캐시 확인(임베딩 유사도 0.8)                        ║
║    → 동적 라우팅 → 멀티쿼리 생성(×3)                              ║
║    → 문서 검색(Weaviate + ES) + RRF 리랭킹                        ║
║    → 답변 생성 → 할루시네이션 검증 → Self-correction(최대 2회)     ║
║    → 캐시 저장                                                     ║
║                                                                   ║
║  예시 질문:                                                        ║
║    - PER이 무엇인가요?                                             ║
║    - ETF 투자 전략을 알려주세요                                     ║
║    - 애플 현재 주가 알려줘                                          ║
║    - 삼성전자와 SK하이닉스 주가 비교해줘                            ║
║                                                                   ║
║  종료: 'exit' 또는 Ctrl+C                                          ║
╚═══════════════════════════════════════════════════════════════════╝
"""


def _langsmith_status() -> str:
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    project = os.getenv("LANGCHAIN_PROJECT", "default")
    if tracing and os.getenv("LANGCHAIN_API_KEY"):
        return f"  LangSmith 추적 활성화 — 프로젝트: {project}"
    return "  LangSmith 추적 비활성화 (.env에서 LANGSMITH_TRACING=true 설정)"


def _pipeline_summary(result: dict) -> str:
    parts = []

    if result.get("cache_hit"):
        sim = result.get("cache_similarity", 0)
        parts.append(f"⚡ 캐시 히트 (유사도: {sim:.3f})")
        return "  " + " | ".join(parts)

    route = result.get("route_decision", "")
    route_label = {"rag": "📚 RAG", "stock": "📈 주식"}.get(route, route)
    if route_label:
        parts.append(f"라우팅: {route_label}")

    n_queries = len(result.get("expanded_queries", []))
    if n_queries > 1:
        parts.append(f"멀티쿼리: {n_queries}개")

    n_docs = len(result.get("reranked_docs", []))
    if n_docs:
        parts.append(f"검색 문서: {n_docs}개")

    verdict = result.get("hallucination_verdict", "")
    if verdict:
        emoji = "✅" if verdict == "grounded" else "⚠️"
        parts.append(f"검증: {emoji}{verdict}")

    corrections = result.get("correction_count", 0)
    if corrections:
        parts.append(f"자기수정: {corrections}회")

    return "  [" + " | ".join(parts) + "]" if parts else ""


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

        print("\n⏳ 처리 중…\n")
        try:
            result = await graph.ainvoke({"messages": conversation_history})
        except Exception as e:
            print(f"[ERROR] {e}")
            print("  서비스 실행 확인: docker compose up -d && python ingest.py")
            continue

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

        summary = _pipeline_summary(result)
        if summary:
            print(f"\n{summary}")


def main() -> None:
    asyncio.run(chat_loop())


if __name__ == "__main__":
    main()
