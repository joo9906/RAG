"""RAG Agent — queries Weaviate (semantic) + Elasticsearch (BM25) and compares results."""

from __future__ import annotations

import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import MessagesState
from langsmith import traceable

from vector_store.weaviate_store import WeaviateStore
from vector_store.elasticsearch_store import ElasticsearchStore

_WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
_ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
_embedder = OpenAIEmbeddings(model="text-embedding-3-small")


def _format_results(results: list[dict], title: str) -> str:
    if not results:
        return f"{title}\n  (결과 없음)"
    lines = [title]
    for i, r in enumerate(results, 1):
        score_label = "유사도" if "weaviate" in r["method"] else "BM25 점수"
        lines.append(f"  [{i}] ({score_label}: {r['score']}) {r['source']} — {r['section']}")
        snippet = r["content"][:160].replace("\n", " ")
        lines.append(f"      {snippet}…")
    return "\n".join(lines)


@traceable(run_type="chain", name="retrieval_comparison")
def _compare_overlap(weaviate_chunks: list[dict], es_chunks: list[dict]) -> str:
    w_ids = {(r["source"], r["chunk_id"]) for r in weaviate_chunks}
    e_ids = {(r["source"], r["chunk_id"]) for r in es_chunks}
    overlap = w_ids & e_ids
    only_w = w_ids - e_ids
    only_e = e_ids - w_ids

    lines = ["📊 검색 방식 비교 분석"]
    lines.append(f"  • 두 방법 모두 찾은 청크: {len(overlap)}개")
    if only_w:
        lines.append(f"  • Weaviate만 찾은 청크: {len(only_w)}개 (의미적으로 유사하나 키워드 불일치)")
    if only_e:
        lines.append(f"  • ES(BM25)만 찾은 청크: {len(only_e)}개 (키워드 일치하나 의미 유사도 낮음)")
    lines.append(
        f"  → {'두 방법이 일관된 결과를 보입니다.' if len(overlap) >= 2 else '검색 방식에 따라 다른 결과가 나왔습니다. 하이브리드 검색이 권장됩니다.'}"
    )
    return "\n".join(lines)


async def rag_node(state: MessagesState) -> dict[str, Any]:
    """LangGraph node: retrieves from both stores, compares, and generates an answer."""
    query = state["messages"][-1].content

    # 1. Embed the query
    query_embedding = await _embedder.aembed_query(query)

    # 2. Retrieve from both stores
    with WeaviateStore(_WEAVIATE_URL) as wstore:
        weaviate_results = wstore.search(query_embedding, k=3)

    with ElasticsearchStore(_ES_URL) as estore:
        es_results = estore.search_bm25(query, k=3)

    # 3. Build a structured comparison report
    w_block = _format_results(
        weaviate_results,
        "🔵 Weaviate 의미 검색 (Semantic / Dense Vector)",
    )
    e_block = _format_results(
        es_results,
        "🟠 Elasticsearch BM25 키워드 검색",
    )
    comparison = _compare_overlap(weaviate_results, es_results)

    # 4. Merge unique context for the LLM
    seen: set[tuple] = set()
    merged_context: list[str] = []
    for r in weaviate_results + es_results:
        key = (r["source"], r["chunk_id"])
        if key not in seen:
            seen.add(key)
            merged_context.append(
                f"[출처: {r['source']} / {r['section']}]\n{r['content']}"
            )

    context_text = "\n\n---\n\n".join(merged_context) if merged_context else "(참고 문서 없음)"

    # 5. Generate final answer
    system_prompt = (
        "당신은 금융 전문 AI 어시스턴트입니다. "
        "아래 참고 문서를 기반으로 사용자의 질문에 정확하고 친절하게 답변하세요. "
        "답변 후 반드시 출처를 명시하세요."
    )
    user_prompt = (
        f"[참고 문서]\n{context_text}\n\n"
        f"[질문]\n{query}"
    )

    llm_response = await _llm.ainvoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    answer = llm_response.content

    # 6. Compose full reply with retrieval details
    full_reply = "\n\n".join(
        [
            "## 📚 RAG 검색 결과 비교",
            w_block,
            e_block,
            comparison,
            "---",
            f"## 💡 최종 답변\n{answer}",
        ]
    )

    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=full_reply)]}
