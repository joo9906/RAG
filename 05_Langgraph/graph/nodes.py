"""LangGraph node functions for the advanced RAG pipeline.

Pipeline order:
  cache_check → router → query_expander → multi_retrieval → reranker
  → answer_generator → hallucination_checker
      ↘ (grounded / max retries) → cache_store → END
      ↘ (hallucinated)           → self_correction → query_expander (loop)
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage
from langchain_openai import OpenAIEmbeddings

from cache.redis_cache import SemanticCache
from graph.state import AdvancedRAGState
from vector_store.elasticsearch_store import ElasticsearchStore
from vector_store.weaviate_store import WeaviateStore

_WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
_ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
_MAX_CORRECTIONS = 2

_llm = ChatOpenAI(model="gpt-4o", temperature=0)
_embedder = OpenAIEmbeddings(model="text-embedding-3-small")
_cache = SemanticCache(redis_url=_REDIS_URL, threshold=0.8)


# ─── Reciprocal Rank Fusion ──────────────────────────────────────────────────

def _rrf(ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60) -> list[dict]:
    """Merge multiple ranked document lists via RRF. Higher score = better rank."""
    scores: dict[tuple, float] = {}
    doc_map: dict[tuple, dict] = {}
    for lst in ranked_lists:
        for rank, doc in enumerate(lst):
            key = (doc["source"], doc["chunk_id"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + k)
            doc_map.setdefault(key, doc)
    top = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    result = []
    for key in top:
        doc = dict(doc_map[key])
        doc["rrf_score"] = round(scores[key], 6)
        result.append(doc)
    return result


# ─── Sync DB helpers (run via asyncio.to_thread) ─────────────────────────────

def _weaviate_search(emb: list[float], k: int = 5) -> list[dict]:
    try:
        with WeaviateStore(_WEAVIATE_URL) as ws:
            return ws.search(emb, k=k)
    except Exception:
        return []


def _es_search(query: str, k: int = 5) -> list[dict]:
    try:
        with ElasticsearchStore(_ES_URL) as es:
            return es.search_bm25(query, k=k)
    except Exception:
        return []


# ─── Node 1: Cache Check ─────────────────────────────────────────────────────

async def cache_check_node(state: AdvancedRAGState) -> dict[str, Any]:
    last_human = next(
        (m for m in reversed(state["messages"]) if m.type == "human"), None
    )
    query = last_human.content if last_human else ""

    answer, score = await _cache.get(query)

    if answer:
        return {
            "original_query": query,
            "cache_hit": True,
            "cache_similarity": round(score, 4),
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }
    return {
        "original_query": query,
        "cache_hit": False,
        "cache_similarity": round(score, 4),
        "correction_count": 0,
        "corrected_query": "",
    }


# ─── Node 2: Dynamic Router ──────────────────────────────────────────────────

async def router_node(state: AdvancedRAGState) -> dict[str, Any]:
    system = (
        "당신은 금융 챗봇 라우터입니다. 질문 유형을 정확히 분류하세요.\n\n"
        "- 'rag': 금융 지식·개념·투자 전략·경제 지표·금리·채권·ETF 등 일반 금융 지식\n"
        "- 'stock': 특정 종목 현재 주가·실시간 시세·주가 이력·종목 비교 등 실시간 데이터\n\n"
        "'rag' 또는 'stock' 중 하나만 출력하세요. 다른 텍스트 없이."
    )
    response = await _llm.ainvoke([
        {"role": "system", "content": system},
        {"role": "user", "content": state["original_query"]},
    ])
    route = response.content.strip().lower()
    return {"route_decision": route if route in ("rag", "stock") else "rag"}


# ─── Node 3: Multi-Query Expander ────────────────────────────────────────────

async def query_expander_node(state: AdvancedRAGState) -> dict[str, Any]:
    base = state.get("corrected_query") or state["original_query"]

    system = (
        "당신은 검색 쿼리 다양화 전문가입니다.\n"
        "주어진 질문을 더 풍부한 검색 결과를 위해 3개의 다양한 표현으로 변형하세요.\n"
        "각 변형은 원래 질문의 다른 측면(동의어, 상위/하위 개념, 관련 용어)을 담아야 합니다.\n"
        '반드시 JSON 배열만 출력: ["변형1", "변형2", "변형3"]'
    )
    response = await _llm.ainvoke([
        {"role": "system", "content": system},
        {"role": "user", "content": f"원래 질문: {base}"},
    ])

    try:
        raw = response.content.strip()
        # LLM이 ```json ... ``` 블록으로 감쌀 경우 처리
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        variants = json.loads(raw)
        if isinstance(variants, list):
            expanded = [base] + [q for q in variants if isinstance(q, str)][:3]
        else:
            expanded = [base]
    except (json.JSONDecodeError, ValueError, IndexError):
        expanded = [base]

    return {"expanded_queries": expanded}


# ─── Node 4: Multi-Query Retrieval ───────────────────────────────────────────

async def multi_retrieval_node(state: AdvancedRAGState) -> dict[str, Any]:
    queries = state.get("expanded_queries") or [state["original_query"]]

    # Embed all queries concurrently
    embeddings: list[list[float]] = await asyncio.gather(
        *[_embedder.aembed_query(q) for q in queries]
    )

    # For each query, retrieve from Weaviate and ES concurrently
    async def _retrieve_pair(q: str, emb: list[float]) -> tuple[list[dict], list[dict]]:
        w, e = await asyncio.gather(
            asyncio.to_thread(_weaviate_search, emb),
            asyncio.to_thread(_es_search, q),
        )
        return w, e

    pair_results = await asyncio.gather(
        *[_retrieve_pair(q, emb) for q, emb in zip(queries, embeddings)]
    )

    ranked_lists: list[list[dict]] = []
    for w_results, e_results in pair_results:
        if w_results:
            ranked_lists.append(w_results)
        if e_results:
            ranked_lists.append(e_results)

    return {"retrieved_docs": ranked_lists}


# ─── Node 5: RRF Reranker ────────────────────────────────────────────────────

async def reranker_node(state: AdvancedRAGState) -> dict[str, Any]:
    ranked_lists: list[list[dict]] = state.get("retrieved_docs") or []
    if not ranked_lists:
        return {"reranked_docs": []}
    reranked = _rrf(ranked_lists, top_k=5)
    return {"reranked_docs": reranked}


# ─── Node 6: Answer Generator ────────────────────────────────────────────────

async def answer_generator_node(state: AdvancedRAGState) -> dict[str, Any]:
    docs = state.get("reranked_docs") or []
    query = state.get("corrected_query") or state["original_query"]

    if docs:
        context = "\n\n---\n\n".join(
            f"[출처: {d['source']} / {d['section']} | RRF 점수: {d.get('rrf_score', 0):.4f}]\n{d['content']}"
            for d in docs
        )
    else:
        context = "(참고 문서 없음)"

    system = (
        "당신은 금융 전문 AI 어시스턴트입니다.\n"
        "아래 참고 문서만을 근거로 사용자의 질문에 답변하세요.\n"
        "문서에 없는 내용은 절대 추측하거나 지어내지 마세요.\n"
        "불확실한 내용은 '문서에 명시되지 않았습니다'라고 명시하세요.\n"
        "답변 말미에 사용한 출처(파일명, 섹션)를 반드시 명시하세요."
    )
    user = f"[참고 문서]\n{context}\n\n[질문]\n{query}"

    response = await _llm.ainvoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    return {"draft_answer": response.content}


# ─── Node 7: Hallucination Checker ───────────────────────────────────────────

async def hallucination_checker_node(state: AdvancedRAGState) -> dict[str, Any]:
    draft = state.get("draft_answer", "")
    docs = state.get("reranked_docs") or []

    if not docs:
        return {
            "hallucination_verdict": "grounded",
            "hallucination_reason": "참고 문서 없음 — 검증 생략",
        }

    context = "\n\n---\n\n".join(
        f"[{d['source']} / {d['section']}]\n{d['content']}" for d in docs
    )

    system = (
        "당신은 RAG 시스템의 할루시네이션 검증 전문가입니다.\n"
        "제공된 소스 문서를 기반으로 답변의 사실 근거를 검증하세요.\n\n"
        "판단 기준:\n"
        "- 'grounded': 답변의 모든 핵심 사실이 소스 문서에서 직접 찾을 수 있음\n"
        "- 'hallucinated': 답변에 소스 문서에 없거나 모순되는 내용이 포함됨\n\n"
        '반드시 다음 JSON 형식으로만 응답하세요:\n{"verdict": "grounded" 또는 "hallucinated", "reason": "구체적 이유"}'
    )
    user = f"[소스 문서]\n{context}\n\n[검증할 답변]\n{draft}"

    response = await _llm.ainvoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])

    try:
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        parsed = json.loads(raw)
        verdict = parsed.get("verdict", "grounded")
        reason = parsed.get("reason", "")
        if verdict not in ("grounded", "hallucinated"):
            verdict = "grounded"
    except (json.JSONDecodeError, ValueError, IndexError):
        verdict = "grounded"
        reason = "JSON 파싱 실패 — grounded 처리"

    return {"hallucination_verdict": verdict, "hallucination_reason": reason}


# ─── Node 8: Self-Correction ─────────────────────────────────────────────────

async def self_correction_node(state: AdvancedRAGState) -> dict[str, Any]:
    system = (
        "당신은 RAG 검색 쿼리 개선 전문가입니다.\n"
        "할루시네이션이 감지된 원인을 분석하고, 더 정확한 문서를 찾을 수 있는\n"
        "개선된 검색 쿼리를 생성하세요.\n"
        "쿼리 텍스트만 출력하세요. 다른 설명 없이."
    )
    user = (
        f"원래 질문: {state['original_query']}\n"
        f"할루시네이션 분석: {state.get('hallucination_reason', '')}\n\n"
        "개선된 검색 쿼리:"
    )
    response = await _llm.ainvoke([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ])
    return {
        "corrected_query": response.content.strip(),
        "correction_count": state.get("correction_count", 0) + 1,
    }


# ─── Node 9: Cache Store (terminal for RAG path) ─────────────────────────────

async def cache_store_node(state: AdvancedRAGState) -> dict[str, Any]:
    answer = state.get("draft_answer", "")
    query = state["original_query"]
    verdict = state.get("hallucination_verdict", "grounded")
    corrections = state.get("correction_count", 0)
    n_queries = len(state.get("expanded_queries", []))
    n_docs = len(state.get("reranked_docs", []))

    verdict_emoji = "✅" if verdict == "grounded" else "⚠️"
    correction_note = f" | 자기수정 {corrections}회" if corrections > 0 else ""

    pipeline_meta = (
        f"\n\n---\n"
        f"*🔍 멀티쿼리 {n_queries}개 | 검색 문서 {n_docs}개 | RRF 리랭킹*  \n"
        f"*{verdict_emoji} 할루시네이션 검증: {verdict}{correction_note}*"
    )
    final = answer + pipeline_meta

    await _cache.set(query, final)

    return {
        "final_answer": final,
        "messages": [AIMessage(content=final)],
    }
