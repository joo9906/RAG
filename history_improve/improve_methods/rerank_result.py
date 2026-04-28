"""
실제 VDB 검색 + LLMReranker 결과 확인 스크립트
================================================

실행 방법 (프로젝트 루트에서):
    python improve_methods/rerank_result.py
    python improve_methods/rerank_result.py --query "기넥신 부작용" --collection ce_company_product_basic_info
    python improve_methods/rerank_result.py --query "..." --collection ... --top-k 10 --threshold 0.2

주요 컬렉션:
    ce_company_product_basic_info       제품 기본 정보
    ce_company_product_clinical_info    제품 임상 정보
    ce_company_product_nonclinical_info 비임상 정보
    ce_disease_knowledge                질환 정보
    ce_detailing_strategy               디테일링 전략
    ce_competitor_product_info          경쟁사 제품 정보
    ce_pharmaceutical_market_info       제약 시장 정보
    ce_call                             콜 기록
    ce_cp_qdrant                        컴플라이언스
    ce_paper_qdrant                     논문
    ce_PM_answer_qdrant                 PM 답변
"""
# isort:imports-stdlib
import argparse
import copy
import io
import logging
import sys
import time
from pathlib import Path

# ── 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Windows cp949 터미널 한글 깨짐 방지
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ── 외부 라이브러리 로그 억제 (httpx 요청/응답 노이즈 제거)
for _noisy in ("httpx", "httpcore", "openai", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 출력 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

W = 74  # 출력 너비

_SCORE_FIELDS = [
    ("relevance_score",      0.60, "relevance    "),
    ("specificity_score",    0.15, "specificity  "),
    ("completeness_score",   0.10, "completeness "),
    ("practicality_score",   0.10, "practicality "),
    ("constraint_fit_score", 0.05, "constraint   "),
]


def _bar(score: int, max_score: int = 10, width: int = 12) -> str:
    filled = round(score / max_score * width)
    return "#" * filled + "." * (width - filled)


def _preview(text: str, max_chars: int = 160) -> str:
    one_line = text.replace("\n", " | ")
    return one_line[:max_chars] + ("..." if len(one_line) > max_chars else "")


def print_vdb_raw(docs, query: str, collection: str, elapsed: float) -> None:
    """VDB 검색 결과 (리랭킹 전) 출력."""
    print()
    print("=" * W)
    print("  [STEP 1] VDB RAW RESULTS  (reranking 전)")
    print(f"  collection : {collection}")
    print(f"  query      : {query}")
    print(f"  elapsed    : {elapsed:.2f}s  |  {len(docs)}개 문서 반환")
    print("=" * W)

    for i, doc in enumerate(docs):
        meta      = doc.metadata
        vdb_score = meta.get("score", 0.0)
        doc_name  = meta.get("document_name", "unknown")
        page      = meta.get("page_number", "")
        page_str  = f"  p.{page}" if page else ""

        print(f"\n  [{i+1}]  vdb_score={vdb_score:.4f}  {doc_name}{page_str}")
        print(f"  {'-' * (W - 4)}")
        print(f"  {_preview(doc.page_content)}")

    print()


def print_rerank_detail(
    user_query: str,
    search_query: str,
    all_docs: list,
    result: list,
    elapsed: float,
    threshold: float,
    model: str = "gpt-4.1-mini",
) -> None:
    """
    리랭킹 결과 상세 출력.

    all_docs : rerank()에 넘긴 문서 리스트.
               _score_document()가 in-place로 sub-score를 metadata에 기록하므로
               필터링된 문서의 점수도 읽을 수 있다.
    result   : rerank()가 반환한 통과 문서 리스트 (rank, overall_score 포함).
    """
    result_by_content = {doc.page_content: doc for doc in result}

    print()
    print("=" * W)
    print("  [STEP 2] RERANKER DETAIL")
    print(f"  user_query : {user_query}")
    if search_query != user_query:
        print(f"  search     : {search_query}")
    print(f"  model      : {model}  |  threshold={threshold}  |  elapsed={elapsed:.2f}s")
    print(f"  result     : {len(all_docs)}개 -> {len(result)}개 PASS / "
          f"{len(all_docs) - len(result)}개 FILTERED")
    print("=" * W)

    for i, doc in enumerate(all_docs):
        meta      = doc.metadata
        r_doc     = result_by_content.get(doc.page_content)
        doc_name  = meta.get("document_name", "unknown")
        vdb_score = meta.get("score", 0.0)

        if r_doc is not None:
            rank    = r_doc.metadata["rank"]
            overall = r_doc.metadata["overall_score"]
            status  = (f"[PASS]  rank={rank}  "
                       f"llm={overall:.3f}  vdb={vdb_score:.4f}")
        elif "relevance_score" in meta:
            # 필터링됐지만 _score_document가 점수를 기록한 경우
            total_w = sum(w for _, w, _ in _SCORE_FIELDS)
            overall = (sum((meta.get(k, 0) / 3.0) * w
                           for k, w, _ in _SCORE_FIELDS) / total_w)
            status  = (f"[FILTERED]  "
                       f"llm={overall:.3f} (< {threshold})  vdb={vdb_score:.4f}")
        else:
            status = f"[FILTERED]  (timeout/error)  vdb={vdb_score:.4f}"

        print(f"\n  doc {i+1}  {status}")
        print(f"  [{doc_name}]")
        print(f"  {'-' * (W - 4)}")
        print(f"  {_preview(doc.page_content)}")
        print()

        # LLM 세부 점수
        score_src = (r_doc.metadata if r_doc is not None else meta)
        if "relevance_score" in score_src:
            for field, weight, lbl in _SCORE_FIELDS:
                score = score_src.get(field, 0)
                bar   = _bar(score)
                pct   = f"{weight * 100:.0f}%"
                print(f"    {lbl}  {bar}  {score}/10  (weight {pct:>4})")
        else:
            print("    (LLM 점수 없음)")

    print()
    print("-" * W)
    print(f"  reranker 소요: {elapsed:.2f}s  ({len(all_docs)}개 문서 병렬 평가)")
    print("=" * W)
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    parser = argparse.ArgumentParser(
        description="실제 VDB 검색 후 LLMReranker 평가 결과 확인"
    )
    parser.add_argument(
        "--query", "-q",
        default="내과 밑 약국에는 어떤 약품을 파는게 좋을까?",
        help="사용자 쿼리 (기본: '내과 밑 약국에는 어떤 약품을 파는게 좋을까?')",
    )
    parser.add_argument(
        "--search-query", "-s",
        default=None,
        help="VDB 검색에 사용할 쿼리 (기본: --query와 동일)",
    )
    parser.add_argument(
        "--collection", "-c",
        default="ce_company_product_basic_info",
        help="Qdrant 컬렉션명 (기본: ce_company_product_basic_info)",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=5,
        help="VDB 검색 결과 수 (기본: 5)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.3,
        help="리랭커 임계값 (기본: 0.3, 이 점수 미만은 제거)",
    )
    args = parser.parse_args()

    user_query   = args.query
    search_query = args.search_query or user_query
    collection   = args.collection
    top_k        = args.top_k
    threshold    = args.threshold

    print()
    print("=" * W)
    print("  VDB Search + LLMReranker")
    print(f"  user_query  : {user_query}")
    print(f"  search      : {search_query}")
    print(f"  collection  : {collection}")
    print(f"  top_k={top_k}  threshold={threshold}")
    print("=" * W)

    # ── Step 1: VDB 초기화 & 검색 ────────────────────────────────────────────
    print("\n[1/3] VDB 초기화...")
    t_vdb_start = time.perf_counter()
    from supervisors.v3.tools.vdb import qdrant_vdb
    t_vdb_init = time.perf_counter() - t_vdb_start
    print(f"      완료: {t_vdb_init:.2f}s")

    print(f"      검색 중... (collection={collection}, top_k={top_k})")
    t_search_start = time.perf_counter()
    try:
        raw_docs = qdrant_vdb.search_collection(
            query=search_query,
            collection_name=collection,
            top_k=top_k,
        )
    except Exception as e:
        print(f"\n[ERROR] VDB 검색 실패: {type(e).__name__}: {e}")
        print("컬렉션명이 올바른지, Qdrant 서버가 실행 중인지 확인하세요.")
        sys.exit(1)

    t_search = time.perf_counter() - t_search_start
    print(f"      완료: {t_search:.2f}s  |  {len(raw_docs)}개 문서 반환")

    if not raw_docs:
        print("\n[WARN] 검색 결과가 없습니다. 쿼리 또는 컬렉션명을 확인하세요.")
        sys.exit(0)

    print_vdb_raw(raw_docs, search_query, collection, t_search)

    # ── Step 2: LLMReranker 초기화 ───────────────────────────────────────────
    print("[2/3] LLMReranker 초기화...")
    t_reranker_start = time.perf_counter()
    from supervisors.v3.tools.vdb import _get_reranker
    reranker = _get_reranker()
    reranker.threshold = threshold
    t_reranker_init = time.perf_counter() - t_reranker_start
    print(f"      완료: {t_reranker_init:.2f}s  "
          f"(프롬프트 {len(reranker.rerank_prompt or '')}자)")

    if reranker.rerank_prompt is None:
        print("[ERROR] rerank_prompt=None: 프롬프트 파일 로드 실패.")
        sys.exit(1)

    # ── Step 3: Reranking ────────────────────────────────────────────────────
    print(f"\n[3/3] LLMReranker 평가 중... ({len(raw_docs)}개 문서 병렬)")
    docs_copy = copy.deepcopy(raw_docs)

    t_rerank_start = time.perf_counter()
    result = reranker.rerank(
        docs=docs_copy,
        user_query=user_query,
        search_query=search_query,
    )
    t_rerank = time.perf_counter() - t_rerank_start

    print_rerank_detail(
        user_query=user_query,
        search_query=search_query,
        all_docs=docs_copy,
        result=result,
        elapsed=t_rerank,
        threshold=threshold,
    )

    # ── 최종 요약 ────────────────────────────────────────────────────────────
    total = t_vdb_init + t_search + t_reranker_init + t_rerank
    print("=" * W)
    print("  SUMMARY")
    print(f"  VDB 초기화      : {t_vdb_init:.2f}s")
    print(f"  VDB 검색        : {t_search:.2f}s  ({len(raw_docs)}개)")
    print(f"  Reranker 초기화 : {t_reranker_init:.2f}s")
    print(f"  Reranker 평가   : {t_rerank:.2f}s  ({len(raw_docs)}개 병렬)")
    print(f"  {'─' * 46}")
    print(f"  총 소요 시간    : {total:.2f}s")
    print(f"  통과 문서       : {len(result)} / {len(raw_docs)}")
    if result:
        top_score = result[0].metadata.get("overall_score", 0.0)
        top_name  = result[0].metadata.get("document_name", "unknown")
        print(f"  rank=1 문서     : {top_name}  (score={top_score:.3f})")
    print("=" * W)
    print()


if __name__ == "__main__":
    main()
