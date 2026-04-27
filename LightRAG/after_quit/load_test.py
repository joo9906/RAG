"""
LightRAG_test.py — 20명 동시 접속 부하 테스트
  · 기존 그래프(lightrag_before_chunk_test)를 대상으로 쿼리
  · 20명이 동시에 질문을 보내는 상황 시뮬레이션
  · 각 유저별 응답 시간, 토큰, 비용 추적
  · 전체 throughput, p50/p95/p99 지연, 실패율 측정
  · 결과를 lightrag_before_chunk_test/load_test_result.md 에 저장
"""
import os
import sys
import json
import asyncio
import time
import statistics
from datetime import datetime

import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete   # openai_embed 는 쓰지 않음
from lightrag.utils import EmbeddingFunc
from openai import AsyncOpenAI                         # 직접 호출로 dimensions 지원

try:
    import tiktoken
    _enc = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    _enc = None

# ==============================================================================
# ⚙️  CONFIG  — LightRAG_process.py 와 동일하게 맞춰 두세요.
# ==============================================================================

# [경로] API 키 JSON 경로
ENV_JSON_PATH = '../.env.json'

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.join(_BASE_DIR, "lightrag_before_chunk_test")

# Qdrant 설정 — LightRAG_process.py 의 QDRANT_URL / QDRANT_COLLECTION 과 동일해야 함
QDRANT_URL        = "http://localhost:6333"
QDRANT_COLLECTION = "lightrag_before_chunk_test"

# [임베딩 모델] — LightRAG_process.py 의 EMB_MODEL / EMB_DIM 과 반드시 일치
# • 저장된 벡터 차원과 다르면 "Embedding dimension mismatch" 에러 발생
EMB_MODEL = "text-embedding-3-large"
EMB_DIM   = 2048     # Qdrant 컬렉션 생성 시 사용한 차원과 같아야 함

# [LLM 단가] gpt-4o-mini 기준 ($/ 1,000 토큰)
_LLM_IN_COST  = 0.000150   # $0.15 / 1M
_LLM_OUT_COST = 0.000600   # $0.60 / 1M
_EMB_COST     = 0.000130   # $0.130 / 1M  (text-embedding-3-large 기준)

# [부하 테스트 설정]
CONCURRENT_USERS = 20     # 동시 접속 유저 수
QUERIES_PER_USER = 3      # 유저당 쿼리 수

# ==============================================================================
# ⭐ CONFIG 끝
# ==============================================================================

# CONFIG 에서 환경변수 적용
with open(ENV_JSON_PATH, 'r', encoding='utf-8') as f:
    _env = json.load(f)
OPENAI_API_KEY = _env['openai_api_key']
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["QDRANT_URL"]     = QDRANT_URL

# OpenAI 비동기 클라이언트 (직접 호출 — dimensions 파라미터 지원)
_oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


# =============================================
# 테스트 질문 풀 (유저별 랜덤 배정)
# =============================================
QUERY_POOL = [
    "리리카와 쎄레브렉스의 안전성 정보를 비교해줘",
    "쎄레브렉스의 주요 효능이 뭐야?",
    "기넥신 관련 상병코드 알려줘",
    "류마티스 관절염 관련 약 알려줘",
    "편두통 치료에 수벡스가 효과적인 이유는?",
    "파킨슨병 초기 환자에게 추천되는 치료 방법은?",
    "프로맥이 소장 점막에 미치는 효과를 설명해줘",
    "레밋치의 주요 적응증은 뭐야?",
    "기넥신의 PSS 가이드라인 요약해줘",
    "EPSILON 연구의 주요 결과를 알려줘",
]


# =============================================
# 토큰 카운트 / 비용 계산
# =============================================
def _count(text: str) -> int:
    if _enc and text:
        return len(_enc.encode(str(text)))
    return max(1, len(str(text)) // 4)


def _cost(in_tok, out_tok, emb_tok=0):
    return (in_tok  / 1000 * _LLM_IN_COST
            + out_tok / 1000 * _LLM_OUT_COST
            + emb_tok / 1000 * _EMB_COST)


# =============================================
# 임베딩 직접 호출 (openai_embed 대신 AsyncOpenAI 사용)
# openai_embed 래퍼는 model/dimensions kwargs를 통과시키지 않아
# "unexpected keyword argument 'dimensions'" 에러가 발생하므로 직접 호출이 필요
# =============================================
async def _raw_embed(texts: list[str]) -> np.ndarray:
    """
    AsyncOpenAI 를 직접 호출하여 EMB_MODEL + EMB_DIM 설정을 정확히 반영.
    • text-embedding-3-large : dimensions 파라미터로 3072 → EMB_DIM 으로 축소
    • text-embedding-3-small : dimensions 미지원, 1536 고정
    반환값: shape=(len(texts), EMB_DIM) 의 float32 numpy 배열
    """
    kwargs = {"model": EMB_MODEL, "input": texts}
    if "large" in EMB_MODEL:
        kwargs["dimensions"] = EMB_DIM  # large 모델만 dimensions 지원
    resp = await _oai_client.embeddings.create(**kwargs)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


# =============================================
# 유저별 호출 추적
# =============================================
class UserTracker:
    """유저 한 명의 쿼리별 LLM/임베딩 토큰을 추적"""
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.llm_in  = 0
        self.llm_out = 0
        self.emb_tok = 0

    def track_llm(self, in_tok, out_tok):
        self.llm_in  += in_tok
        self.llm_out += out_tok

    def track_emb(self, tok):
        self.emb_tok += tok

    def reset(self):
        self.llm_in = self.llm_out = self.emb_tok = 0


def _make_tracked_llm(tracker: UserTracker):
    """유저별 LLM 래퍼 — 호출마다 토큰 수 집계"""
    async def _llm(prompt, system_prompt=None, history_messages=[], **kwargs):
        in_tok = _count(system_prompt or "") + _count(prompt)
        for m in (history_messages or []):
            in_tok += _count(m.get("content", ""))
        result = await gpt_4o_mini_complete(
            prompt, system_prompt=system_prompt,
            history_messages=history_messages, **kwargs,
        )
        tracker.track_llm(in_tok, _count(result))
        return result
    return _llm


def _make_tracked_embed(tracker: UserTracker):
    """
    유저별 임베딩 래퍼.
    _raw_embed() 를 사용하여 EMB_MODEL + EMB_DIM 을 정확히 전달.
    openai_embed 래퍼를 쓰지 않아 dimensions 에러 없음.
    """
    async def _emb(texts, **kwargs):
        # _raw_embed 는 kwargs 를 쓰지 않고 CONFIG 값을 직접 사용
        tok = sum(_count(x) for x in (texts if isinstance(texts, list) else [texts]))
        result = await _raw_embed(list(texts) if not isinstance(texts, list) else texts)
        tracker.track_emb(tok)
        return result
    return _emb


# =============================================
# 단일 유저 시뮬레이션
# =============================================
async def simulate_user(user_id: int, queries: list[str], mode: str = "hybrid") -> list[dict]:
    """
    유저 한 명이 RAG 인스턴스를 만들어 쿼리를 순차 실행.
    LightRAG 인스턴스는 동시 실행에서 공유해도 안전하지만,
    토큰 추적을 유저별로 분리하기 위해 래퍼를 각각 생성.
    """
    tracker = UserTracker(user_id)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=_make_tracked_llm(tracker),
        embedding_func=EmbeddingFunc(
            embedding_dim=EMB_DIM,    # CONFIG 와 동일해야 Qdrant 차원 일치
            max_token_size=8192,
            func=_make_tracked_embed(tracker),
        ),
        vector_storage="QdrantVectorDBStorage",
        vector_db_storage_cls_kwargs={
            # CONFIG 의 QDRANT_COLLECTION — WORKING_DIR 이름과 맞춰두면 충돌 방지
            "collection_name": QDRANT_COLLECTION,
        },
    )
    await rag.initialize_storages()

    results = []
    for qi, query in enumerate(queries):
        tracker.reset()
        t0 = time.perf_counter()
        try:
            answer = await rag.aquery(query, param=QueryParam(mode=mode))
            sec    = time.perf_counter() - t0
            cost   = _cost(tracker.llm_in, tracker.llm_out, tracker.emb_tok)
            results.append({
                "user_id":      user_id,
                "query_idx":    qi,
                "query":        query,
                "answer_len":   len(answer),
                "answer_preview": answer[:100],
                "sec":          sec,
                "llm_in":       tracker.llm_in,
                "llm_out":      tracker.llm_out,
                "emb_tok":      tracker.emb_tok,
                "cost":         cost,
                "error":        None,
            })
        except Exception as e:
            sec = time.perf_counter() - t0
            results.append({
                "user_id":    user_id,
                "query_idx":  qi,
                "query":      query,
                "answer_len": 0,
                "answer_preview": "",
                "sec":        sec,
                "llm_in": 0, "llm_out": 0, "emb_tok": 0, "cost": 0,
                "error":      str(e),
            })

    return results


# =============================================
# 부하 테스트 실행
# =============================================
async def run_load_test():
    import random
    random.seed(42)

    print(f"{'='*70}")
    print(f"  LightRAG_test.py — {CONCURRENT_USERS}명 동시 접속 부하 테스트")
    print(f"{'='*70}")
    print(f"  동시 유저    : {CONCURRENT_USERS}명")
    print(f"  유저당 쿼리  : {QUERIES_PER_USER}개")
    print(f"  총 쿼리      : {CONCURRENT_USERS * QUERIES_PER_USER}개")
    print(f"  임베딩 모델  : {EMB_MODEL}  dim={EMB_DIM}")
    print(f"  질문 풀      : {len(QUERY_POOL)}개")
    print(f"  working_dir  : {WORKING_DIR}")
    print()

    # 유저별 질문 배정
    user_queries = [
        random.choices(QUERY_POOL, k=QUERIES_PER_USER)
        for _ in range(CONCURRENT_USERS)
    ]

    # 동시 실행
    print(f"[시작] {CONCURRENT_USERS}명 동시 실행 중...")
    t_start = time.perf_counter()

    tasks = [simulate_user(uid, qs) for uid, qs in enumerate(user_queries)]
    all_results_nested = await asyncio.gather(*tasks, return_exceptions=True)

    total_sec = time.perf_counter() - t_start
    print(f"[완료] 전체 소요: {total_sec:.2f}초\n")

    # 결과 평탄화
    all_results = []
    for res in all_results_nested:
        if isinstance(res, Exception):
            print(f"  [오류] 유저 실행 실패: {res}")
            continue
        all_results.extend(res)

    # 통계 계산
    success = [r for r in all_results if r["error"] is None]
    failed  = [r for r in all_results if r["error"] is not None]

    if not success:
        print("모든 쿼리 실패!")
        if failed:
            print(f"첫 번째 오류: {failed[0]['error']}")
        return

    latencies = sorted(r["sec"] for r in success)
    n = len(latencies)
    p50 = latencies[int(n * 0.50)]
    p95 = latencies[int(n * 0.95)]
    p99 = latencies[min(int(n * 0.99), n - 1)]
    avg = statistics.mean(latencies)

    total_cost    = sum(r["cost"]    for r in all_results)
    total_llm_in  = sum(r["llm_in"]  for r in all_results)
    total_llm_out = sum(r["llm_out"] for r in all_results)
    total_emb_tok = sum(r["emb_tok"] for r in all_results)
    throughput    = len(success) / total_sec

    # 터미널 출력
    print(f"{'='*60}")
    print(f"  [결과 요약]")
    print(f"{'='*60}")
    print(f"  성공: {len(success)}/{len(all_results)}  |  실패: {len(failed)}")
    print(f"  총 소요: {total_sec:.2f}초  |  처리량: {throughput:.1f} req/sec")
    print()
    print(f"  [지연 시간]")
    print(f"    평균 : {avg:.2f}초")
    print(f"    p50  : {p50:.2f}초")
    print(f"    p95  : {p95:.2f}초")
    print(f"    p99  : {p99:.2f}초")
    print(f"    최소 : {min(latencies):.2f}초")
    print(f"    최대 : {max(latencies):.2f}초")
    print()
    print(f"  [토큰 / 비용]")
    print(f"    LLM 입력  : {total_llm_in:>9,} tok")
    print(f"    LLM 출력  : {total_llm_out:>9,} tok")
    print(f"    임베딩    : {total_emb_tok:>9,} tok")
    print(f"    총 비용   : ${total_cost:.5f} (= ₩{total_cost*1380:,.0f})")
    print(f"    쿼리당 평균: ${total_cost/max(len(success),1):.5f}")
    print(f"{'='*60}")

    # 유저별 요약
    print(f"\n  [유저별 평균 응답 시간]")
    print(f"  {'유저':<6} {'쿼리수':>4} {'평균(초)':>8} {'총비용':>10}")
    print(f"  {'─'*32}")
    for uid in range(CONCURRENT_USERS):
        user_res = [r for r in success if r["user_id"] == uid]
        if user_res:
            u_avg  = statistics.mean(r["sec"] for r in user_res)
            u_cost = sum(r["cost"] for r in user_res)
            print(f"  U{uid:<5} {len(user_res):>4} {u_avg:>7.2f}초 ${u_cost:>9.5f}")

    if failed:
        print(f"\n  [실패 목록]")
        for r in failed:
            print(f"    U{r['user_id']} Q{r['query_idx']}: {r['error'][:80]}")

    # ── 결과 파일 저장 ──────────────────────────────────────────
    out_path = os.path.join(WORKING_DIR, "../load_test_result.md")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# 부하 테스트 결과 — {CONCURRENT_USERS}명 동시 접속",
        f"",
        f"생성: {now}",
        f"모델: gpt-4o-mini  |  임베딩: {EMB_MODEL} dim={EMB_DIM}",
        f"동시 유저: {CONCURRENT_USERS}  |  유저당 쿼리: {QUERIES_PER_USER}"
        f"  |  총 쿼리: {CONCURRENT_USERS * QUERIES_PER_USER}",
        "",
        "## 요약",
        "",
        "| 항목 | 값 |",
        "|------|-----|",
        f"| 성공/전체 | {len(success)}/{len(all_results)} |",
        f"| 총 소요 시간 | {total_sec:.2f}초 |",
        f"| 처리량 | {throughput:.1f} req/sec |",
        f"| 평균 지연 | {avg:.2f}초 |",
        f"| p50 | {p50:.2f}초 |",
        f"| p95 | {p95:.2f}초 |",
        f"| p99 | {p99:.2f}초 |",
        f"| 최소/최대 | {min(latencies):.2f}초 / {max(latencies):.2f}초 |",
        f"| LLM 입력 토큰 | {total_llm_in:,} |",
        f"| LLM 출력 토큰 | {total_llm_out:,} |",
        f"| 임베딩 토큰 | {total_emb_tok:,} |",
        f"| 총 비용 | ${total_cost:.5f} (= ₩{total_cost*1380:,.0f}) |",
        f"| 쿼리당 평균 비용 | ${total_cost/max(len(success),1):.5f} |",
        "",
        "## 지연 분포",
        "",
        "```",
    ]

    # ASCII 히스토그램
    buckets: dict[int, int] = {}
    for lat in latencies:
        b = int(lat)
        buckets[b] = buckets.get(b, 0) + 1
    for b in sorted(buckets):
        bar = "#" * buckets[b]
        lines.append(f"  {b:>2}~{b+1}초: {bar} ({buckets[b]}개)")
    lines.append("```")
    lines.append("")

    # 유저별 테이블
    lines += [
        "## 유저별 성능",
        "",
        "| 유저 | 쿼리수 | 평균(초) | 총비용 |",
        "|------|--------|---------|--------|",
    ]
    for uid in range(CONCURRENT_USERS):
        user_res = [r for r in success if r["user_id"] == uid]
        if user_res:
            u_avg  = statistics.mean(r["sec"] for r in user_res)
            u_cost = sum(r["cost"] for r in user_res)
            lines.append(f"| U{uid} | {len(user_res)} | {u_avg:.2f}초 | ${u_cost:.5f} |")
    lines.append("")

    # 전체 쿼리 상세
    lines += [
        "## 전체 쿼리 상세",
        "",
        "| 유저 | Q# | 질문 | 시간 | LLM입력 | 비용 | 상태 |",
        "|------|----|------|------|---------|------|------|",
    ]
    for r in all_results:
        status  = "OK" if r["error"] is None else "FAIL"
        q_short = r["query"][:30]
        lines.append(
            f"| U{r['user_id']} | {r['query_idx']} | {q_short} | "
            f"{r['sec']:.2f}초 | {r['llm_in']:,} | ${r['cost']:.5f} | {status} |"
        )
    lines.append("")

    if failed:
        lines += ["## 실패 목록", ""]
        for r in failed:
            lines.append(f"- U{r['user_id']} Q{r['query_idx']}: `{r['error'][:100]}`")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  [결과 저장] {out_path}")


# =============================================
# 메인
# =============================================
if __name__ == "__main__":
    asyncio.run(run_load_test())
