"""
gen_test.py — lightrag_gem_ra  vs  lightrag_c  성능 비교

두 LightRAG 지식그래프에 동일한 질문 5개를 투입하여 다음 항목을 비교한다.
  · 그래프 통계: 노드 수, 엣지 수, GraphML 크기
  · 쿼리별: 응답 시간, LLM 입력/출력 토큰, 임베딩 토큰, 비용(USD/KRW)
  · 답변 전문
  · 종합 비용 비교표

결과는 lightrag_gem_ra/summary.md 에 저장한다.

[캐시 처리 방침]
LightRAG 초기화 시 `enable_llm_cache=False` 를 명시하여 
이전 응답이 캐시되어 비용이 0으로 측정되는 현상을 방지한다.
"""

import os
import sys
import asyncio
import time
import networkx as nx
from datetime import datetime
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import tiktoken
    _enc = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    _enc = None

# ─────────────────────────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────────────────────────
_BASE        = os.path.dirname(os.path.abspath(__file__))   # LightRAG/
DIR_GEM_RA   = os.path.join(_BASE, "lightrag_gem_ra")
DIR_C        = os.path.join(_BASE, "lightrag_c")
SUMMARY_PATH = os.path.join(DIR_GEM_RA, "summary.md")

# ─────────────────────────────────────────────────────────────────
# API 환경 변수
# ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_BASE, "..", ".env"))
except ImportError:
    pass

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

# ─────────────────────────────────────────────────────────────────
# LLM 단가 (gpt-4o-mini 기준)
# ─────────────────────────────────────────────────────────────────
_LLM_IN_COST  = 0.000150   # $ per 1,000 tokens  (입력)
_LLM_OUT_COST = 0.000600   # $ per 1,000 tokens  (출력)
_EMB_COST     = 0.000020   # $ per 1,000 tokens  (text-embedding-3-small)
KRW_RATE      = 1_380      # USD → KRW 환율 (고정 가정치)

# ─────────────────────────────────────────────────────────────────
# 비교용 질문 5개
# ─────────────────────────────────────────────────────────────────
QUESTIONS = [
    "리리카와 쎄레브렉스의 안전성 정보를 비교해줘",
    "기넥신의 주요 상병코드와 효능을 알려줘",
    "류마티스관절염 치료에 사용되는 약물과 관련 상병코드는?",
    "프로맥의 소장 점막 손상 예방 효과에 대해 설명해줘",
    "수벡스정(편두통 치료제)의 작용기전과 특장점을 설명해줘",
]

# ─────────────────────────────────────────────────────────────────
# 시스템 정의
# ─────────────────────────────────────────────────────────────────
SYSTEMS = [
    {
        "id":          "gem_ra",
        "name":        "lightrag_gem_ra",
        "label":       "RAGAnything (gem_RA.py)",
        "working_dir": DIR_GEM_RA,
        "collection":  None,          # 파일 기반 vdb (Qdrant 미사용)
        "desc":        "gem_RA.py로 구축: RAGAnything + docling 파싱 + 파일 기반 벡터 DB",
    },
    {
        "id":          "c",
        "name":        "lightrag_c",
        "label":       "직접 삽입 (c.py)",
        "working_dir": DIR_C,
        "collection":  "lightrag_c",  # Qdrant 컬렉션 지정
        "desc":        "c.py로 구축: 원본 MD 직접 삽입 + Qdrant 벡터 DB",
    },
]

# ─────────────────────────────────────────────────────────────────
# 토큰 카운터 & 메트릭 클래스
# ─────────────────────────────────────────────────────────────────
def _count(text) -> int:
    if _enc and text:
        try:
            return len(_enc.encode(str(text)))
        except Exception:
            pass
    return max(1, len(str(text)) // 4)


class QueryMetrics:
    """단일 쿼리에 대한 토큰/시간/비용 메트릭"""
    def __init__(self):
        self.llm_in: int = 0
        self.llm_out: int = 0
        self.emb: int = 0
        self.llm_calls: int = 0
        self.emb_calls: int = 0
        self.call_log: list = []   # {"type": "llm"|"emb", "in": n, "out": n, "sec": f}
        self.sec: float = 0.0

    def cost(self) -> float:
        return (
            self.llm_in  / 1000 * _LLM_IN_COST
            + self.llm_out / 1000 * _LLM_OUT_COST
            + self.emb     / 1000 * _EMB_COST
        )

    def reset(self):
        self.__init__()


# ─────────────────────────────────────────────────────────────────
# 전역 메트릭 싱글턴 (asyncio 단일 스레드이므로 안전)
# ─────────────────────────────────────────────────────────────────
_M = QueryMetrics()


async def _tracked_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    if history_messages is None:
        history_messages = []
    in_tok = _count(system_prompt or "") + _count(prompt)
    for m in history_messages:
        in_tok += _count(m.get("content", ""))

    t0 = time.perf_counter()
    result = await gpt_4o_mini_complete(
        prompt, system_prompt=system_prompt,
        history_messages=history_messages, **kwargs,
    )
    dur = time.perf_counter() - t0
    out_tok = _count(result)

    _M.llm_in   += in_tok
    _M.llm_out  += out_tok
    _M.llm_calls += 1
    _M.call_log.append({"type": "llm", "in": in_tok, "out": out_tok, "sec": dur})
    return result


async def _tracked_embed(texts, **kwargs):
    _fn = openai_embed.func if hasattr(openai_embed, "func") else openai_embed
    tok = sum(_count(x) for x in (texts if isinstance(texts, list) else [texts]))

    t0 = time.perf_counter()
    result = await _fn(texts, **kwargs)
    dur = time.perf_counter() - t0

    _M.emb       += tok
    _M.emb_calls += 1
    _M.call_log.append({"type": "emb", "in": tok, "out": 0, "sec": dur})
    return result


# ─────────────────────────────────────────────────────────────────
# 그래프 통계 수집
# ─────────────────────────────────────────────────────────────────
def get_graph_stats(wdir: str) -> dict:
    if not os.path.isdir(wdir):
        return {"nodes": 0, "edges": 0, "file": "디렉토리 없음", "size_mb": 0.0}
    files = [
        os.path.join(wdir, f)
        for f in os.listdir(wdir)
        if f.endswith(".graphml")
    ]
    if not files:
        return {"nodes": 0, "edges": 0, "file": "GraphML 없음", "size_mb": 0.0}

    latest = max(files, key=os.path.getmtime)
    G = nx.read_graphml(latest)
    size_mb = os.path.getsize(latest) / 1024 / 1024

    # 엔티티 타입 분포
    type_dist: dict[str, int] = {}
    for _, attrs in G.nodes(data=True):
        etype = attrs.get("entity_type", "UNKNOWN")
        type_dist[etype] = type_dist.get(etype, 0) + 1

    # 평균 degree
    degrees = [d for _, d in G.degree()]
    avg_deg = sum(degrees) / max(len(degrees), 1)

    return {
        "nodes":    G.number_of_nodes(),
        "edges":    G.number_of_edges(),
        "file":     os.path.basename(latest),
        "size_mb":  size_mb,
        "type_dist": type_dist,
        "avg_degree": avg_deg,
        "density": nx.density(G),
    }


# ─────────────────────────────────────────────────────────────────
# LightRAG 인스턴스 생성
# ─────────────────────────────────────────────────────────────────
_EMBED_FUNC = EmbeddingFunc(
    embedding_dim=1536,
    max_token_size=8192,
    func=_tracked_embed,
)


def build_rag(sys_info: dict) -> LightRAG:
    kwargs: dict = dict(
        working_dir=sys_info["working_dir"],
        llm_model_func=_tracked_llm,
        embedding_func=_EMBED_FUNC,
        enable_llm_cache=False,  # 테스트를 위해 LLM 캐시 비활성화
    )
    if sys_info["collection"]:
        kwargs["vector_storage"] = "QdrantVectorDBStorage"
        kwargs["vector_db_storage_cls_kwargs"] = {
            "collection_name": sys_info["collection"]
        }
    return LightRAG(**kwargs)


# ─────────────────────────────────────────────────────────────────
# 단일 쿼리 실행 + 타이밍 분석
# ─────────────────────────────────────────────────────────────────
async def run_one_query(rag: LightRAG, question: str) -> dict:
    """질문 1건 실행 → {answer, metrics_snapshot, timing} 반환

    캐시는 enable_llm_cache=False 로 테스트 전에 이미 비워진 상태이므로
    반드시 실제 LLM 호출이 발생한다.
    """
    _M.reset()

    t0 = time.perf_counter()
    try:
        answer = await rag.aquery(
            question,
            param=QueryParam(mode="hybrid"),
        )
    except Exception as e:
        answer = f"[쿼리 오류] {e}"
    total_sec = time.perf_counter() - t0
    _M.sec = total_sec

    # 타이밍 분해
    llm_log = [c for c in _M.call_log if c["type"] == "llm"]
    emb_log = [c for c in _M.call_log if c["type"] == "emb"]
    gen_llm       = llm_log[-1] if llm_log else None
    retrieval_llm = llm_log[:-1] if len(llm_log) > 1 else []

    timing = {
        "total_sec":   total_sec,
        "emb_sec":     sum(c["sec"] for c in emb_log),
        "ret_llm_sec": sum(c["sec"] for c in retrieval_llm),
        "gen_llm_sec": gen_llm["sec"] if gen_llm else 0.0,
        "other_sec":   0.0,   # 계산 후 채움
        "emb_calls":   len(emb_log),
        "ret_calls":   len(retrieval_llm),
    }
    timing["other_sec"] = max(
        0.0,
        total_sec - timing["emb_sec"] - timing["ret_llm_sec"] - timing["gen_llm_sec"],
    )

    return {
        "answer":    str(answer),
        "llm_in":   _M.llm_in,
        "llm_out":  _M.llm_out,
        "emb":      _M.emb,
        "llm_calls": _M.llm_calls,
        "emb_calls": _M.emb_calls,
        "cost":     _M.cost(),
        "timing":   timing,
    }


# ─────────────────────────────────────────────────────────────────
# 비교 실험 메인 로직
# ─────────────────────────────────────────────────────────────────
async def compare():
    wall_start = time.perf_counter()
    now_dt = datetime.now()
    now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*65}")
    print(f"  🔬 LightRAG 그래프 성능 비교")
    print(f"  실험 시각: {now_str}")
    print(f"{'='*65}")

    # ── 1. 그래프 통계 ───────────────────────────────────────────
    print("\n📊 [STEP 1] 그래프 통계 수집...")
    graph_stats: dict[str, dict] = {}
    for sys in SYSTEMS:
        stats = get_graph_stats(sys["working_dir"])
        graph_stats[sys["id"]] = stats
        print(
            f"  [{sys['id']}] 노드: {stats['nodes']:,} | "
            f"엣지: {stats['edges']:,} | "
            f"GraphML: {stats['file']} ({stats['size_mb']:.2f} MB) | "
            f"평균 degree: {stats['avg_degree']:.2f}"
        )

    # ── 2. RAG 초기화 ────────────────────────────────────────────
    print("\n🔧 [STEP 2] LightRAG 초기화...")
    rags: dict[str, LightRAG] = {}
    for sys in SYSTEMS:
        print(f"  [{sys['id']}] 초기화 중...", end=" ", flush=True)
        rag = build_rag(sys)
        await rag.initialize_storages()
        rags[sys["id"]] = rag
        print("✅")

    # ── 3. 질문별 비교 실행 ──────────────────────────────────────
    print("\n🧪 [STEP 3] 질문 비교 실행 (각 5문 × 2시스템)\n")
    all_results: list[dict] = []

    for q_no, question in enumerate(QUESTIONS, 1):
        print(f"  {'─'*60}")
        print(f"  Q{q_no}. {question}")
        print(f"  {'─'*60}")

        q_entry = {
            "q_no":     q_no,
            "question": question,
            "by_sys":   {},
        }

        for sys in SYSTEMS:
            print(f"  ▶ [{sys['name']}] 쿼리 중... ", end="", flush=True)
            res = await run_one_query(rags[sys["id"]], question)
            q_entry["by_sys"][sys["id"]] = res
            print(
                f"완료 {res['timing']['total_sec']:.1f}초 | "
                f"${res['cost']:.5f} (≈₩{res['cost']*KRW_RATE:,.0f})"
            )

        all_results.append(q_entry)
        print()

    total_wall = time.perf_counter() - wall_start

    # ── 4. summary.md 생성 ───────────────────────────────────────
    print(f"📝 [STEP 4] summary.md 작성 중...")
    _write_summary(now_str, graph_stats, all_results, total_wall)
    print(f"  ✅ 저장 완료: {SUMMARY_PATH}")
    print(f"\n총 소요 시간: {total_wall:.1f}초")


# ─────────────────────────────────────────────────────────────────
# summary.md 생성 함수
# ─────────────────────────────────────────────────────────────────
def _write_summary(
    now_str: str,
    graph_stats: dict,
    all_results: list,
    total_wall: float,
) -> None:
    lines: list[str] = []
    sep  = "─" * 70

    # ── 헤더 ──────────────────────────────────────────────────────
    lines += [
        "# LightRAG 그래프 성능 비교 보고서",
        "",
        f"> 실험 시각: {now_str}",
        f"> 총 소요 시간: {total_wall:.1f}초",
        f"> 비교 모델: gpt-4o-mini (LLM) / text-embedding-3-small (임베딩)",
        f"> 쿼리 모드: hybrid (local + global + naive 통합)",
        f"> 캐시: **비활성화** (no_cache=True — 실제 LLM 호출 기반 측정)",
        "",
    ]

    # ── 시스템 개요 ───────────────────────────────────────────────
    lines += ["## 비교 대상 시스템", ""]
    for sys in SYSTEMS:
        lines += [
            f"### {sys['name']}",
            f"- **경로**: `{sys['working_dir']}`",
            f"- **설명**: {sys['desc']}",
            f"- **벡터 저장소**: {'Qdrant (컬렉션: ' + sys['collection'] + ')' if sys['collection'] else '파일 기반 vdb (JSON)'}",
            "",
        ]

    # ── 그래프 통계 ───────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 그래프 통계",
        "",
        "| 항목 | lightrag_gem_ra | lightrag_c |",
        "|------|:--------------:|:---------:|",
    ]

    stats_a = graph_stats.get("gem_ra", {})
    stats_c = graph_stats.get("c", {})

    def _sv(d, k, fmt=","):
        v = d.get(k, "-")
        if isinstance(v, (int, float)) and fmt:
            return f"{v:{fmt}}" if fmt != "f2" else f"{v:.2f}"
        return str(v)

    rows = [
        ("노드(엔티티) 수",  "nodes",  ","),
        ("엣지(관계) 수",    "edges",  ","),
        ("GraphML 파일명",   "file",   ""),
        ("GraphML 크기",     "size_mb", "f2"),
        ("평균 노드 Degree", "avg_degree", "f2"),
        ("그래프 밀도",      "density",   "f2"),
    ]

    for label, key, fmt in rows:
        va = _sv(stats_a, key, fmt) + (" MB" if key == "size_mb" else "")
        vc = _sv(stats_c, key, fmt) + (" MB" if key == "size_mb" else "")
        lines.append(f"| {label} | {va} | {vc} |")

    lines += [""]

    # 엔티티 타입 분포
    lines += ["### 엔티티 타입 분포", ""]
    lines += ["| 타입 | lightrag_gem_ra | lightrag_c |", "|------|:--------------:|:---------:|"]
    all_types = sorted(
        set(stats_a.get("type_dist", {}).keys()) | set(stats_c.get("type_dist", {}).keys())
    )
    for t in all_types:
        va = stats_a.get("type_dist", {}).get(t, 0)
        vc = stats_c.get("type_dist", {}).get(t, 0)
        lines.append(f"| `{t}` | {va:,} | {vc:,} |")
    lines += [""]

    # ── 쿼리별 비교 요약표 ────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 쿼리별 성능 비교 요약",
        "",
        "### 응답 시간 (초)",
        "",
        "| # | 질문 | gem_ra | lightrag_c | 차이 |",
        "|---|------|:------:|:----------:|:----:|",
    ]
    for r in all_results:
        a = r["by_sys"].get("gem_ra", {})
        c = r["by_sys"].get("c", {})
        ta = a.get("timing", {}).get("total_sec", 0)
        tc = c.get("timing", {}).get("total_sec", 0)
        diff = ta - tc
        sign = "+" if diff > 0 else ""
        lines.append(
            f"| Q{r['q_no']} | {r['question'][:35]} "
            f"| {ta:.2f}초 | {tc:.2f}초 | {sign}{diff:.2f}초 |"
        )

    lines += ["", "### 비용 (USD)", "", "| # | 질문 | gem_ra | lightrag_c | 차이 |", "|---|------|:------:|:----------:|:----:|"]
    for r in all_results:
        a = r["by_sys"].get("gem_ra", {})
        c = r["by_sys"].get("c", {})
        ca = a.get("cost", 0)
        cc = c.get("cost", 0)
        diff = ca - cc
        sign = "+" if diff > 0 else ""
        lines.append(
            f"| Q{r['q_no']} | {r['question'][:35]} "
            f"| ${ca:.5f} | ${cc:.5f} | {sign}${diff:.5f} |"
        )

    lines += ["", "### 토큰 사용량", ""]
    lines += [
        "| # | 시스템 | LLM 입력 | LLM 출력 | 임베딩 | LLM 호출 | 답변 길이(char) |",
        "|---|--------|:--------:|:--------:|:------:|:--------:|:--------------:|",
    ]
    for r in all_results:
        for sys in SYSTEMS:
            res = r["by_sys"].get(sys["id"], {})
            lines.append(
                f"| Q{r['q_no']} | {sys['name']} "
                f"| {res.get('llm_in', 0):,} "
                f"| {res.get('llm_out', 0):,} "
                f"| {res.get('emb', 0):,} "
                f"| {res.get('llm_calls', 0)} "
                f"| {len(res.get('answer', '')):,} |"
            )

    lines += [""]

    # ── 타이밍 분해 (시스템별) ────────────────────────────────────
    lines += [
        "---",
        "",
        "## 단계별 타이밍 분해",
        "",
    ]
    for sys in SYSTEMS:
        lines += [f"### {sys['name']}", ""]
        lines += [
            "| # | 질문 | 임베딩 | 그래프/벡터 서칭 | 검색보조LLM | 답변생성LLM | 합계 |",
            "|---|------|:------:|:---------------:|:----------:|:----------:|:----:|",
        ]
        for r in all_results:
            res = r["by_sys"].get(sys["id"], {})
            t = res.get("timing", {})
            lines.append(
                f"| Q{r['q_no']} | {r['question'][:30]} "
                f"| {t.get('emb_sec', 0):.2f}s "
                f"| {t.get('other_sec', 0):.2f}s "
                f"| {t.get('ret_llm_sec', 0):.2f}s "
                f"| {t.get('gen_llm_sec', 0):.2f}s "
                f"| **{t.get('total_sec', 0):.2f}s** |"
            )
        lines += [""]

    # ── 총 비용 비교 ──────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 총 비용 비교",
        "",
    ]

    def _sum_sys(sys_id: str, key: str) -> float:
        return sum(r["by_sys"].get(sys_id, {}).get(key, 0) for r in all_results)

    for sys in SYSTEMS:
        sid = sys["id"]
        total_cost = _sum_sys(sid, "cost")
        total_in   = int(_sum_sys(sid, "llm_in"))
        total_out  = int(_sum_sys(sid, "llm_out"))
        total_emb  = int(_sum_sys(sid, "emb"))
        total_sec  = sum(
            r["by_sys"].get(sid, {}).get("timing", {}).get("total_sec", 0)
            for r in all_results
        )
        lines += [
            f"### {sys['name']}",
            "",
            f"| 항목 | 값 |",
            "|------|------|",
            f"| LLM 총 입력 토큰 | {total_in:,} tok |",
            f"| LLM 총 출력 토큰 | {total_out:,} tok |",
            f"| 임베딩 총 토큰   | {total_emb:,} tok |",
            f"| 총 LLM 비용      | ${total_in/1000*_LLM_IN_COST + total_out/1000*_LLM_OUT_COST:.5f} |",
            f"| 총 EMB 비용      | ${total_emb/1000*_EMB_COST:.5f} |",
            f"| **총 합계 비용** | **${total_cost:.5f}** (≈ ₩{total_cost*KRW_RATE:,.0f}) |",
            f"| 5개 쿼리 소요    | {total_sec:.1f}초 |",
            "",
        ]

    # 두 시스템 최종 비교
    cost_a = sum(r["by_sys"].get("gem_ra", {}).get("cost", 0) for r in all_results)
    cost_c = sum(r["by_sys"].get("c",      {}).get("cost", 0) for r in all_results)
    sec_a  = sum(r["by_sys"].get("gem_ra", {}).get("timing", {}).get("total_sec", 0) for r in all_results)
    sec_c  = sum(r["by_sys"].get("c",      {}).get("timing", {}).get("total_sec", 0) for r in all_results)

    lines += [
        "### 최종 비교표",
        "",
        "| 지표 | lightrag_gem_ra | lightrag_c | 비교 |",
        "|------|:--------------:|:---------:|:----:|",
        f"| 노드 수           | {stats_a.get('nodes', 0):,} | {stats_c.get('nodes', 0):,} | {'gem_ra 多' if stats_a.get('nodes',0) > stats_c.get('nodes',0) else 'c 多'} |",
        f"| 엣지 수           | {stats_a.get('edges', 0):,} | {stats_c.get('edges', 0):,} | {'gem_ra 多' if stats_a.get('edges',0) > stats_c.get('edges',0) else 'c 多'} |",
        f"| 총 쿼리 시간     | {sec_a:.1f}초 | {sec_c:.1f}초 | {'gem_ra 빠름' if sec_a < sec_c else 'c 빠름'} |",
        f"| 총 쿼리 비용     | ${cost_a:.5f} | ${cost_c:.5f} | {'gem_ra 저렴' if cost_a < cost_c else 'c 저렴'} |",
        f"| 벡터 저장소       | 파일 기반 vdb | Qdrant | - |",
        "",
    ]

    # ── 질문별 전체 답변 ──────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 질문별 상세 답변",
        "",
    ]
    for r in all_results:
        lines += [
            f"### Q{r['q_no']}. {r['question']}",
            "",
        ]
        for sys in SYSTEMS:
            res = r["by_sys"].get(sys["id"], {})
            t   = res.get("timing", {})
            lines += [
                f"#### [{sys['name']}]",
                f"- **응답 시간**: {t.get('total_sec', 0):.2f}초",
                f"- **비용**: ${res.get('cost', 0):.5f} (≈ ₩{res.get('cost', 0)*KRW_RATE:,.0f})",
                f"- **LLM**: 입력 {res.get('llm_in', 0):,}tok / 출력 {res.get('llm_out', 0):,}tok / {res.get('llm_calls', 0)}회",
                f"- **임베딩**: {res.get('emb', 0):,}tok / {res.get('emb_calls', 0)}회",
                f"- **타이밍**: 임베딩 {t.get('emb_sec',0):.2f}s | 서칭 {t.get('other_sec',0):.2f}s | 검색보조LLM {t.get('ret_llm_sec',0):.2f}s | 답변생성 {t.get('gen_llm_sec',0):.2f}s",
                "",
                "**답변:**",
                "",
                res.get("answer", "(없음)"),
                "",
                sep,
                "",
            ]

    # ── 푸터 ──────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        f"*생성 시각: {now_str} | 총 실험 시간: {total_wall:.1f}초*",
    ]

    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ─────────────────────────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(compare())
