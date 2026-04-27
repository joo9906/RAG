"""
LightRAG_process.py — 삽입 + 시각화 + 쿼리 파이프라인
  · 삽입   : 매 LLM/EMB 호출마다 토큰·비용·시간 출력, 파일별 소계
  · 시각화 : knowledge_graph.html
  · 쿼리   : 단계별 타이밍 + 쿼리별 비용 + 답변 출처 추적
  · 캐시   : 쿼리 임베딩 + LLM 응답 캐시 (코사인 유사도 매칭)
  · 모드비교: naive/local/global/hybrid 4종 비교
  · 배치   : 파일에서 질문 읽어 일괄 실행 → md 저장
  · 통계   : 그래프 노드/엣지/타입 분포/연결도 분석
"""
import os
import sys
import html
import asyncio
import argparse
import time
import json
import hashlib
import re
import numpy as np
from datetime import datetime
import networkx as nx
from pyvis.network import Network
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete
from lightrag.utils import EmbeddingFunc
from openai import AsyncOpenAI

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import tiktoken
    _enc = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    _enc = None

# ==============================================================================
# ⚙️  CONFIG  — 이 블록만 수정하면 전체 파이프라인이 맞춰 동작합니다.
# ==============================================================================

# ------------------------------------------------------------------------------
# [1] 데이터 경로
# ------------------------------------------------------------------------------

# API 키 로드 소스
# • .env.json 파일에 {"openai_api_key": "sk-..."} 형식으로 저장하세요.
ENV_JSON_PATH = '../.env.json'          # 환경변수 JSON 파일 위치

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# LightRAG가 생성하는 모든 파일을 저장하는 디렉터리
# (GraphML, KV 스토어, 캐시 JSON, 시각화 HTML 등)
WORKING_DIR = os.path.join(_BASE_DIR, "lightrag_before_chunk_test")

# 삽입할 마크다운 파일이 들어있는 폴더
MD_DIR      = os.path.join(_BASE_DIR, "../test_md")

# Qdrant 벡터 DB 접속 주소
# • 도커로 로컬에 띄움: docker run -d -p 6333:6333 --name qdrant-local qdrant/qdrant
QDRANT_URL  = "http://localhost:6333"

# Qdrant 컬렉션 이름 (WORKING_DIR 하나당 일치시켜놓으면 충돌 방지)
QDRANT_COLLECTION = "lightrag_before_chunk_test"

# ------------------------------------------------------------------------------
# [2] LLM 모델 선택
# ------------------------------------------------------------------------------

# 사용 가능한 값:
#   "mini" → gpt-4o-mini   (빠르고 저렴한 범용 모델)
#   "4o"   → gpt-4o        (높은 답변 품질, 복잡한 의학 질문에 유리)
LLM_MODEL = "mini"

# 각 모델의 주의
#   1억 토큰당 공식 단가 (입력/출력 구분)
_COST_TABLE = {
    "mini": {"in": 0.000150, "out": 0.000600, "name": "gpt-4o-mini"},
    #          ^ $0.15/1M     ^ $0.60/1M
    "4o":   {"in": 0.002500, "out": 0.010000, "name": "gpt-4o"},
    #          ^ $2.50/1M     ^ $10.00/1M
}

# ------------------------------------------------------------------------------
# [3] 임베딩 모델 선택
# ------------------------------------------------------------------------------

# 사용할 OpenAI 임베딩 모델
# • "text-embedding-3-small": 1536인 (dim 선택 불가), $0.020/1M — 파일기반 vdb에 적합
# • "text-embedding-3-large": 3072인 기본, dimensions로 축소 가능, $0.130/1M — Qdrant에 적합
EMB_MODEL = "text-embedding-3-large"

# 임베딩 출력 차원 (= Qdrant 컬렉션 차원과 반드시 일치해야 함)
# • small:  1536인 (고정값, 수정 불가)
# • large:  256 / 1024 / 2048 / 3072 등 자유롭게 선택 가능
# 특이사항: Qdrant 컬렉션을 처음 만들 때의 차원과 항상 동일해야 함.
#          이미 만들어진 컬렉션은 정의 후 관리 > 컬렉션을 삭제 및 재생성이 필요합니다.
EMB_DIM   = 2048

# 임베딩 단가 (참고용, 비용 계산에만 사용)
# • small: 0.000020  ($0.020/1M)
# • large: 0.000130  ($0.130/1M)
_EMB_COST = 0.000130

# 임베딩 토큰 입력 최대 치 (이보다 긴 텍스트는 잘림)
# • small 분석에: 8,191이 구조적 한계
# • large 비으슴 + 좋은 표현력 원하면 8192 권장
EMB_MAX_TOKENS = 8192

# ------------------------------------------------------------------------------
# [4] 청크(Chunk) 설정
# ------------------------------------------------------------------------------

# 도큐먼트 1청크의 최대 토큰 수
# • 작을수록: 세분화 높음, LLM 호출 증가, 복잡한 관계 담기 어려움
# • 클수록: 청크가 크지만 엔티티 정밀도 하락, 중복 내용 증가
# • 합의나쁨점: LightRAG 노드수 vs 비용 트레이드off
CHUNK_SIZE    = 1000   # 토큰 (gpt-4o-mini 기준 약 750 단어)
CHUNK_OVERLAP = 150    # 이전 청크와 겹치는 토큰 수(컨텍스트 유지용)

# ------------------------------------------------------------------------------
# [5] 쿼리 캐시 설정
# ------------------------------------------------------------------------------

# 유사 질문 코사인 유사도 임계값
# • 0.95 이상: 거의 동일한 질문에만 히트 (엄기)
# • 0.90 이상: 비슷한 질문에도 히트 (편리)
# • 0.85 이하: 가대화 (LLM 비용 절감 효과는 있으나 정확도 하락 위험)
CACHE_SIMILARITY_THRESHOLD = 0.8

# ==============================================================================
# ⭐ CONFIG 끝
# ==============================================================================

# CONFIG에서 환경변수 적용 (이 아래는 직접 수정하지 마세요)
with open(ENV_JSON_PATH, 'r', encoding='utf-8') as f:
    _env = json.load(f)
OPENAI_API_KEY = _env['openai_api_key']
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["QDRANT_URL"]     = QDRANT_URL

os.makedirs(WORKING_DIR, exist_ok=True)

# OpenAI 비동기 클라이언트 (직접 호출용 — dimensions 파라미터 지원)
_oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def _raw_embed(texts: list[str]) -> np.ndarray:
    """
    AsyncOpenAI 클라이언트를 직접 호출하는 임베딩 함수.
    LightRAG의 openai_embed 래퍼는 dimensions/model kwargs를 지원하지 않으므로
    OpenAI API를 직접 호출하여 CONFIG 설정을 정확히 반영한다.

    • EMB_MODEL : CONFIG [3] (text-embedding-3-small / large)
    • EMB_DIM   : CONFIG [3] (large만 축소 가능, small은 1536 고정)
    반환값: shape=(len(texts), EMB_DIM) 의 float32 numpy 배열
    """
    kwargs = {"model": EMB_MODEL, "input": texts}
    # text-embedding-3-large는 dimensions 파라미터로 출력 차원을 줄일 수 있음
    # text-embedding-3-small은 dimensions 미지원 (1536 고정)
    if "large" in EMB_MODEL:
        kwargs["dimensions"] = EMB_DIM
    resp = await _oai_client.embeddings.create(**kwargs)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)



# =============================================
# 보조 함수 (CONFIG 변수를 파생)
# =============================================

def _llm_cost_rates():
    """LLM_MODEL에 해당하는 단가 dict 반환 (in/out 키)"""
    return _COST_TABLE.get(LLM_MODEL, _COST_TABLE["mini"])

# =============================================
# 토큰 추적 상태
# =============================================
_phase    = "insert"
_call_log = []
_file_log = []

_usage = {
    "insert": {"llm_in": 0, "llm_out": 0, "emb": 0},
    "query":  {"llm_in": 0, "llm_out": 0, "emb": 0},
}

_file_summary: list = []


def _count(text: str) -> int:
    if _enc and text:
        return len(_enc.encode(str(text)))
    return len(str(text)) // 4


def _cost(in_tok, out_tok, emb_tok=0):
    rates = _llm_cost_rates()
    return (in_tok  / 1000 * rates["in"]
            + out_tok / 1000 * rates["out"]
            + emb_tok / 1000 * _EMB_COST)


# =============================================
# 쿼리 임베딩 + 응답 캐시
# =============================================
CACHE_SIMILARITY_THRESHOLD = 0.92

class QueryCache:
    def __init__(self, cache_path: str):
        self.path = cache_path
        self.data: dict = {}
        self._load()

    def _load(self):
        """캐시 JSON을 로드함.
        임베딩 차원이 현재 CONFIG(EMB_DIM)과 다르면 자동 쳐소하여
        차원 불일치로 인한 크래시를 방지한다."""
        if not os.path.exists(self.path):
            self.data = {}
            return

        with open(self.path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        # 캐시에 저장된 첫 번째 항목의 embedding 차원과 현재 EMB_DIM 비교
        sample_dim = None
        for entry in loaded.values():
            emb = entry.get("embedding")
            if emb:
                sample_dim = len(emb)
                break

        if sample_dim is not None and sample_dim != EMB_DIM:
            # 차원 불일치 → 기존 캐시는 무효 (다른 모델로 먼저 왼던 쳠시)
            bak_path = self.path + f".dim{sample_dim}.bak"
            import shutil
            shutil.copy2(self.path, bak_path)
            print(f"  [캐시] ⚠ 차원 불일치 ({sample_dim}D != {EMB_DIM}D)")
            print(f"  [캐시] 기존 캐시를 {os.path.basename(bak_path)} 로 백업 후 초기화합니다.")
            self.data = {}
            self.save()  # 빈 캐시로 덮어쓰기
        else:
            self.data = loaded
            print(f"  [캐시] {len(self.data)}건 로드됨 (dim={sample_dim or EMB_DIM})")

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _cosine_sim(a: list, b: list) -> float:
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        if a_arr.shape != b_arr.shape:
            # 차원이 다른 항목은 유사도 0으로 치리 (스킵)
            return 0.0
        dot = np.dot(a_arr, b_arr)
        norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        return float(dot / norm) if norm > 0 else 0.0

    def search(self, query_emb: list, threshold: float = None) -> tuple:
        """query_emb와 가장 유사한 캐시 항목 반환.
        차원이 다른 항목은 자동으로 스킵하여 ValueError를 방지한다."""
        if threshold is None:
            threshold = CACHE_SIMILARITY_THRESHOLD
        best_sim = 0.0
        best_entry = None
        q_dim = len(query_emb)
        for cid, entry in self.data.items():
            emb = entry.get("embedding")
            if not emb:
                continue
            # 차원이 다른 유산 항목 스킵 (이전 모델로 저장된 센트리 방어)
            if len(emb) != q_dim:
                continue
            sim = self._cosine_sim(query_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry
        if best_sim >= threshold:
            return best_entry, best_sim
        return None, best_sim

    def store(self, query, embedding, answer, mode, llm_in, llm_out, emb_tok, cost, sec):
        cache_id = hashlib.md5(query.encode("utf-8")).hexdigest()[:12]
        self.data[cache_id] = {
            "query": query,
            "embedding": embedding,   # list[float], len=EMB_DIM
            "emb_dim": len(embedding), # 나중에 차원 확인용
            "answer": answer,
            "mode": mode,
            "llm_in_tok": llm_in, "llm_out_tok": llm_out,
            "emb_tok": emb_tok, "cost": round(cost, 6), "sec": round(sec, 2),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.save()


    def summary(self) -> str:
        n = len(self.data)
        if n == 0:
            return "캐시 비어 있음"
        total_cost = sum(e.get("cost", 0) for e in self.data.values())
        return f"캐시 {n}건 | 누적 비용 ${total_cost:.5f}"


_query_cache: QueryCache | None = None


# =============================================
# LLM / 임베딩 래퍼
# =============================================
async def tracked_llm(prompt, system_prompt=None, history_messages=[], **kwargs):
    in_tok = _count(system_prompt or "") + _count(prompt)
    for m in (history_messages or []):
        in_tok += _count(m.get("content", ""))

    t = time.time()
    if LLM_MODEL == "4o":
        result = await gpt_4o_complete(
            prompt, system_prompt=system_prompt,
            history_messages=history_messages, **kwargs,
        )
    else:
        result = await gpt_4o_mini_complete(
            prompt, system_prompt=system_prompt,
            history_messages=history_messages, **kwargs,
        )
    dur     = time.time() - t
    out_tok = _count(result)

    _usage[_phase]["llm_in"]  += in_tok
    _usage[_phase]["llm_out"] += out_tok

    if _phase == "insert":
        call_no = sum(1 for e in _file_log if e["kind"] == "llm") + 1
        _file_log.append({"kind": "llm", "in": in_tok, "out": out_tok, "sec": dur})
        c = _cost(in_tok, out_tok)
        print(
            f"    LLM #{call_no:<3} | "
            f"입력 {in_tok:>6,} tok | 출력 {out_tok:>5,} tok | "
            f"${c:.5f} | {dur:.1f}초"
        )
    else:
        # 쿼리 단계: prompt 일부 저장 (출처 추적용)
        _call_log.append({
            "type": "llm", "in": in_tok, "out": out_tok, "sec": dur,
            "prompt_preview": prompt[:800],
            "result_preview": result[:300],
        })

    return result


async def tracked_embed(texts, **kwargs):
    # _raw_embed: OpenAI API를 직접 호출 (model + dimensions 정확히 적용)
    t = time.time()
    result = await _raw_embed(list(texts) if not isinstance(texts, list) else texts)
    dur = time.time() - t
    tok = sum(_count(x) for x in (texts if isinstance(texts, list) else [texts]))

    _usage[_phase]["emb"] += tok

    if _phase == "insert":
        emb_no = sum(1 for e in _file_log if e["kind"] == "emb") + 1
        _file_log.append({"kind": "emb", "in": tok, "out": 0, "sec": dur})
        print(f"    EMB #{emb_no:<3} | {tok:>6,} tok | {dur:.2f}초")
    else:
        _call_log.append({"type": "emb", "in": tok, "out": 0, "sec": dur})

    return result


# =============================================
# 삽입
# =============================================
async def insert_documents(rag: LightRAG) -> None:
    global _file_log

    md_files = sorted(f for f in os.listdir(MD_DIR) if f.endswith(".md"))
    if not md_files:
        print(f"MD 파일 없음: {MD_DIR}")
        return

    total = len(md_files)
    for idx, fname in enumerate(md_files, 1):
        _file_log = []
        fpath = os.path.join(MD_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            document = f.read().strip()

        if not document:
            print(f"  [{idx}/{total}] 건너뜀 (빈 파일): {fname}")
            continue

        print(f"\n  [{idx}/{total}] > {fname}  ({len(document):,} chars)")
        print(f"  {'─'*72}")

        t_file = time.time()
        await rag.ainsert(document)
        file_sec = time.time() - t_file

        llm_calls = [e for e in _file_log if e["kind"] == "llm"]
        emb_calls = [e for e in _file_log if e["kind"] == "emb"]
        in_tok  = sum(e["in"]  for e in llm_calls)
        out_tok = sum(e["out"] for e in llm_calls)
        emb_tok = sum(e["in"]  for e in emb_calls)

        print(f"  {'─'*72}")
        print(
            f"  [{idx}/{total}] done  "
            f"LLM {len(llm_calls)}회 | "
            f"입력 {in_tok:>7,} | 출력 {out_tok:>6,} | "
            f"임베딩 {emb_tok:>6,} | "
            f"소계 ${_cost(in_tok, out_tok, emb_tok):.5f} | {file_sec:.1f}초"
        )
        _file_summary.append({
            "name": fname[:42], "calls": len(llm_calls),
            "in": in_tok, "out": out_tok, "emb": emb_tok, "sec": file_sec,
        })


# =============================================
# 삽입 요약 테이블
# =============================================
def print_insert_summary(elapsed: float) -> None:
    W = 104
    rates = _llm_cost_rates()
    print(f"\n{'='*W}")
    print(f"{'[ 삽입 토큰 사용량 요약 ]':^{W}}")
    print(f"{'='*W}")
    print(
        f"  {'파일명':<43}| {'LLM':>4} | "
        f"{'입력tok':>9} | {'출력tok':>8} | "
        f"{'임베딩tok':>10} | {'비용($)':>9} | {'시간':>6}"
    )
    print(f"  {'─'*99}")

    t_calls = t_in = t_out = t_emb = 0
    t_cost = t_sec = 0.0

    for s in _file_summary:
        c = _cost(s["in"], s["out"], s["emb"])
        print(
            f"  {s['name']:<43}| {s['calls']:>4} | "
            f"{s['in']:>9,} | {s['out']:>8,} | "
            f"{s['emb']:>10,} | {c:>9.5f} | {s['sec']:>5.1f}초"
        )
        t_calls += s["calls"]; t_in += s["in"]; t_out += s["out"]
        t_emb   += s["emb"];   t_cost += c;     t_sec += s["sec"]

    print(f"  {'─'*99}")
    print(
        f"  {'합계':<43}| {t_calls:>4} | "
        f"{t_in:>9,} | {t_out:>8,} | "
        f"{t_emb:>10,} | {t_cost:>9.5f} | {t_sec:>5.1f}초"
    )
    print(f"{'='*W}")
    print(f"  LLM 입력  : {t_in:>9,} tok  ${t_in / 1000 * rates['in']:.5f}")
    print(f"  LLM 출력  : {t_out:>9,} tok  ${t_out / 1000 * rates['out']:.5f}")
    print(f"  임베딩    : {t_emb:>9,} tok  ${t_emb / 1000 * _EMB_COST:.5f}")
    print(f"  소계      :             ${t_cost:.5f}  (= ₩{t_cost * 1380:,.0f})")
    print(f"  소요 시간 : {elapsed:.1f}초")
    print(f"{'='*W}\n")


# =============================================
# 그래프 시각화
# =============================================
def visualize_graph(graphml_path=None, output_html=None, max_nodes=1000):
    if output_html is None:
        output_html = os.path.join(WORKING_DIR, "knowledge_graph.html")

    if graphml_path is None:
        candidates = [
            os.path.join(WORKING_DIR, f)
            for f in os.listdir(WORKING_DIR)
            if f.endswith(".graphml")
        ]
        if not candidates:
            print("GraphML 파일 없음. 삽입을 먼저 실행하세요.")
            return
        graphml_path = max(candidates, key=os.path.getmtime)

    print(f"[시각화] GraphML 로딩: {graphml_path}")
    G = nx.read_graphml(graphml_path)
    print(f"  노드: {G.number_of_nodes()}, 엣지: {G.number_of_edges()}")

    if G.number_of_nodes() > max_nodes:
        print(f"  노드 {max_nodes}개 초과 -> 상위 {max_nodes}개만 표시")
        top = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        G = G.subgraph([n for n, _ in top]).copy()

    net = Network(height="900px", width="100%", bgcolor="#0f0f1a",
                  font_color="#e0e0e0", directed=G.is_directed(), notebook=False)
    net.set_options("""
    {
      "physics": { "barnesHut": { "gravitationalConstant": -8000,
        "springLength": 150, "springConstant": 0.04, "damping": 0.09 },
        "minVelocity": 0.75 },
      "edges": { "smooth": {"type":"dynamic"}, "color": {"inherit":"both"},
        "width": 1.5, "font": {"size":10,"color":"#aaaaaa","strokeWidth":0} },
      "nodes": { "shape":"dot", "font":{"size":13,"color":"#ffffff"}, "borderWidth":2 },
      "interaction": { "hover":true, "navigationButtons":true,
        "keyboard":true, "tooltipDelay":200 }
    }
    """)

    type_colors: dict = {}
    palette = ["#4fc3f7","#81c784","#ffb74d","#e57373","#ba68c8",
               "#4db6ac","#f06292","#aed581","#ff8a65","#90a4ae"]

    def get_color(etype):
        if etype not in type_colors:
            type_colors[etype] = palette[len(type_colors) % len(palette)]
        return type_colors[etype]

    for node_id, attrs in G.nodes(data=True):
        label  = attrs.get("entity_name", attrs.get("id", str(node_id)))
        etype  = attrs.get("entity_type", "UNKNOWN")
        desc   = attrs.get("description", "")
        color  = get_color(etype)
        degree = G.degree(node_id)
        net.add_node(str(node_id), label=str(label)[:40],
                     title=f"<b>[{etype}]</b> {label}<br/>{desc[:200]}",
                     color={"background": color, "border": "#ffffff",
                            "highlight": {"background": "#ffffff", "border": color}},
                     size=max(10, min(40, degree * 3)),
                     font={"color": "#ffffff", "size": 13})

    for src, dst, attrs in G.edges(data=True):
        relation = attrs.get("relation_name", attrs.get("relation", ""))
        try:    weight = float(attrs.get("weight", 1.0))
        except: weight = 1.0
        keywords = attrs.get("keywords", "")
        net.add_edge(str(src), str(dst),
                     title=f"{relation}<br/>keywords: {keywords}",
                     label=str(relation)[:25] if relation else "",
                     width=max(1.0, min(5.0, weight * 2)),
                     color={"opacity": 0.7})

    legend = ("<div style='position:fixed;top:10px;right:10px;background:#1a1a2e;"
              "padding:12px;border-radius:8px;font-family:sans-serif;font-size:12px;"
              "color:#fff;z-index:999;'><b>Entity Types</b><br/>")
    for etype, color in type_colors.items():
        legend += (f"<span style='display:inline-block;width:12px;height:12px;"
                   f"background:{color};border-radius:50%;margin-right:5px;'></span>"
                   f"{html.escape(etype)}<br/>")
    legend += "</div>"

    net.save_graph(output_html)
    with open(output_html, "r", encoding="utf-8") as f:
        content = f.read()
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(content.replace("</body>", f"{legend}</body>"))

    print(f"  시각화 완료 -> {os.path.abspath(output_html)}")


# =============================================
# [신규] 그래프 통계 (--stats)
# =============================================
def print_graph_stats():
    candidates = [
        os.path.join(WORKING_DIR, f)
        for f in os.listdir(WORKING_DIR) if f.endswith(".graphml")
    ]
    if not candidates:
        print("GraphML 파일 없음. 삽입을 먼저 실행하세요.")
        return

    gpath = max(candidates, key=os.path.getmtime)
    G = nx.read_graphml(gpath)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # 엔티티 타입 분포
    type_dist: dict[str, int] = {}
    for _, attrs in G.nodes(data=True):
        etype = attrs.get("entity_type", "UNKNOWN")
        type_dist[etype] = type_dist.get(etype, 0) + 1

    # 연결도
    degrees = [d for _, d in G.degree()]
    avg_deg = sum(degrees) / len(degrees) if degrees else 0
    max_deg = max(degrees) if degrees else 0
    isolated = sum(1 for d in degrees if d == 0)

    # 연결 컴포넌트
    if G.is_directed():
        components = list(nx.weakly_connected_components(G))
    else:
        components = list(nx.connected_components(G))

    # 연결도 상위 엔티티
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]

    W = 60
    print(f"\n{'='*W}")
    print(f"  [그래프 통계] {gpath}")
    print(f"{'='*W}")
    print(f"  노드: {n_nodes:,}개  |  엣지: {n_edges:,}개")
    print(f"  연결 컴포넌트: {len(components)}개  |  고립 노드: {isolated}개")
    print(f"  평균 연결도: {avg_deg:.1f}  |  최대 연결도: {max_deg}")
    print()

    print(f"  [엔티티 타입 분포]")
    for etype, cnt in sorted(type_dist.items(), key=lambda x: -x[1]):
        bar = "#" * min(30, cnt)
        print(f"    {etype:<20} {cnt:>5}개  {bar}")
    print()

    print(f"  [연결도 상위 10 엔티티]")
    for node_id, deg in top_nodes:
        attrs = G.nodes[node_id]
        name  = attrs.get("entity_name", node_id)[:35]
        etype = attrs.get("entity_type", "?")
        print(f"    {name:<36} [{etype:<12}] 연결={deg}")
    print(f"{'='*W}\n")


# =============================================
# [신규] 답변 출처 추출 (프롬프트에서)
# =============================================
def _extract_sources(call_log: list) -> list[str]:
    """쿼리 LLM 호출 프롬프트에서 엔티티/출처 이름을 추출."""
    sources = []
    for entry in call_log:
        if entry.get("type") != "llm":
            continue
        prompt = entry.get("prompt_preview", "")
        # LightRAG 프롬프트에서 엔티티 이름 패턴 추출
        # 보통 "Entity: XXX" 또는 큰따옴표 안의 이름
        names = re.findall(r'entity[_\s]*name["\s:]*([^"\n,]+)', prompt, re.IGNORECASE)
        names += re.findall(r'"([A-Z][A-Za-z0-9\s\-]{2,30})"', prompt)
        for n in names:
            n = n.strip()
            if n and n not in sources and len(n) > 1:
                sources.append(n)
    return sources[:15]  # 상위 15개만


# =============================================
# 쿼리 임베딩 단독 호출
# =============================================
async def _embed_query(query: str) -> list:
    # _raw_embed를 통해 model+dimensions가 적용된 임베딩으로 쿼리 벡터 획득
    result = await _raw_embed([query])
    return result[0].tolist()


# =============================================
# 쿼리 (단건) — 캐시 → 실행 → 출처 → 캐시 저장
# =============================================
async def _run_one_query(rag: LightRAG, query: str, mode: str = "hybrid",
                         silent: bool = False) -> dict:
    """
    단일 쿼리 실행. 결과 dict 반환.
    silent=True면 터미널 출력 최소화 (배치/모드비교용).
    """
    global _call_log, _query_cache
    _call_log = []
    u_before = {k: _usage["query"][k] for k in _usage["query"]}

    if not silent:
        print(f"\n{'='*60}")
        print(f"[쿼리] {query}  (mode={mode})")
        print('='*60)

    # 1) 쿼리 임베딩
    t_cache = time.time()
    query_emb = await _embed_query(query)
    emb_sec = time.time() - t_cache
    emb_tok = _count(query)
    _usage[_phase]["emb"] += emb_tok
    if not silent:
        print(f"  [임베딩] {emb_tok}tok | {emb_sec:.2f}초")

    # 2) 캐시 조회
    if _query_cache is not None and mode == "hybrid":
        cached, sim = _query_cache.search(query_emb)
        if cached is not None:
            if not silent:
                print(f"\n  ** 캐시 히트 ** (유사도: {sim:.4f})")
                print(f"  원본 질문: {cached['query']}")
                print(f"\n{cached['answer']}")
                print(f"\n  [캐시 히트] 절약 ${cached['cost']:.5f} | {emb_sec:.2f}초")
            return {
                "query": query, "mode": mode, "answer": cached["answer"],
                "cache_hit": True, "similarity": sim,
                "sec": emb_sec, "llm_in": 0, "llm_out": 0, "emb_tok": emb_tok,
                "cost": 0, "sources": [],
            }
        if not silent and sim > 0:
            print(f"  [캐시] 최대 유사도 {sim:.4f} (임계값 {CACHE_SIMILARITY_THRESHOLD} 미달)")

    # 3) 정상 실행
    t_total = time.time()
    result  = await rag.aquery(query, param=QueryParam(mode=mode, enable_rerank=False))
    total_sec = time.time() - t_total

    if not silent:
        print(result)

    # 타이밍
    emb_calls     = [c for c in _call_log if c["type"] == "emb"]
    llm_calls     = [c for c in _call_log if c["type"] == "llm"]
    retrieval_llm = llm_calls[:-1] if len(llm_calls) > 1 else []
    gen_llm       = llm_calls[-1] if llm_calls else None

    t_emb     = sum(c["sec"] for c in emb_calls)
    t_ret_llm = sum(c["sec"] for c in retrieval_llm)
    t_gen     = gen_llm["sec"] if gen_llm else 0
    t_other   = total_sec - t_emb - t_ret_llm - t_gen

    q_in   = _usage["query"]["llm_in"]  - u_before["llm_in"]
    q_out  = _usage["query"]["llm_out"] - u_before["llm_out"]
    q_emb  = _usage["query"]["emb"]     - u_before["emb"]
    q_cost = _cost(q_in, q_out, q_emb)

    # 출처 추적
    sources = _extract_sources(_call_log)

    if not silent:
        print(f"\n  ┌─ [타이밍 분석] {'─'*35}")
        print(f"  │  쿼리 임베딩      : {t_emb:.2f}초  ({len(emb_calls)}회)")
        print(f"  │  벡터·그래프 서칭 : {t_other:.2f}초")
        if retrieval_llm:
            print(f"  │  검색 보조 LLM    : {t_ret_llm:.2f}초  ({len(retrieval_llm)}회)")
        print(f"  │  답변 생성 LLM    : {t_gen:.2f}초")
        print(f"  │  {'─'*40}")
        print(f"  │  합계             : {total_sec:.2f}초")
        print(f"  └─ 비용             : ${q_cost:.5f}  (= ₩{q_cost*1380:,.0f})")
        if sources:
            print(f"\n  [출처 엔티티] {', '.join(sources[:8])}")

    # 4) 캐시 저장
    if _query_cache is not None and mode == "hybrid":
        _query_cache.store(
            query=query, embedding=query_emb, answer=result,
            mode=mode, llm_in=q_in, llm_out=q_out, emb_tok=q_emb,
            cost=q_cost, sec=total_sec,
        )
        if not silent:
            print(f"  [캐시] 저장 완료 ({_query_cache.summary()})")

    return {
        "query": query, "mode": mode, "answer": result,
        "cache_hit": False, "sec": total_sec,
        "llm_in": q_in, "llm_out": q_out, "emb_tok": q_emb,
        "cost": q_cost, "sources": sources,
    }


# =============================================
# 기본 쿼리 목록
# =============================================
DEFAULT_QUERIES = [
    "기넥신 누구한테 영업할까?",
    "외과에 어떤 약을 추천할까",
    "류마티스 관절염에 도움이 되는 약",
    "기넥신을 누구한테 쓰면 안돼?"
]

async def run_queries(rag: LightRAG) -> None:
    global _phase
    _phase = "query"
    for query in DEFAULT_QUERIES:
        await _run_one_query(rag, query)


# =============================================
# [신규] 모드 비교 (--mode-compare)
# =============================================
MODES = ["naive", "local", "global", "hybrid"]

async def run_mode_compare(rag: LightRAG, query: str) -> None:
    global _phase
    _phase = "query"

    print(f"\n{'='*70}")
    print(f"  [모드 비교] {query}")
    print(f"{'='*70}")

    results = []
    for mode in MODES:
        print(f"\n  --- {mode} ---")
        res = await _run_one_query(rag, query, mode=mode, silent=True)
        results.append(res)
        print(f"  {mode:<8} | {res['sec']:>6.2f}초 | "
              f"in={res['llm_in']:,} out={res['llm_out']:,} | "
              f"${res['cost']:.5f}")

    # 비교 테이블
    print(f"\n  {'─'*66}")
    print(f"  {'모드':<8} | {'시간':>6} | {'LLM입력':>8} | {'LLM출력':>8} | {'비용':>9} | {'답변길이':>6}")
    print(f"  {'─'*66}")
    for r in results:
        print(f"  {r['mode']:<8} | {r['sec']:>5.2f}초 | {r['llm_in']:>8,} | "
              f"{r['llm_out']:>8,} | ${r['cost']:>8.5f} | {len(r['answer']):>5,}자")
    print(f"  {'─'*66}")

    # 답변 비교 (각 200자 미리보기)
    print(f"\n  [답변 미리보기]")
    for r in results:
        preview = r["answer"].replace("\n", " ") #[:200]
        print(f"  [{r['mode']}] {preview}...")
    print()


# =============================================
# [신규] 배치 쿼리 (--batch FILE)
# =============================================
async def run_batch_queries(rag: LightRAG, batch_file: str) -> None:
    global _phase
    _phase = "query"

    with open(batch_file, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not queries:
        print(f"질문이 없습니다: {batch_file}")
        return

    print(f"\n[배치 쿼리] {len(queries)}개 질문 로드")
    results = []
    for i, query in enumerate(queries, 1):
        print(f"\n  [{i}/{len(queries)}] {query}")
        res = await _run_one_query(rag, query, silent=True)
        results.append(res)
        llm_info = "캐시히트" if res["cache_hit"] else f"LLM in={res['llm_in']:,}"
        print(f"    -> {res['sec']:.2f}초 | ${res['cost']:.5f} | {llm_info}")

    # 결과 저장
    out_path = os.path.join(WORKING_DIR, "batch_result.md")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# 배치 쿼리 결과",
        f"\n생성: {now}  |  모델: {_COST_TABLE[LLM_MODEL]['name']}  |  "
        f"질문: {len(results)}개",
        "",
        "| # | 질문 | 시간 | LLM입력 | LLM출력 | 비용 | 캐시 |",
        "|---|------|------|---------|---------|------|------|",
    ]
    total_cost = 0
    total_sec = 0
    for i, r in enumerate(results, 1):
        total_cost += r["cost"]
        total_sec  += r["sec"]
        cache_str = f"HIT({r.get('similarity',0):.2f})" if r["cache_hit"] else "-"
        lines.append(
            f"| {i} | {r['query'][:40]} | {r['sec']:.2f}초 | "
            f"{r['llm_in']:,} | {r['llm_out']:,} | ${r['cost']:.5f} | {cache_str} |"
        )
    lines.append(
        f"| | **합계** | **{total_sec:.2f}초** | | | **${total_cost:.5f}** | |"
    )
    lines.append("")

    for i, r in enumerate(results, 1):
        lines.append(f"## Q{i}. {r['query']}")
        if r["sources"]:
            lines.append(f"출처: {', '.join(r['sources'][:5])}")
        lines.append(f"\n{r['answer']}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  [배치 결과 저장] {out_path}")
    print(f"  합계: {total_sec:.2f}초 | ${total_cost:.5f}")


# =============================================
# 전체 크레딧 최종 요약
# =============================================
def print_total_cost(total_elapsed: float) -> None:
    ins = _usage["insert"]
    qry = _usage["query"]
    ic  = _cost(ins["llm_in"], ins["llm_out"], ins["emb"])
    qc  = _cost(qry["llm_in"], qry["llm_out"], qry["emb"])
    tot = ic + qc

    print(f"\n{'='*52}")
    print(f"[전체 크레딧 최종 요약]")
    print(f"  모델: {_COST_TABLE[LLM_MODEL]['name']}")
    print(f"  삽입(관계 형성) : ${ic:.5f}  (= ₩{ic*1380:,.0f})")
    print(f"  쿼리            : ${qc:.5f}  (= ₩{qc*1380:,.0f})")
    print(f"  ─────────────────────────────────────")
    print(f"  합계            : ${tot:.5f}  (= ₩{tot*1380:,.0f})")
    print(f"  총 소요 시간    : {total_elapsed:.1f}초")
    print(f"{'='*52}\n")


# =============================================
# 메인
# =============================================
# CHUNK_SIZE / CHUNK_OVERLAP 은 CONFIG 블록에서 이미 정의됨
# (argparse --chunk-size, --chunk-overlap 옵션으로 런타임 재정의 가능)


def _build_rag() -> LightRAG:
    print(f"  [청크] size={CHUNK_SIZE}tok  overlap={CHUNK_OVERLAP}tok")
    print(f"  [임베딩] {EMB_MODEL}  dim={EMB_DIM}  ${_EMB_COST*1000:.3f}/1M")
    return LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=tracked_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMB_DIM,           # CONFIG [3] EMB_DIM
            max_token_size=EMB_MAX_TOKENS,   # CONFIG [3] EMB_MAX_TOKENS
            func=tracked_embed,
            model_name=EMB_MODEL,            # 컬렉션 접미사 생성용 (예: text_embedding_3_large_2048d)
        ),
        chunk_token_size=CHUNK_SIZE,             # CONFIG [4] CHUNK_SIZE
        chunk_overlap_token_size=CHUNK_OVERLAP,  # CONFIG [4] CHUNK_OVERLAP
        entity_extract_max_gleaning=2,           # 기본 1 → 모호한 청크 재추출 횟수
        force_llm_summary_on_merge=3,            # 기본 8 → 머지 시 LLM 요약 발동 기준 낮춤
        addon_params={
            "language": "Korean",                # 한국어 문서 추출 정확도 향상
        },
        vector_storage="QdrantVectorDBStorage",
        vector_db_storage_cls_kwargs={
            # CONFIG [1] QDRANT_COLLECTION 참조 — WORKING_DIR와 이름을 맞춰두면 충돌 방지
            "collection_name": QDRANT_COLLECTION,
        },
    )


async def main() -> None:
    global _phase, _query_cache, CHUNK_SIZE, CHUNK_OVERLAP, LLM_MODEL, CACHE_SIMILARITY_THRESHOLD

    parser = argparse.ArgumentParser(
        description="LightRAG_process.py — 토큰 모니터링 + 쿼리 캐시 파이프라인",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("-q", "--query", nargs="+", metavar="WORD",
                        help="쿼리 실행 (삽입·시각화 생략)")
    parser.add_argument("--no-cache", action="store_true",
                        help="쿼리 캐시 비활성화")
    parser.add_argument("--cache-threshold", type=float, default=CACHE_SIMILARITY_THRESHOLD,
                        help=f"캐시 유사도 임계값 (기본: {CACHE_SIMILARITY_THRESHOLD})")
    parser.add_argument("--show-cache", action="store_true",
                        help="캐시 내용 출력 후 종료")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"청크 최대 토큰 수 (기본: {CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP,
                        help=f"청크 오버랩 토큰 수 (기본: {CHUNK_OVERLAP})")
    parser.add_argument("--llm", choices=["mini", "4o"], default=LLM_MODEL,
                        help="LLM 모델: mini=gpt-4o-mini, 4o=gpt-4o")
    # 신규 옵션
    parser.add_argument("--stats", action="store_true",
                        help="그래프 통계 (노드/엣지/타입 분포/연결도) 출력")
    parser.add_argument("--mode-compare", nargs="+", metavar="WORD",
                        help="4개 모드(naive/local/global/hybrid) 비교")
    parser.add_argument("--batch", metavar="FILE",
                        help="파일에서 질문 읽어 일괄 실행 -> batch_result.md 저장")
    parser.add_argument("--skip-insert", action="store_true",
                        help="삽입 건너뛰고 시각화+쿼리만 실행")
    args = parser.parse_args()

    # 설정 반영
    CHUNK_SIZE    = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    LLM_MODEL     = args.llm
    print(f"  [모델] {_COST_TABLE[LLM_MODEL]['name']}  "
          f"(in=${_COST_TABLE[LLM_MODEL]['in']*1000:.2f}/1M  "
          f"out=${_COST_TABLE[LLM_MODEL]['out']*1000:.2f}/1M)")

    # 캐시
    cache_path = os.path.join(WORKING_DIR, "query_cache.json")
    if not args.no_cache:
        _query_cache = QueryCache(cache_path)
        if args.cache_threshold != CACHE_SIMILARITY_THRESHOLD:
            CACHE_SIMILARITY_THRESHOLD = args.cache_threshold
            print(f"  [캐시] 유사도 임계값: {CACHE_SIMILARITY_THRESHOLD}")
    else:
        _query_cache = None
        print("  [캐시] 비활성화")

    # --show-cache
    if args.show_cache:
        if _query_cache and _query_cache.data:
            print(f"\n{'='*60}")
            print(f"  쿼리 캐시 ({len(_query_cache.data)}건)")
            print(f"{'='*60}")
            for cid, entry in _query_cache.data.items():
                print(f"\n  [{cid}] {entry['query']}")
                print(f"    {entry['created_at']} | ${entry['cost']:.5f} | {entry['sec']}초")
                print(f"    답변: {entry['answer'][:100]}...")
        else:
            print("캐시가 비어 있습니다.")
        return

    # --stats
    if args.stats:
        print_graph_stats()
        return

    t_start = time.time()
    rag = _build_rag()
    await rag.initialize_storages()

    # --mode-compare
    if args.mode_compare:
        query_text = " ".join(args.mode_compare)
        await run_mode_compare(rag, query_text)
        print_total_cost(time.time() - t_start)
        return

    # --batch
    if args.batch:
        await run_batch_queries(rag, args.batch)
        print_total_cost(time.time() - t_start)
        return

    # -q 쿼리 전용
    if args.query:
        query_text = " ".join(args.query)
        print(f"[쿼리 모드] working_dir: {WORKING_DIR}")
        if _query_cache:
            print(f"[쿼리 캐시] {_query_cache.summary()}")
        _phase = "query"
        await _run_one_query(rag, query_text)
        print_total_cost(time.time() - t_start)
        return

    # ── 전체 파이프라인 ──────────────────────────
    print("LightRAG_process.py -- 삽입 + 시각화 + 통계 + 쿼리 + 캐시\n")

    # 1. 삽입
    if not args.skip_insert:
        _phase = "insert"
        t1 = time.time()
        await insert_documents(rag)
        print_insert_summary(time.time() - t1)
    else:
        print("[삽입 건너뜀]\n")

    # 2. 시각화
    t2 = time.time()
    visualize_graph()
    print(f"[시간] 시각화: {time.time()-t2:.1f}초\n")

    # 3. 그래프 통계
    print_graph_stats()

    # 4. 쿼리
    t3 = time.time()
    await run_queries(rag)
    print(f"\n[시간] 쿼리 합계: {time.time()-t3:.1f}초")

    # 5. 캐시 상태
    if _query_cache:
        print(f"\n[쿼리 캐시] 최종 -> {_query_cache.summary()}")
        print(f"  저장 위치: {cache_path}")

    print_total_cost(time.time() - t_start)


if __name__ == "__main__":
    asyncio.run(main())
