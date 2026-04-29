"""
total_process.py — LightRAG 통합 파이프라인

  [삽입]    MD 파일 → LightRAG 삽입, 파일별 LLM/EMB 토큰·비용·시간 출력
  [힐링]    고립 노드(degree=0) 자동 감지 + 4가지 전략으로 해소
              A. prune  — 품질 미달 노드 삭제
              B. embed  — 임베딩 코사인 유사도로 연결
              C. llm    — LLM 관계 제안으로 연결
              D. relink — 소스 청크에서 관계 재추출
  [시각화]  knowledge_graph.html (pyvis)
  [통계]    노드/엣지/타입 분포/연결도/고립 노드 분석
  [쿼리]    단건 / 모드비교(naive·local·global·hybrid) / 배치
  [캐시]    쿼리 임베딩 + LLM 응답 캐시 (코사인 유사도 매칭)

사용 예)
  python total_process.py                          # 전체 파이프라인 (삽입→힐링→시각화→통계→쿼리)
  python total_process.py --skip-insert            # 삽입 생략, 나머지 실행
  python total_process.py --heal                   # 힐링만 (A+B+C 기본 조합)
  python total_process.py --heal --heal-all        # 힐링 A+B+C+D 모두
  python total_process.py --heal --dry-run         # 변경 내용 미리보기만
  python total_process.py --stats                  # 그래프 통계만
  python total_process.py -q 기넥신 누구한테 써?  # 단건 쿼리
  python total_process.py --mode-compare 기넥신 효능은?
  python total_process.py --batch questions.txt
"""

import os
import sys
import html
import copy
import asyncio
import argparse
import time
import json
import hashlib
import re
import shutil
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
# CONFIG  — 이 블록만 수정하면 전체 파이프라인이 맞춰 동작합니다
# ==============================================================================

# [1] 데이터 경로
ENV_JSON_PATH     = "../../.env.json"          # {"openai_api_key": "sk-..."}
_BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR       = os.path.join(_BASE_DIR, "lightrag_before_chunk_test")
MD_DIR            = os.path.join(_BASE_DIR, "../../test_md")
QDRANT_URL        = "http://localhost:6333"
QDRANT_COLLECTION = "lightrag_before_chunk_test"

# [2] LLM 모델  "mini" → gpt-4o-mini  |  "4o" → gpt-4o
LLM_MODEL = "mini"
_COST_TABLE = {
    "mini": {"in": 0.000150, "out": 0.000600, "name": "gpt-4o-mini"},
    "4o":   {"in": 0.002500, "out": 0.010000, "name": "gpt-4o"},
}

# [3] 임베딩 모델
EMB_MODEL      = "text-embedding-3-large"   # or "text-embedding-3-small"
EMB_DIM        = 2048                        # large: 256/1024/2048/3072 자유 선택
_EMB_COST      = 0.000130                    # $0.130/1M
EMB_MAX_TOKENS = 8192

# [4] 청크 설정
CHUNK_SIZE    = 1000   # 토큰
CHUNK_OVERLAP = 150    # 이전 청크와 겹치는 토큰

# [5] 쿼리 캐시
CACHE_SIMILARITY_THRESHOLD = 0.92

# [6] 힐링 기본값
HEAL_PRUNE_MIN_DESC    = 10      # 전략 A: 설명 최소 길이 (이하 삭제)
HEAL_EMBED_THRESHOLD   = 0.75    # 전략 B: 임베딩 코사인 유사도 임계값
HEAL_EMBED_TOP_K       = 2       # 전략 B: 고립 노드 1개당 최대 연결 수
HEAL_LLM_LIMIT         = 50      # 전략 C: LLM 처리 최대 고립 노드 수
HEAL_LLM_MIN_CONF      = 0.5     # 전략 C: 채택할 최소 confidence
HEAL_RELINK_LIMIT      = 30      # 전략 D: re-link 처리 최대 노드 수
HEAL_LLM_BATCH_CANDS   = 20      # 전략 C: 한 LLM 호출당 후보 노드 수

# ==============================================================================
# CONFIG 끝
# ==============================================================================

# 환경변수 적용
with open(os.path.join(_BASE_DIR, ENV_JSON_PATH), "r", encoding="utf-8") as _f:
    _env = json.load(_f)
OPENAI_API_KEY = _env["openai_api_key"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["QDRANT_URL"]     = QDRANT_URL

os.makedirs(WORKING_DIR, exist_ok=True)

_oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ==============================================================================
# 토큰 카운터 & 비용
# ==============================================================================
_phase    = "insert"   # insert | query | heal
_call_log = []
_file_log = []

_usage = {
    "insert": {"llm_in": 0, "llm_out": 0, "emb": 0},
    "query":  {"llm_in": 0, "llm_out": 0, "emb": 0},
    "heal":   {"llm_in": 0, "llm_out": 0, "emb": 0},
}
_file_summary: list = []


def _count(text: str) -> int:
    if _enc and text:
        return len(_enc.encode(str(text)))
    return len(str(text)) // 4


def _llm_cost_rates():
    return _COST_TABLE.get(LLM_MODEL, _COST_TABLE["mini"])


def _cost(in_tok, out_tok, emb_tok=0):
    r = _llm_cost_rates()
    return (in_tok / 1000 * r["in"]
            + out_tok / 1000 * r["out"]
            + emb_tok / 1000 * _EMB_COST)


# ==============================================================================
# 임베딩 헬퍼 (OpenAI 직접 호출)
# ==============================================================================
async def _raw_embed(texts: list[str]) -> np.ndarray:
    """dimensions 파라미터 포함하여 임베딩 API를 직접 호출."""
    kwargs = {"model": EMB_MODEL, "input": texts}
    if "large" in EMB_MODEL:
        kwargs["dimensions"] = EMB_DIM
    resp = await _oai_client.embeddings.create(**kwargs)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ==============================================================================
# LightRAG 래퍼 (토큰 추적)
# ==============================================================================
async def tracked_llm(prompt, system_prompt=None, history_messages=[], **kwargs):
    # LightRAG 내부에서 _priority 등 내부 kwarg를 전달하는 경우 제거
    kwargs.pop("_priority", None)
    kwargs.pop("_timeout", None)
    kwargs.pop("_queue_timeout", None)
    in_tok = _count(system_prompt or "") + _count(prompt)
    for m in (history_messages or []):
        in_tok += _count(m.get("content", ""))

    t = time.time()
    if LLM_MODEL == "4o":
        result = await gpt_4o_complete(
            prompt, system_prompt=system_prompt,
            history_messages=history_messages, **kwargs)
    else:
        result = await gpt_4o_mini_complete(
            prompt, system_prompt=system_prompt,
            history_messages=history_messages, **kwargs)
    dur     = time.time() - t
    out_tok = _count(result)

    _usage[_phase]["llm_in"]  += in_tok
    _usage[_phase]["llm_out"] += out_tok

    if _phase == "insert":
        call_no = sum(1 for e in _file_log if e["kind"] == "llm") + 1
        _file_log.append({"kind": "llm", "in": in_tok, "out": out_tok, "sec": dur})
        c = _cost(in_tok, out_tok)
        print(f"    LLM #{call_no:<3} | 입력 {in_tok:>6,}tok | 출력 {out_tok:>5,}tok | ${c:.5f} | {dur:.1f}초")
    else:
        _call_log.append({
            "type": "llm", "in": in_tok, "out": out_tok, "sec": dur,
            "prompt_preview": prompt[:800],
            "result_preview": result[:300],
        })
    return result


async def tracked_embed(texts, **kwargs):
    t      = time.time()
    result = await _raw_embed(list(texts) if not isinstance(texts, list) else texts)
    dur    = time.time() - t
    tok    = sum(_count(x) for x in (texts if isinstance(texts, list) else [texts]))

    _usage[_phase]["emb"] += tok

    if _phase == "insert":
        emb_no = sum(1 for e in _file_log if e["kind"] == "emb") + 1
        _file_log.append({"kind": "emb", "in": tok, "out": 0, "sec": dur})
        print(f"    EMB #{emb_no:<3} | {tok:>6,}tok | {dur:.2f}초")
    else:
        _call_log.append({"type": "emb", "in": tok, "out": 0, "sec": dur})
    return result


# ==============================================================================
# 쿼리 캐시
# ==============================================================================
class QueryCache:
    def __init__(self, cache_path: str):
        self.path = cache_path
        self.data: dict = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            self.data = {}
            return
        with open(self.path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        sample_dim = None
        for entry in loaded.values():
            emb = entry.get("embedding")
            if emb:
                sample_dim = len(emb)
                break

        if sample_dim is not None and sample_dim != EMB_DIM:
            bak = self.path + f".dim{sample_dim}.bak"
            shutil.copy2(self.path, bak)
            print(f"  [캐시] 차원 불일치 ({sample_dim}D != {EMB_DIM}D) — 초기화 (백업: {os.path.basename(bak)})")
            self.data = {}
            self.save()
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
            return 0.0
        dot  = np.dot(a_arr, b_arr)
        norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        return float(dot / norm) if norm > 0 else 0.0

    def search(self, query_emb: list, threshold: float = None) -> tuple:
        if threshold is None:
            threshold = CACHE_SIMILARITY_THRESHOLD
        best_sim, best_entry = 0.0, None
        q_dim = len(query_emb)
        for entry in self.data.values():
            emb = entry.get("embedding")
            if not emb or len(emb) != q_dim:
                continue
            sim = self._cosine_sim(query_emb, emb)
            if sim > best_sim:
                best_sim, best_entry = sim, entry
        return (best_entry, best_sim) if best_sim >= threshold else (None, best_sim)

    def store(self, query, embedding, answer, mode, llm_in, llm_out, emb_tok, cost, sec):
        cid = hashlib.md5(query.encode()).hexdigest()[:12]
        self.data[cid] = {
            "query": query, "embedding": embedding, "emb_dim": len(embedding),
            "answer": answer, "mode": mode,
            "llm_in_tok": llm_in, "llm_out_tok": llm_out, "emb_tok": emb_tok,
            "cost": round(cost, 6), "sec": round(sec, 2),
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

# ==============================================================================
# LightRAG 인스턴스 빌더
# ==============================================================================
def _build_rag() -> LightRAG:
    print(f"  [청크] size={CHUNK_SIZE}tok  overlap={CHUNK_OVERLAP}tok")
    print(f"  [임베딩] {EMB_MODEL}  dim={EMB_DIM}  ${_EMB_COST*1000:.3f}/1M")
    return LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=tracked_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMB_DIM,
            max_token_size=EMB_MAX_TOKENS,
            func=tracked_embed,
            model_name=EMB_MODEL,
        ),
        chunk_token_size=CHUNK_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP,
        entity_extract_max_gleaning=2,
        force_llm_summary_on_merge=3,
        addon_params={"language": "Korean"},
        vector_storage="QdrantVectorDBStorage",
        vector_db_storage_cls_kwargs={"collection_name": QDRANT_COLLECTION},
    )


# ==============================================================================
# [1] 삽입 파이프라인
# ==============================================================================
async def insert_documents(rag: LightRAG) -> None:
    global _file_log
    md_files = sorted(f for f in os.listdir(MD_DIR) if f.endswith(".md"))
    if not md_files:
        print(f"  MD 파일 없음: {MD_DIR}")
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
        in_tok    = sum(e["in"]  for e in llm_calls)
        out_tok   = sum(e["out"] for e in llm_calls)
        emb_tok   = sum(e["in"]  for e in emb_calls)

        print(f"  {'─'*72}")
        print(
            f"  [{idx}/{total}] done  LLM {len(llm_calls)}회 | "
            f"입력 {in_tok:>7,} | 출력 {out_tok:>6,} | 임베딩 {emb_tok:>6,} | "
            f"소계 ${_cost(in_tok, out_tok, emb_tok):.5f} | {file_sec:.1f}초"
        )
        _file_summary.append({
            "name": fname[:42], "calls": len(llm_calls),
            "in": in_tok, "out": out_tok, "emb": emb_tok, "sec": file_sec,
        })


def print_insert_summary(elapsed: float) -> None:
    W = 104
    rates = _llm_cost_rates()
    print(f"\n{'='*W}")
    print(f"{'[ 삽입 토큰 사용량 요약 ]':^{W}}")
    print(f"{'='*W}")
    print(f"  {'파일명':<43}| {'LLM':>4} | {'입력tok':>9} | {'출력tok':>8} | {'임베딩tok':>10} | {'비용($)':>9} | {'시간':>6}")
    print(f"  {'─'*99}")
    t_calls = t_in = t_out = t_emb = 0
    t_cost  = t_sec = 0.0
    for s in _file_summary:
        c = _cost(s["in"], s["out"], s["emb"])
        print(f"  {s['name']:<43}| {s['calls']:>4} | {s['in']:>9,} | {s['out']:>8,} | {s['emb']:>10,} | {c:>9.5f} | {s['sec']:>5.1f}초")
        t_calls += s["calls"]; t_in += s["in"]; t_out += s["out"]
        t_emb += s["emb"];     t_cost += c;     t_sec += s["sec"]
    print(f"  {'─'*99}")
    print(f"  {'합계':<43}| {t_calls:>4} | {t_in:>9,} | {t_out:>8,} | {t_emb:>10,} | {t_cost:>9.5f} | {t_sec:>5.1f}초")
    print(f"{'='*W}")
    print(f"  LLM 입력  : {t_in:>9,}tok  ${t_in /1000*rates['in']:.5f}")
    print(f"  LLM 출력  : {t_out:>9,}tok  ${t_out/1000*rates['out']:.5f}")
    print(f"  임베딩    : {t_emb:>9,}tok  ${t_emb/1000*_EMB_COST:.5f}")
    print(f"  소계      :             ${t_cost:.5f}  (= ₩{t_cost*1380:,.0f})")
    print(f"  소요 시간 : {elapsed:.1f}초")
    print(f"{'='*W}\n")


# ==============================================================================
# [2] 그래프 통계
# ==============================================================================
def _find_graphml() -> str:
    files = [
        os.path.join(WORKING_DIR, f)
        for f in os.listdir(WORKING_DIR) if f.endswith(".graphml")
    ]
    if not files:
        raise FileNotFoundError(f"GraphML 파일 없음: {WORKING_DIR}")
    return max(files, key=os.path.getmtime)


def print_graph_stats(G: nx.Graph = None, label: str = "") -> None:
    if G is None:
        try:
            gpath = _find_graphml()
            G = nx.read_graphml(gpath)
        except FileNotFoundError as e:
            print(f"  {e}")
            return

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    avg_deg = sum(degrees) / len(degrees) if degrees else 0.0
    max_deg = max(degrees) if degrees else 0
    isolated = sum(1 for d in degrees if d == 0)

    if G.is_directed():
        comps = list(nx.weakly_connected_components(G))
    else:
        comps = list(nx.connected_components(G))

    type_dist: dict[str, int] = {}
    for _, attrs in G.nodes(data=True):
        etype = attrs.get("entity_type", "UNKNOWN")
        type_dist[etype] = type_dist.get(etype, 0) + 1

    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]

    W   = 64
    hdr = f"  그래프 통계{f'  [{label}]' if label else ''}"
    print(f"\n{'='*W}")
    print(hdr)
    print(f"{'='*W}")
    print(f"  노드: {n_nodes:,}개  |  엣지: {n_edges:,}개")
    print(f"  연결 컴포넌트: {len(comps)}개  |  고립 노드: {isolated}개 ({isolated/n_nodes*100:.1f}%)" if n_nodes else "")
    print(f"  평균 연결도: {avg_deg:.2f}  |  최대 연결도: {max_deg}")
    print()
    print("  [엔티티 타입 분포]")
    for etype, cnt in sorted(type_dist.items(), key=lambda x: -x[1]):
        bar = "#" * min(30, cnt)
        print(f"    {etype:<22} {cnt:>5}개  {bar}")
    print()
    print("  [연결도 상위 10 엔티티]")
    for nid, deg in top_nodes:
        attrs = G.nodes[nid]
        name  = attrs.get("entity_name", nid)[:35]
        etype = attrs.get("entity_type", "?")
        print(f"    {name:<36} [{etype:<12}] 연결={deg}")
    print(f"{'='*W}\n")


def print_isolated_detail(G: nx.Graph, limit: int = 50) -> None:
    isolated = [n for n, d in G.degree() if d == 0]
    print(f"\n  고립 노드 목록 (총 {len(isolated)}개, 최대 {limit}개 표시)")
    print(f"  {'노드 ID':<36} {'타입':<14} {'설명 길이':>8}  {'설명 미리보기'}")
    print(f"  {'-'*82}")
    for nid in isolated[:limit]:
        attrs   = G.nodes[nid]
        etype   = attrs.get("entity_type", "?")
        desc    = attrs.get("description", "")
        preview = desc[:50].replace("\n", " ")
        print(f"  {nid[:36]:<36} {etype:<14} {len(desc):>8}자  {preview}")
    if len(isolated) > limit:
        print(f"  ... 외 {len(isolated)-limit}개")
    print()


# ==============================================================================
# [3] 힐링 파이프라인 — 고립 노드 해소 4가지 전략
# ==============================================================================

# ── 공통 유틸 ───────────────────────────────
def _get_isolated(G: nx.Graph) -> list[str]:
    return [n for n, d in G.degree() if d == 0]


def _node_text(attrs: dict) -> str:
    name = attrs.get("entity_name", "")
    desc = attrs.get("description", "")
    return f"{name}: {desc}".strip(": ") or "unknown"


async def _batch_embed(texts: list[str], batch: int = 128) -> np.ndarray:
    parts = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        emb   = await _raw_embed(chunk)
        parts.append(emb)
        if len(texts) > batch:
            print(f"    임베딩 배치 {i//batch+1}/{(len(texts)-1)//batch+1}")
    return np.vstack(parts)


async def _direct_llm(system: str, user: str) -> str:
    """힐링 전용 직접 LLM 호출 (tracked_llm 우회, heal 단계 비용 집계)."""
    in_tok = _count(system) + _count(user)
    t = time.time()
    resp = await _oai_client.chat.completions.create(
        model=_COST_TABLE.get(LLM_MODEL, _COST_TABLE["mini"])["name"],
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        temperature=0.2,
        max_tokens=2048,
    )
    dur    = time.time() - t
    result = resp.choices[0].message.content or ""
    out_tok = _count(result)
    _usage["heal"]["llm_in"]  += in_tok
    _usage["heal"]["llm_out"] += out_tok
    return result


def _parse_json_array(raw: str) -> list[dict]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return []


# ── 전략 A: Prune ───────────────────────────
def _strategy_prune(
    G: nx.Graph,
    min_desc_len: int = HEAL_PRUNE_MIN_DESC,
    dry_run: bool = False,
) -> tuple[nx.Graph, int]:
    """
    고립 노드 중 품질 기준 미달 노드를 삭제합니다.
    삭제 기준 (OR):
      ① 설명이 완전히 비어있음
      ② entity_type == "UNKNOWN"  AND  설명 길이 < min_desc_len
    """
    isolated  = _get_isolated(G)
    to_remove = []
    for nid in isolated:
        attrs = G.nodes[nid]
        etype = attrs.get("entity_type", "UNKNOWN")
        desc  = (attrs.get("description") or "").strip()
        if not desc:
            to_remove.append((nid, "설명없음"))
        elif etype == "UNKNOWN" and len(desc) < min_desc_len:
            to_remove.append((nid, f"UNKNOWN+설명짧음({len(desc)}자)"))

    W = 64
    print(f"\n  {'─'*W}")
    print(f"  [전략 A: Prune]  고립 {len(isolated)}개 중 삭제 후보 {len(to_remove)}개")
    for nid, reason in to_remove[:15]:
        name = G.nodes[nid].get("entity_name", nid)
        print(f"    삭제 예정: {name[:42]:<42}  ({reason})")
    if len(to_remove) > 15:
        print(f"    ... 외 {len(to_remove)-15}개")

    if dry_run:
        print("  [dry_run] 저장하지 않습니다.")
        return G, 0

    G2 = copy.deepcopy(G)
    for nid, _ in to_remove:
        G2.remove_node(nid)
    print(f"  완료: {len(to_remove)}개 노드 삭제됨")
    return G2, len(to_remove)


# ── 전략 B: Embed ────────────────────────────
async def _strategy_embed(
    G: nx.Graph,
    threshold: float = HEAL_EMBED_THRESHOLD,
    top_k: int = HEAL_EMBED_TOP_K,
    dry_run: bool = False,
) -> tuple[nx.Graph, int]:
    """
    고립 노드와 연결된 노드 간 임베딩 코사인 유사도를 계산하여
    threshold 이상이면 엣지를 추가합니다.
    """
    isolated  = _get_isolated(G)
    connected = [n for n in G.nodes() if n not in set(isolated)]

    W = 64
    print(f"\n  {'─'*W}")
    if not isolated:
        print("  [전략 B: Embed]  고립 노드 없음 — 스킵")
        return G, 0
    if not connected:
        print("  [전략 B: Embed]  연결된 노드 없음 — 스킵")
        return G, 0

    print(f"  [전략 B: Embed]  고립 {len(isolated)}개 × 연결 {len(connected)}개")
    print(f"  임베딩 계산 중 (threshold={threshold}, top_k={top_k})...")

    iso_texts  = [_node_text(G.nodes[n]) for n in isolated]
    conn_texts = [_node_text(G.nodes[n]) for n in connected]

    _usage["heal"]["emb"] += sum(_count(t) for t in iso_texts + conn_texts)

    iso_embs  = await _batch_embed(iso_texts)
    conn_embs = await _batch_embed(conn_texts)

    G2 = copy.deepcopy(G)
    edge_log: list[tuple] = []

    for i, iso_id in enumerate(isolated):
        sims = [(connected[j], _cosine_sim(iso_embs[i], conn_embs[j]))
                for j in range(len(connected))]
        sims.sort(key=lambda x: -x[1])
        best = [(cid, s) for cid, s in sims if s >= threshold][:top_k]

        for conn_id, sim in best:
            iso_name  = G.nodes[iso_id].get("entity_name",  iso_id)
            conn_name = G.nodes[conn_id].get("entity_name", conn_id)
            edge_log.append((iso_id, conn_id, iso_name, conn_name, sim))
            if not dry_run:
                G2.add_edge(iso_id, conn_id,
                            relation_name="semantic_similarity",
                            keywords="semantic_similarity",
                            description=f"임베딩 유사도 기반 자동 연결 (cosine={sim:.4f})",
                            weight=round(float(sim), 4),
                            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    healed = len({iso_id for iso_id, _, _, _, _ in edge_log})
    print(f"  추가 엣지 {len(edge_log)}개 (고립 노드 {healed}개 해소)")
    for _, _, iso_n, conn_n, sim in edge_log[:15]:
        print(f"    {iso_n[:32]:<32} ─({sim:.3f})─> {conn_n[:32]}")
    if len(edge_log) > 15:
        print(f"    ... 외 {len(edge_log)-15}개")

    if dry_run:
        print("  [dry_run] 저장하지 않습니다.")
        return G, 0

    print(f"  완료: 엣지 {len(edge_log)}개 추가됨")
    return G2, len(edge_log)


# ── 전략 C: LLM ─────────────────────────────
_LLM_HEAL_SYSTEM = """\
당신은 의료·제약 지식그래프 전문가입니다.
[고립 노드] 목록의 각 엔티티와 [후보 노드] 목록 중 실제로 관계가 있을 법한 후보를 골라 JSON으로 반환하세요.

출력 형식 (JSON 배열):
[
  {
    "isolated_id": "고립 노드의 node_id",
    "candidate_id": "연결할 후보 노드의 node_id",
    "relation": "관계 종류 (한국어, 15자 이내)",
    "description": "관계 설명 (50자 이내)",
    "confidence": 0.0 ~ 1.0
  }
]

규칙:
- 관계가 불명확하면 그 고립 노드는 배열에서 제외하세요.
- confidence < 0.5 인 항목은 제외하세요.
- 반드시 JSON 배열만 출력하세요.
"""


async def _strategy_llm(
    G: nx.Graph,
    llm_limit: int = HEAL_LLM_LIMIT,
    min_confidence: float = HEAL_LLM_MIN_CONF,
    dry_run: bool = False,
) -> tuple[nx.Graph, int]:
    """
    LLM에게 고립 노드와 연결 가능한 노드를 제안받아 엣지를 삽입합니다.
    연결도 상위 노드를 후보로 제시합니다.
    """
    isolated  = _get_isolated(G)
    connected = [n for n in G.nodes() if n not in set(isolated)]

    W = 64
    print(f"\n  {'─'*W}")
    if not isolated:
        print("  [전략 C: LLM]  고립 노드 없음 — 스킵")
        return G, 0

    top_cands = sorted(connected, key=lambda n: G.degree(n), reverse=True)[:HEAL_LLM_BATCH_CANDS]
    target    = isolated[:llm_limit]
    CHUNK     = 10

    print(f"  [전략 C: LLM]  고립 {len(target)}개 처리 (배치={CHUNK}, 후보={len(top_cands)}개)")

    def _fmt(nid: str, attrs: dict) -> str:
        name  = attrs.get("entity_name",  nid)
        etype = attrs.get("entity_type",  "?")
        desc  = (attrs.get("description") or "")[:80].replace("\n", " ")
        return f"  id={nid}  name={name}  type={etype}  desc={desc}"

    G2 = copy.deepcopy(G)
    added = 0
    all_suggestions: list[dict] = []

    for start in range(0, len(target), CHUNK):
        chunk      = target[start:start+CHUNK]
        iso_block  = "\n".join(_fmt(n, G.nodes[n]) for n in chunk)
        cand_block = "\n".join(_fmt(n, G.nodes[n]) for n in top_cands)
        user       = f"[고립 노드]\n{iso_block}\n\n[후보 노드]\n{cand_block}"

        batch_no = start // CHUNK + 1
        total_b  = (len(target) - 1) // CHUNK + 1
        print(f"\n  배치 {batch_no}/{total_b}  ({len(chunk)}개 노드) — LLM 호출 중...")
        t0  = time.time()
        raw = await _direct_llm(_LLM_HEAL_SYSTEM, user)
        print(f"  LLM 응답 ({time.time()-t0:.1f}초)")

        suggestions = _parse_json_array(raw)
        valid = [
            s for s in suggestions
            if isinstance(s, dict)
            and s.get("confidence", 0) >= min_confidence
            and s.get("isolated_id")  in G.nodes
            and s.get("candidate_id") in G.nodes
        ]
        print(f"  제안 {len(suggestions)}개 → 채택 {len(valid)}개 (confidence≥{min_confidence})")

        for sug in valid:
            iso_n  = G.nodes[sug["isolated_id"]].get("entity_name",  sug["isolated_id"])
            cand_n = G.nodes[sug["candidate_id"]].get("entity_name", sug["candidate_id"])
            conf   = sug.get("confidence", 0)
            rel    = sug.get("relation", "관련")
            print(f"    [{conf:.2f}] {iso_n[:30]:<30} ─[{rel}]─> {cand_n[:30]}")

        all_suggestions.extend(valid)
        if not dry_run:
            for sug in valid:
                G2.add_edge(
                    sug["isolated_id"], sug["candidate_id"],
                    relation_name=sug.get("relation", "관련"),
                    keywords=sug.get("relation", "관련"),
                    description=sug.get("description", "LLM 제안 관계"),
                    weight=round(float(sug.get("confidence", 0.5)), 4),
                    created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                added += 1
        await asyncio.sleep(0.5)

    if dry_run:
        print(f"\n  [dry_run] 저장하지 않습니다. (제안 총 {len(all_suggestions)}개)")
        return G, 0

    healed = len({s["isolated_id"] for s in all_suggestions})
    print(f"\n  완료: 엣지 {added}개 추가 (고립 노드 {healed}개 해소)")
    return G2, added


# ── 전략 D: Re-link ──────────────────────────
_RELINK_SYSTEM = """\
당신은 지식그래프 전문가입니다.
아래 [텍스트]에서 [타겟 엔티티]와 관련이 있는 다른 엔티티들과의 관계를 추출하세요.

출력 형식 (JSON 배열):
[
  {
    "other_entity": "관련 엔티티 이름 (원문 그대로)",
    "relation": "관계 종류 (한국어, 15자 이내)",
    "description": "관계 설명 (50자 이내)",
    "confidence": 0.0 ~ 1.0
  }
]

규칙:
- confidence < 0.6 인 항목은 제외하세요.
- 반드시 JSON 배열만 출력하세요.
"""


async def _strategy_relink(
    G: nx.Graph,
    relink_limit: int = HEAL_RELINK_LIMIT,
    dry_run: bool = False,
) -> tuple[nx.Graph, int]:
    """
    고립 노드의 source_id 로 원본 청크를 찾아 LLM이 관계를 재추출합니다.
    KV 스토어 JSON 파일을 탐색하여 청크 텍스트를 로드합니다.
    """
    W = 64
    print(f"\n  {'─'*W}")

    # 청크 KV 스토어 탐색
    kv_files = [
        os.path.join(WORKING_DIR, f)
        for f in os.listdir(WORKING_DIR)
        if "chunk" in f.lower() and f.endswith(".json")
    ]
    if not kv_files:
        print("  [전략 D: Re-link]  청크 KV 스토어 없음 — 스킵")
        return G, 0

    chunk_map: dict[str, str] = {}
    for kp in kv_files:
        try:
            with open(kp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for cid, cval in data.items():
                    text = (cval.get("content") or cval.get("text") or ""
                            if isinstance(cval, dict) else str(cval))
                    if text:
                        chunk_map[cid] = text
        except Exception:
            pass

    if not chunk_map:
        print("  [전략 D: Re-link]  청크 텍스트 로드 실패 — 스킵")
        return G, 0

    isolated = _get_isolated(G)
    print(f"  [전략 D: Re-link]  고립 {len(isolated)}개 / 청크 {len(chunk_map)}개")

    SEP = "<SEP>"
    # 기존 노드 이름 → node_id 역색인
    name_to_id: dict[str, str] = {}
    for nid, attrs in G.nodes(data=True):
        name = (attrs.get("entity_name") or nid).strip().lower()
        name_to_id[name] = nid

    G2 = copy.deepcopy(G)
    added = 0

    for idx, nid in enumerate(isolated[:relink_limit]):
        attrs   = G.nodes[nid]
        name    = attrs.get("entity_name", nid)
        src_ids = [s.strip() for s in (attrs.get("source_id") or "").split(SEP)
                   if s.strip() in chunk_map]
        if not src_ids:
            continue

        chunk_text = "\n---\n".join(chunk_map[s] for s in src_ids[:3])[:3000]
        user = f"[타겟 엔티티]\n{name}\n\n[텍스트]\n{chunk_text}"

        print(f"  [{idx+1}/{min(len(isolated), relink_limit)}] {name[:40]:<40}", end=" ", flush=True)
        t0  = time.time()
        raw = await _direct_llm(_RELINK_SYSTEM, user)
        suggestions = _parse_json_array(raw)
        valid = [s for s in suggestions
                 if isinstance(s, dict) and s.get("confidence", 0) >= 0.6]
        print(f"제안 {len(valid)}개  ({time.time()-t0:.1f}초)")

        for sug in valid:
            other_name = (sug.get("other_entity") or "").strip().lower()
            target_id  = name_to_id.get(other_name)
            if not target_id or target_id == nid:
                continue
            rel  = sug.get("relation", "관련")
            desc = sug.get("description", "")
            conf = float(sug.get("confidence", 0.6))
            other_display = G.nodes.get(target_id, {}).get("entity_name", target_id)
            print(f"    + {name[:25]} ─[{rel}]─> {other_display[:25]}  (conf={conf:.2f})")
            if not dry_run:
                G2.add_edge(nid, target_id,
                            relation_name=rel, keywords=rel, description=desc,
                            weight=round(conf, 4),
                            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                added += 1

        await asyncio.sleep(0.3)

    if dry_run:
        print("  [dry_run] 저장하지 않습니다.")
        return G, 0

    print(f"\n  완료: 엣지 {added}개 추가됨")
    return G2, added


# ── 힐링 파이프라인 진입점 ───────────────────
async def run_heal(
    do_prune: bool = True,
    do_embed: bool = True,
    do_llm: bool = True,
    do_relink: bool = False,
    dry_run: bool = False,
    embed_threshold: float = HEAL_EMBED_THRESHOLD,
    embed_top_k: int = HEAL_EMBED_TOP_K,
    llm_limit: int = HEAL_LLM_LIMIT,
    llm_min_confidence: float = HEAL_LLM_MIN_CONF,
    relink_limit: int = HEAL_RELINK_LIMIT,
    prune_min_desc: int = HEAL_PRUNE_MIN_DESC,
    isolated_detail: bool = False,
) -> None:
    global _phase
    _phase = "heal"

    try:
        gpath = _find_graphml()
    except FileNotFoundError as e:
        print(f"\n  오류: {e}")
        return

    G = nx.read_graphml(gpath)
    print(f"\n  GraphML: {os.path.basename(gpath)}")
    print_graph_stats(G, "힐링 전")

    if isolated_detail:
        print_isolated_detail(G)

    n_isolated_before = sum(1 for _, d in G.degree() if d == 0)
    if n_isolated_before == 0:
        print("  고립 노드가 없습니다. 힐링을 건너뜁니다.")
        return

    # 백업
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = gpath + f".bak_{ts}"
    shutil.copy2(gpath, backup_path)
    print(f"  백업: {os.path.basename(backup_path)}\n")

    total_changed = 0

    if do_prune:
        G, n = _strategy_prune(G, min_desc_len=prune_min_desc, dry_run=dry_run)
        total_changed += n

    if do_embed:
        G, n = await _strategy_embed(G, threshold=embed_threshold, top_k=embed_top_k, dry_run=dry_run)
        total_changed += n

    if do_llm:
        G, n = await _strategy_llm(G, llm_limit=llm_limit, min_confidence=llm_min_confidence, dry_run=dry_run)
        total_changed += n

    if do_relink:
        G, n = await _strategy_relink(G, relink_limit=relink_limit, dry_run=dry_run)
        total_changed += n

    # 저장 & 결과 리포트
    if not dry_run and total_changed > 0:
        nx.write_graphml(G, gpath)
        print(f"\n  그래프 저장 완료: {os.path.basename(gpath)}")
        print_graph_stats(G, "힐링 후")
        n_isolated_after = sum(1 for _, d in G.degree() if d == 0)
        resolved = n_isolated_before - n_isolated_after
        print(f"  [힐링 요약] 고립 노드: {n_isolated_before} → {n_isolated_after}  "
              f"(해소 {resolved}개, {resolved/n_isolated_before*100:.1f}%)" if n_isolated_before else "")
    elif dry_run:
        print("\n  [dry_run] 파일 저장 없음")
    else:
        print("\n  변경 사항 없음")


def print_heal_cost() -> None:
    h = _usage["heal"]
    c = _cost(h["llm_in"], h["llm_out"], h["emb"])
    if c == 0:
        return
    rates = _llm_cost_rates()
    print(f"\n  [힐링 비용]")
    print(f"    LLM 입력  : {h['llm_in']:>8,}tok  ${h['llm_in']/1000*rates['in']:.5f}")
    print(f"    LLM 출력  : {h['llm_out']:>8,}tok  ${h['llm_out']/1000*rates['out']:.5f}")
    print(f"    임베딩    : {h['emb']:>8,}tok  ${h['emb']/1000*_EMB_COST:.5f}")
    print(f"    소계      :             ${c:.5f}  (= ₩{c*1380:,.0f})")


# ==============================================================================
# [4] 그래프 시각화
# ==============================================================================
def visualize_graph(graphml_path: str = None, output_html: str = None, max_nodes: int = 1000) -> None:
    if output_html is None:
        output_html = os.path.join(WORKING_DIR, "knowledge_graph.html")

    if graphml_path is None:
        try:
            graphml_path = _find_graphml()
        except FileNotFoundError:
            print("  GraphML 파일 없음 — 시각화 스킵")
            return

    print(f"  [시각화] {os.path.basename(graphml_path)}")
    G = nx.read_graphml(graphml_path)
    print(f"  노드: {G.number_of_nodes():,}  엣지: {G.number_of_edges():,}")

    if G.number_of_nodes() > max_nodes:
        print(f"  노드 {max_nodes}개 초과 → 연결도 상위 {max_nodes}개만 표시")
        top = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        G   = G.subgraph([n for n, _ in top]).copy()

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

    def get_color(etype: str) -> str:
        if etype not in type_colors:
            type_colors[etype] = palette[len(type_colors) % len(palette)]
        return type_colors[etype]

    for nid, attrs in G.nodes(data=True):
        label  = attrs.get("entity_name", attrs.get("id", str(nid)))
        etype  = attrs.get("entity_type", "UNKNOWN")
        desc   = attrs.get("description", "")
        color  = get_color(etype)
        degree = G.degree(nid)
        net.add_node(str(nid), label=str(label)[:40],
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

    print(f"  시각화 완료 → {os.path.abspath(output_html)}")


# ==============================================================================
# [5] 쿼리 파이프라인
# ==============================================================================
DEFAULT_QUERIES = [
    "기넥신 누구한테 영업할까?",
    "외과에 어떤 약을 추천할까",
    "류마티스 관절염에 도움이 되는 약",
    "기넥신을 누구한테 쓰면 안돼?",
]
MODES = ["naive", "local", "global", "hybrid"]


def _extract_sources(call_log: list) -> list[str]:
    sources = []
    for entry in call_log:
        if entry.get("type") != "llm":
            continue
        prompt = entry.get("prompt_preview", "")
        names  = re.findall(r'entity[_\s]*name["\s:]*([^"\n,]+)', prompt, re.IGNORECASE)
        names += re.findall(r'"([A-Z][A-Za-z0-9\s\-]{2,30})"', prompt)
        for n in names:
            n = n.strip()
            if n and n not in sources and len(n) > 1:
                sources.append(n)
    return sources[:15]


async def _embed_query(query: str) -> list:
    result = await _raw_embed([query])
    return result[0].tolist()


async def _run_one_query(
    rag: LightRAG, query: str, mode: str = "hybrid", silent: bool = False
) -> dict:
    global _call_log, _query_cache
    _call_log = []
    u_before  = {k: _usage["query"][k] for k in _usage["query"]}

    if not silent:
        print(f"\n{'='*60}")
        print(f"[쿼리] {query}  (mode={mode})")
        print("="*60)

    # 1) 쿼리 임베딩
    t0        = time.time()
    query_emb = await _embed_query(query)
    emb_sec   = time.time() - t0
    emb_tok   = _count(query)
    _usage["query"]["emb"] += emb_tok
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

    # 3) 실행
    t_total   = time.time()
    result    = await rag.aquery(query, param=QueryParam(mode=mode, enable_rerank=False))
    total_sec = time.time() - t_total

    if not silent:
        print(result)

    emb_calls     = [c for c in _call_log if c["type"] == "emb"]
    llm_calls     = [c for c in _call_log if c["type"] == "llm"]
    retrieval_llm = llm_calls[:-1] if len(llm_calls) > 1 else []
    gen_llm       = llm_calls[-1]  if llm_calls else None

    t_emb     = sum(c["sec"] for c in emb_calls)
    t_ret_llm = sum(c["sec"] for c in retrieval_llm)
    t_gen     = gen_llm["sec"] if gen_llm else 0
    t_other   = total_sec - t_emb - t_ret_llm - t_gen

    q_in   = _usage["query"]["llm_in"]  - u_before["llm_in"]
    q_out  = _usage["query"]["llm_out"] - u_before["llm_out"]
    q_emb  = _usage["query"]["emb"]     - u_before["emb"]
    q_cost = _cost(q_in, q_out, q_emb)

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
            print(f"  [캐시] 저장 ({_query_cache.summary()})")

    return {
        "query": query, "mode": mode, "answer": result,
        "cache_hit": False, "sec": total_sec,
        "llm_in": q_in, "llm_out": q_out, "emb_tok": q_emb,
        "cost": q_cost, "sources": sources,
    }


async def run_queries(rag: LightRAG) -> None:
    global _phase
    _phase = "query"
    for q in DEFAULT_QUERIES:
        await _run_one_query(rag, q)


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
        print(f"  {mode:<8} | {res['sec']:>6.2f}초 | in={res['llm_in']:,} out={res['llm_out']:,} | ${res['cost']:.5f}")

    print(f"\n  {'─'*66}")
    print(f"  {'모드':<8} | {'시간':>6} | {'LLM입력':>8} | {'LLM출력':>8} | {'비용':>9} | {'답변길이':>6}")
    print(f"  {'─'*66}")
    for r in results:
        print(f"  {r['mode']:<8} | {r['sec']:>5.2f}초 | {r['llm_in']:>8,} | "
              f"{r['llm_out']:>8,} | ${r['cost']:>8.5f} | {len(r['answer']):>5,}자")
    print(f"  {'─'*66}\n")
    print("  [답변 미리보기]")
    for r in results:
        print(f"  [{r['mode']}] {r['answer'].replace(chr(10), ' ')}")
    print()


async def run_batch_queries(rag: LightRAG, batch_file: str) -> None:
    global _phase
    _phase = "query"

    with open(batch_file, "r", encoding="utf-8") as f:
        queries = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    if not queries:
        print(f"  질문 없음: {batch_file}")
        return

    print(f"\n  [배치 쿼리] {len(queries)}개 질문 로드")
    results = []
    for i, q in enumerate(queries, 1):
        print(f"\n  [{i}/{len(queries)}] {q}")
        res = await _run_one_query(rag, q, silent=True)
        results.append(res)
        info = "캐시히트" if res["cache_hit"] else f"LLM in={res['llm_in']:,}"
        print(f"    → {res['sec']:.2f}초 | ${res['cost']:.5f} | {info}")

    out_path = os.path.join(WORKING_DIR, "batch_result.md")
    now      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines    = [
        "# 배치 쿼리 결과",
        f"\n생성: {now}  |  모델: {_COST_TABLE[LLM_MODEL]['name']}  |  질문: {len(results)}개",
        "",
        "| # | 질문 | 시간 | LLM입력 | LLM출력 | 비용 | 캐시 |",
        "|---|------|------|---------|---------|------|------|",
    ]
    total_cost = total_sec = 0
    for i, r in enumerate(results, 1):
        total_cost += r["cost"]
        total_sec  += r["sec"]
        cache_str   = f"HIT({r.get('similarity',0):.2f})" if r["cache_hit"] else "-"
        lines.append(f"| {i} | {r['query'][:40]} | {r['sec']:.2f}초 | "
                     f"{r['llm_in']:,} | {r['llm_out']:,} | ${r['cost']:.5f} | {cache_str} |")
    lines.append(f"| | **합계** | **{total_sec:.2f}초** | | | **${total_cost:.5f}** | |")
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


# ==============================================================================
# 전체 비용 최종 요약
# ==============================================================================
def print_total_cost(total_elapsed: float) -> None:
    ins = _usage["insert"]
    qry = _usage["query"]
    heal= _usage["heal"]
    ic  = _cost(ins["llm_in"],  ins["llm_out"],  ins["emb"])
    qc  = _cost(qry["llm_in"],  qry["llm_out"],  qry["emb"])
    hc  = _cost(heal["llm_in"], heal["llm_out"], heal["emb"])
    tot = ic + qc + hc

    rates = _llm_cost_rates()
    print(f"\n{'='*54}")
    print("[전체 크레딧 최종 요약]")
    print(f"  모델: {_COST_TABLE[LLM_MODEL]['name']}")
    print(f"  삽입(관계 형성) : ${ic:.5f}  (= ₩{ic*1380:,.0f})")
    if hc > 0:
        print(f"  힐링(고립 해소) : ${hc:.5f}  (= ₩{hc*1380:,.0f})")
    print(f"  쿼리            : ${qc:.5f}  (= ₩{qc*1380:,.0f})")
    print(f"  {'─'*40}")
    print(f"  합계            : ${tot:.5f}  (= ₩{tot*1380:,.0f})")
    print(f"  총 소요 시간    : {total_elapsed:.1f}초")
    print(f"{'='*54}\n")


# ==============================================================================
# CLI & 메인
# ==============================================================================
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="total_process.py",
        description="LightRAG 통합 파이프라인 (삽입 → 힐링 → 시각화 → 통계 → 쿼리)",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── 공통 ─────────────────────────────
    p.add_argument("--llm", choices=["mini", "4o"], default=LLM_MODEL,
                   help="LLM 모델: mini=gpt-4o-mini, 4o=gpt-4o (기본: mini)")
    p.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                   help=f"청크 최대 토큰 수 (기본: {CHUNK_SIZE})")
    p.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP,
                   help=f"청크 오버랩 토큰 수 (기본: {CHUNK_OVERLAP})")

    # ── 삽입 제어 ─────────────────────────
    p.add_argument("--skip-insert", action="store_true",
                   help="삽입 건너뜀 (시각화·통계·쿼리만 실행)")

    # ── 힐링 ──────────────────────────────
    p.add_argument("--heal", action="store_true",
                   help="힐링 파이프라인 실행 (기본: 전략 A+B+C)")
    p.add_argument("--heal-all", action="store_true",
                   help="힐링 전략 A+B+C+D 모두 실행")
    p.add_argument("--heal-prune", action="store_true",
                   help="힐링 — 전략 A(Prune)만 포함")
    p.add_argument("--heal-embed", action="store_true",
                   help="힐링 — 전략 B(Embed)만 포함")
    p.add_argument("--heal-llm", action="store_true",
                   help="힐링 — 전략 C(LLM)만 포함")
    p.add_argument("--heal-relink", action="store_true",
                   help="힐링 — 전략 D(Re-link)만 포함")
    p.add_argument("--dry-run", action="store_true",
                   help="힐링 변경 내용 미리보기만 (저장 안 함)")
    p.add_argument("--prune-min-desc", type=int, default=HEAL_PRUNE_MIN_DESC,
                   help=f"전략 A: 삭제 기준 최소 설명 길이 (기본: {HEAL_PRUNE_MIN_DESC}자)")
    p.add_argument("--embed-threshold", type=float, default=HEAL_EMBED_THRESHOLD,
                   help=f"전략 B: 연결 최소 코사인 유사도 (기본: {HEAL_EMBED_THRESHOLD})")
    p.add_argument("--embed-top-k", type=int, default=HEAL_EMBED_TOP_K,
                   help=f"전략 B: 노드당 최대 연결 수 (기본: {HEAL_EMBED_TOP_K})")
    p.add_argument("--llm-limit", type=int, default=HEAL_LLM_LIMIT,
                   help=f"전략 C: LLM 처리 최대 고립 노드 수 (기본: {HEAL_LLM_LIMIT})")
    p.add_argument("--llm-min-confidence", type=float, default=HEAL_LLM_MIN_CONF,
                   help=f"전략 C: 채택 최소 confidence (기본: {HEAL_LLM_MIN_CONF})")
    p.add_argument("--relink-limit", type=int, default=HEAL_RELINK_LIMIT,
                   help=f"전략 D: Re-link 최대 노드 수 (기본: {HEAL_RELINK_LIMIT})")
    p.add_argument("--isolated-detail", action="store_true",
                   help="힐링 전 고립 노드 상세 목록 출력")

    # ── 통계 ──────────────────────────────
    p.add_argument("--stats", action="store_true",
                   help="그래프 통계만 출력하고 종료")

    # ── 시각화 ────────────────────────────
    p.add_argument("--visualize", action="store_true",
                   help="시각화만 실행하고 종료")
    p.add_argument("--max-nodes", type=int, default=1000,
                   help="시각화 최대 노드 수 (기본: 1000)")

    # ── 쿼리 ──────────────────────────────
    p.add_argument("-q", "--query", nargs="+", metavar="WORD",
                   help="단건 쿼리 (삽입·힐링·시각화 생략)")
    p.add_argument("--mode", choices=MODES, default="hybrid",
                   help=f"쿼리 모드 (기본: hybrid)")
    p.add_argument("--mode-compare", nargs="+", metavar="WORD",
                   help="4종 모드(naive/local/global/hybrid) 동시 비교")
    p.add_argument("--batch", metavar="FILE",
                   help="파일에서 질문 읽어 일괄 실행 → batch_result.md 저장")

    # ── 쿼리 캐시 ─────────────────────────
    p.add_argument("--no-cache", action="store_true",
                   help="쿼리 캐시 비활성화")
    p.add_argument("--cache-threshold", type=float, default=CACHE_SIMILARITY_THRESHOLD,
                   help=f"캐시 유사도 임계값 (기본: {CACHE_SIMILARITY_THRESHOLD})")
    p.add_argument("--show-cache", action="store_true",
                   help="캐시 내용 출력 후 종료")

    return p


async def main() -> None:
    global _phase, _query_cache, CHUNK_SIZE, CHUNK_OVERLAP, LLM_MODEL, CACHE_SIMILARITY_THRESHOLD

    parser = _build_parser()
    args   = parser.parse_args()

    # 런타임 설정 반영
    CHUNK_SIZE    = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    LLM_MODEL     = args.llm
    if args.cache_threshold != CACHE_SIMILARITY_THRESHOLD:
        CACHE_SIMILARITY_THRESHOLD = args.cache_threshold

    rates = _COST_TABLE[LLM_MODEL]
    print(f"  [모델] {rates['name']}  "
          f"(in=${rates['in']*1000:.3f}/1M  out=${rates['out']*1000:.3f}/1M)")

    # 캐시 초기화
    cache_path = os.path.join(WORKING_DIR, "query_cache.json")
    if not args.no_cache:
        _query_cache = QueryCache(cache_path)
    else:
        _query_cache = None
        print("  [캐시] 비활성화")

    t_start = time.time()

    # ── 즉시 종료 단축 경로 ──────────────────────────────────────

    if args.show_cache:
        if _query_cache and _query_cache.data:
            print(f"\n{'='*60}  쿼리 캐시 ({len(_query_cache.data)}건)  {'='*60}")
            for cid, entry in _query_cache.data.items():
                print(f"  [{cid}] {entry['query']}")
                print(f"    {entry['created_at']} | ${entry['cost']:.5f} | {entry['sec']}초")
                print(f"    {entry['answer'][:120]}...")
        else:
            print("  캐시가 비어 있습니다.")
        return

    if args.stats:
        print_graph_stats()
        return

    if args.visualize:
        visualize_graph(max_nodes=args.max_nodes)
        return

    # ── LightRAG 인스턴스 (쿼리·힐링에만 필요) ───────────────────
    need_rag = bool(args.query or args.mode_compare or args.batch
                   or not (args.heal or args.heal_all or args.heal_prune
                            or args.heal_embed or args.heal_llm or args.heal_relink))

    # ── --heal / --heal-all 단독 ─────────────────────────────────
    heal_requested = (args.heal or args.heal_all or args.heal_prune
                      or args.heal_embed or args.heal_llm or args.heal_relink)

    if heal_requested and not need_rag:
        # 힐링 전략 결정
        if args.heal_all:
            do_prune, do_embed, do_llm, do_relink = True, True, True, True
        elif any([args.heal_prune, args.heal_embed, args.heal_llm, args.heal_relink]):
            do_prune   = args.heal_prune
            do_embed   = args.heal_embed
            do_llm     = args.heal_llm
            do_relink  = args.heal_relink
        else:  # --heal → 기본 A+B+C
            do_prune, do_embed, do_llm, do_relink = True, True, True, False

        await run_heal(
            do_prune=do_prune, do_embed=do_embed,
            do_llm=do_llm,     do_relink=do_relink,
            dry_run=args.dry_run,
            embed_threshold=args.embed_threshold,
            embed_top_k=args.embed_top_k,
            llm_limit=args.llm_limit,
            llm_min_confidence=args.llm_min_confidence,
            relink_limit=args.relink_limit,
            prune_min_desc=args.prune_min_desc,
            isolated_detail=args.isolated_detail,
        )
        print_heal_cost()
        print_total_cost(time.time() - t_start)
        return

    # ── RAG 인스턴스 생성 ────────────────────────────────────────
    rag = _build_rag()
    await rag.initialize_storages()

    # ── 쿼리 단독 모드들 ─────────────────────────────────────────
    if args.mode_compare:
        query_text = " ".join(args.mode_compare)
        await run_mode_compare(rag, query_text)
        print_total_cost(time.time() - t_start)
        return

    if args.batch:
        await run_batch_queries(rag, args.batch)
        print_total_cost(time.time() - t_start)
        return

    if args.query:
        _phase = "query"
        query_text = " ".join(args.query)
        print(f"  [쿼리 모드] {WORKING_DIR}")
        await _run_one_query(rag, query_text, mode=args.mode)
        print_total_cost(time.time() - t_start)
        return

    # ════════════════════════════════════════════════════════════
    # 전체 파이프라인: 삽입 → 힐링 → 시각화 → 통계 → 기본 쿼리
    # ════════════════════════════════════════════════════════════
    print("\ntotal_process.py — 통합 파이프라인 시작\n")

    # 1. 삽입
    if not args.skip_insert:
        _phase = "insert"
        t1 = time.time()
        await insert_documents(rag)
        print_insert_summary(time.time() - t1)
    else:
        print("  [삽입 건너뜀]\n")

    # 2. 힐링 (고립 노드 해소)
    #    전체 파이프라인에서는 기본 전략 A+B+C 자동 실행
    #    --dry-run 이면 미리보기만
    heal_strategies = {
        "do_prune":  True,
        "do_embed":  True,
        "do_llm":    True,
        "do_relink": args.heal_all,
    }
    if heal_requested:   # 명시적 힐링 옵션이 있으면 그것을 따름
        if args.heal_all:
            heal_strategies = {"do_prune": True, "do_embed": True, "do_llm": True, "do_relink": True}
        elif any([args.heal_prune, args.heal_embed, args.heal_llm, args.heal_relink]):
            heal_strategies = {
                "do_prune":  args.heal_prune,
                "do_embed":  args.heal_embed,
                "do_llm":    args.heal_llm,
                "do_relink": args.heal_relink,
            }

    await run_heal(
        **heal_strategies,
        dry_run=args.dry_run,
        embed_threshold=args.embed_threshold,
        embed_top_k=args.embed_top_k,
        llm_limit=args.llm_limit,
        llm_min_confidence=args.llm_min_confidence,
        relink_limit=args.relink_limit,
        prune_min_desc=args.prune_min_desc,
        isolated_detail=args.isolated_detail,
    )
    print_heal_cost()

    # 3. 시각화
    print()
    t2 = time.time()
    visualize_graph(max_nodes=args.max_nodes)
    print(f"  [시간] 시각화: {time.time()-t2:.1f}초\n")

    # 4. 통계
    print_graph_stats(label="최종")

    # 5. 기본 쿼리
    t3 = time.time()
    await run_queries(rag)
    print(f"\n  [시간] 쿼리 합계: {time.time()-t3:.1f}초")

    if _query_cache:
        print(f"\n  [쿼리 캐시] {_query_cache.summary()}")
        print(f"  저장 위치: {cache_path}")

    print_total_cost(time.time() - t_start)


if __name__ == "__main__":
    asyncio.run(main())
