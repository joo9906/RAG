"""
entity_solution.py — 고립 노드(isolated node) 해소 유틸리티

그래프 내 엣지가 없는 노드(degree=0)를 세 가지 전략으로 처리합니다.

  전략 A  prune    — 품질 기준 미달 노드 삭제
                      (entity_type=UNKNOWN 이거나 설명이 너무 짧은 노드)
  전략 B  embed    — 임베딩 코사인 유사도로 가장 가까운 노드와 연결
                      (API 호출 없이도 사용 가능하지만 임베딩 계산은 필요)
  전략 C  llm      — LLM이 고립 노드와 연관 있는 노드를 제안, 관계 엣지 삽입

사용 예)
  # 1) 현황 분석만 (그래프 변경 없음)
  python entity_solution.py --stats

  # 2) 품질 미달 노드 삭제
  python entity_solution.py --prune

  # 3) 임베딩 유사도로 연결 (기본 임계값 0.75)
  python entity_solution.py --embed --embed-threshold 0.78

  # 4) LLM이 관계 제안 (최대 30개 노드 처리)
  python entity_solution.py --llm --llm-limit 30

  # 5) 전략 조합 (순서대로 실행: prune → embed → llm)
  python entity_solution.py --prune --embed --llm

  # 6) 특정 WORKING_DIR 지정 (process.py CONFIG와 다를 때)
  python entity_solution.py --stats --working-dir ./lightrag_data

모든 전략 실행 전에 .graphml 파일이 자동 백업됩니다.
"""

import os
import sys
import json
import copy
import argparse
import asyncio
import time
import shutil
import re
from datetime import datetime
from typing import Optional

import numpy as np
import networkx as nx
from openai import AsyncOpenAI

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ==============================================================================
# CONFIG — process.py 와 동일한 값으로 맞춰두세요
# ==============================================================================
ENV_JSON_PATH  = "../.env.json"
_BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR    = os.path.join(_BASE_DIR, "lightrag_before_chunk_test")  # process.py 와 동일
LLM_MODEL_NAME = "gpt-4o-mini"          # "gpt-4o-mini" or "gpt-4o"
EMB_MODEL      = "text-embedding-3-large"
EMB_DIM        = 2048                   # large 모델 축소 차원 (process.py 와 동일)

# 전략 B: 임베딩 유사도 임계값 기본값
DEFAULT_EMBED_THRESHOLD = 0.75
# 전략 B: 고립 노드 하나당 연결할 최대 후보 수
DEFAULT_TOP_K = 2
# 전략 A: 설명 최소 길이 (이보다 짧으면 삭제 대상)
MIN_DESC_LEN = 10
# 전략 C: 배치 크기 (한 번의 LLM 호출에 담을 후보 노드 수)
LLM_BATCH_CANDIDATES = 20
# ==============================================================================

_oai: Optional[AsyncOpenAI] = None


def _load_api_key() -> str:
    env_path = os.path.join(_BASE_DIR, ENV_JSON_PATH)
    if not os.path.exists(env_path):
        raise FileNotFoundError(f".env.json 없음: {env_path}")
    with open(env_path, "r", encoding="utf-8") as f:
        return json.load(f)["openai_api_key"]


def _get_client() -> AsyncOpenAI:
    global _oai
    if _oai is None:
        _oai = AsyncOpenAI(api_key=_load_api_key())
    return _oai


# ─────────────────────────────────────────────
# GraphML 유틸
# ─────────────────────────────────────────────

def _find_graphml(working_dir: str) -> str:
    """working_dir 안에서 가장 최근 수정된 .graphml 파일 경로 반환."""
    files = [
        os.path.join(working_dir, f)
        for f in os.listdir(working_dir)
        if f.endswith(".graphml")
    ]
    if not files:
        raise FileNotFoundError(f"GraphML 파일 없음: {working_dir}")
    return max(files, key=os.path.getmtime)


def _load_graph(working_dir: str) -> tuple[nx.Graph, str]:
    path = _find_graphml(working_dir)
    G = nx.read_graphml(path)
    return G, path


def _backup(path: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path + f".bak_{ts}"
    shutil.copy2(path, backup_path)
    return backup_path


def _save_graph(G: nx.Graph, path: str) -> None:
    nx.write_graphml(G, path)


# ─────────────────────────────────────────────
# 기본 분석
# ─────────────────────────────────────────────

def get_isolated_nodes(G: nx.Graph) -> list[str]:
    """degree=0 인 노드 id 목록 반환."""
    return [n for n, d in G.degree() if d == 0]


def print_stats(G: nx.Graph, label: str = "") -> None:
    n_nodes  = G.number_of_nodes()
    n_edges  = G.number_of_edges()
    degrees  = [d for _, d in G.degree()]
    avg_deg  = sum(degrees) / len(degrees) if degrees else 0.0
    max_deg  = max(degrees) if degrees else 0
    isolated = sum(1 for d in degrees if d == 0)

    if G.is_directed():
        comps = list(nx.weakly_connected_components(G))
    else:
        comps = list(nx.connected_components(G))

    type_dist: dict[str, int] = {}
    for _, attrs in G.nodes(data=True):
        etype = attrs.get("entity_type", "UNKNOWN")
        type_dist[etype] = type_dist.get(etype, 0) + 1

    W = 62
    hdr = f"  그래프 통계{f'  [{label}]' if label else ''}"
    print(f"\n{'='*W}")
    print(hdr)
    print(f"{'='*W}")
    print(f"  노드: {n_nodes:,}개  |  엣지: {n_edges:,}개")
    print(f"  연결 컴포넌트: {len(comps)}개  |  고립 노드: {isolated}개")
    print(f"  평균 연결도: {avg_deg:.2f}  |  최대 연결도: {max_deg}")
    print()
    print("  [엔티티 타입 분포]")
    for etype, cnt in sorted(type_dist.items(), key=lambda x: -x[1]):
        bar = "#" * min(30, cnt)
        print(f"    {etype:<22} {cnt:>5}개  {bar}")
    print(f"{'='*W}\n")


def print_isolated_detail(G: nx.Graph, limit: int = 30) -> None:
    isolated = get_isolated_nodes(G)
    print(f"\n  고립 노드 목록 (총 {len(isolated)}개, 최대 {limit}개 표시)")
    print(f"  {'노드 ID':<36} {'타입':<14} {'설명 길이':>8}  {'설명 미리보기'}")
    print(f"  {'-'*80}")
    for nid in isolated[:limit]:
        attrs = G.nodes[nid]
        etype = attrs.get("entity_type", "?")
        desc  = attrs.get("description", "")
        dlen  = len(desc)
        preview = desc[:45].replace("\n", " ")
        print(f"  {nid[:36]:<36} {etype:<14} {dlen:>8}자  {preview}")
    if len(isolated) > limit:
        print(f"  ... 외 {len(isolated) - limit}개")
    print()


# ─────────────────────────────────────────────
# 전략 A: Prune (품질 미달 노드 삭제)
# ─────────────────────────────────────────────

def strategy_prune(
    G: nx.Graph,
    min_desc_len: int = MIN_DESC_LEN,
    remove_unknown: bool = True,
    dry_run: bool = False,
) -> tuple[nx.Graph, int]:
    """
    고립 노드 중 품질 기준 미달인 노드를 삭제합니다.
    삭제 기준 (OR 조건):
      1) entity_type == "UNKNOWN"  AND  설명 길이 < min_desc_len
      2) 설명이 아예 없음 (빈 문자열)

    dry_run=True 이면 삭제 목록만 출력하고 실제로는 삭제하지 않습니다.
    반환: (수정된 그래프, 삭제 노드 수)
    """
    isolated  = get_isolated_nodes(G)
    to_remove = []

    for nid in isolated:
        attrs = G.nodes[nid]
        etype = attrs.get("entity_type", "UNKNOWN")
        desc  = (attrs.get("description") or "").strip()

        if not desc:                                      # 설명 없음
            to_remove.append((nid, "설명없음"))
        elif remove_unknown and etype == "UNKNOWN" and len(desc) < min_desc_len:
            to_remove.append((nid, f"UNKNOWN+설명짧음({len(desc)}자)"))

    print(f"\n  [전략 A: Prune]  삭제 후보: {len(to_remove)}/{len(isolated)}개")
    for nid, reason in to_remove[:20]:
        attrs = G.nodes[nid]
        name  = attrs.get("entity_name", nid)
        print(f"    삭제: {name[:40]:<40}  ({reason})")
    if len(to_remove) > 20:
        print(f"    ... 외 {len(to_remove)-20}개")

    if dry_run:
        print("  [dry_run] 실제 삭제는 수행하지 않습니다.")
        return G, 0

    G2 = copy.deepcopy(G)
    for nid, _ in to_remove:
        G2.remove_node(nid)
    print(f"  삭제 완료: {len(to_remove)}개 노드 제거")
    return G2, len(to_remove)


# ─────────────────────────────────────────────
# 임베딩 헬퍼
# ─────────────────────────────────────────────

async def _embed_texts(texts: list[str]) -> np.ndarray:
    """OpenAI 임베딩 API 호출. shape=(N, EMB_DIM)."""
    client  = _get_client()
    kwargs  = {"model": EMB_MODEL, "input": texts}
    if "large" in EMB_MODEL:
        kwargs["dimensions"] = EMB_DIM
    resp = await client.embeddings.create(**kwargs)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _node_text(attrs: dict) -> str:
    """노드의 대표 텍스트 (이름 + 설명)."""
    name = attrs.get("entity_name", "")
    desc = attrs.get("description", "")
    return f"{name}: {desc}".strip(": ") or "unknown"


# ─────────────────────────────────────────────
# 전략 B: Embed (임베딩 유사도로 연결)
# ─────────────────────────────────────────────

async def strategy_embed(
    G: nx.Graph,
    threshold: float = DEFAULT_EMBED_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    dry_run: bool = False,
) -> tuple[nx.Graph, int]:
    """
    고립 노드와 기존 연결 노드 간의 임베딩 코사인 유사도를 계산하여
    임계값 이상이면 엣지를 추가합니다.

    threshold : 연결할 최소 코사인 유사도 (기본 0.75)
    top_k     : 고립 노드 하나당 연결할 최대 후보 수
    """
    isolated    = get_isolated_nodes(G)
    connected   = [n for n in G.nodes() if n not in set(isolated)]

    if not isolated:
        print("\n  [전략 B: Embed]  고립 노드 없음 — 스킵")
        return G, 0

    if not connected:
        print("\n  [전략 B: Embed]  연결된 노드가 없습니다 — 스킵")
        return G, 0

    print(f"\n  [전략 B: Embed]  고립 노드 {len(isolated)}개 × 연결 노드 {len(connected)}개")
    print(f"  임베딩 계산 중 (model={EMB_MODEL}, dim={EMB_DIM})...")

    # 배치 임베딩
    BATCH = 128
    iso_texts  = [_node_text(G.nodes[n]) for n in isolated]
    conn_texts = [_node_text(G.nodes[n]) for n in connected]

    async def batch_embed(texts):
        results = []
        for i in range(0, len(texts), BATCH):
            chunk = texts[i:i+BATCH]
            emb   = await _embed_texts(chunk)
            results.append(emb)
            if len(texts) > BATCH:
                print(f"    임베딩 배치 {i//BATCH+1}/{(len(texts)-1)//BATCH+1} 완료")
        return np.vstack(results)

    iso_embs  = await batch_embed(iso_texts)
    conn_embs = await batch_embed(conn_texts)

    G2        = copy.deepcopy(G)
    added     = 0
    edge_log  = []

    for i, iso_id in enumerate(isolated):
        sims = [(conn_id, _cosine_sim(iso_embs[i], conn_embs[j]))
                for j, conn_id in enumerate(connected)]
        sims.sort(key=lambda x: -x[1])
        best = [(cid, s) for cid, s in sims if s >= threshold][:top_k]

        for conn_id, sim in best:
            iso_name  = G.nodes[iso_id].get("entity_name",  iso_id)
            conn_name = G.nodes[conn_id].get("entity_name", conn_id)
            edge_log.append((iso_name, conn_name, sim))
            if not dry_run:
                G2.add_edge(iso_id, conn_id,
                            relation_name="semantic_similarity",
                            keywords="semantic_similarity",
                            description=f"임베딩 유사도 기반 연결 (cosine={sim:.4f})",
                            weight=round(float(sim), 4),
                            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            added += 1

    # 보고
    print(f"\n  추가될 엣지 {len(edge_log)}개 (고립 노드 {sum(1 for _,_,s in edge_log)//max(top_k,1) if edge_log else 0}개 해소 예상):")
    for iso_n, conn_n, sim in edge_log[:20]:
        print(f"    {iso_n[:32]:<32} ─({sim:.3f})─> {conn_n[:32]}")
    if len(edge_log) > 20:
        print(f"    ... 외 {len(edge_log)-20}개")

    if dry_run:
        print("  [dry_run] 실제 수정은 수행하지 않습니다.")
        return G, 0

    print(f"  엣지 추가 완료: {added}개")
    return G2, added


# ─────────────────────────────────────────────
# 전략 C: LLM (관계 제안 + 엣지 삽입)
# ─────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = """\
당신은 의료·제약 지식그래프 전문가입니다.
아래에 제시되는 두 가지 목록을 읽고,
[고립 노드] 목록의 각 엔티티와 [후보 노드] 목록 중에서
실제로 관계가 있을 법한 후보를 골라 JSON으로 반환하세요.

출력 형식 (JSON 배열):
[
  {
    "isolated_id": "고립 노드의 node_id",
    "candidate_id": "연결할 후보 노드의 node_id",
    "relation": "관계 종류 (한국어, 15자 이내)",
    "description": "관계 설명 (50자 이내)",
    "confidence": 0.0 ~ 1.0  // 확신 정도
  },
  ...
]

규칙:
- 관계가 명확하지 않으면 해당 고립 노드는 건너뛰세요 (배열에 포함하지 말 것).
- confidence < 0.5 인 항목은 제외하세요.
- 하나의 고립 노드에 여러 후보를 제안할 수 있습니다.
- 반드시 JSON 배열만 출력하고, 다른 텍스트는 일절 포함하지 마세요.
"""


def _build_llm_prompt(
    isolated_nodes: list[tuple[str, dict]],
    candidate_nodes: list[tuple[str, dict]],
) -> str:
    def fmt(nid: str, attrs: dict) -> str:
        name  = attrs.get("entity_name",  nid)
        etype = attrs.get("entity_type",  "?")
        desc  = (attrs.get("description") or "")[:80].replace("\n", " ")
        return f"  id={nid}  name={name}  type={etype}  desc={desc}"

    lines = ["[고립 노드]"]
    for nid, attrs in isolated_nodes:
        lines.append(fmt(nid, attrs))
    lines.append("")
    lines.append("[후보 노드]")
    for nid, attrs in candidate_nodes:
        lines.append(fmt(nid, attrs))
    return "\n".join(lines)


async def _call_llm(system: str, user: str, model: str = LLM_MODEL_NAME) -> str:
    client = _get_client()
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.2,
        max_tokens=2048,
    )
    return resp.choices[0].message.content or ""


def _parse_llm_json(raw: str) -> list[dict]:
    """LLM 출력에서 JSON 배열 파싱. 실패 시 빈 리스트 반환."""
    raw = raw.strip()
    # 마크다운 코드블록 제거
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        # 배열 부분만 추출 시도
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return []


async def strategy_llm(
    G: nx.Graph,
    llm_limit: int = 50,
    candidate_limit: int = LLM_BATCH_CANDIDATES,
    dry_run: bool = False,
    min_confidence: float = 0.5,
) -> tuple[nx.Graph, int]:
    """
    LLM에게 고립 노드와 관련 있는 노드를 제안받아 엣지를 추가합니다.

    llm_limit      : LLM으로 처리할 최대 고립 노드 수 (비용 제어)
    candidate_limit: 한 번의 LLM 호출에 포함할 후보(연결된) 노드 수
    min_confidence : 채택할 최소 confidence (0.0~1.0)
    """
    isolated  = get_isolated_nodes(G)
    if not isolated:
        print("\n  [전략 C: LLM]  고립 노드 없음 — 스킵")
        return G, 0

    # 연결도 높은 노드를 후보로 사용 (top candidate_limit 개)
    connected = [n for n in G.nodes() if n not in set(isolated)]
    top_conn  = sorted(connected, key=lambda n: G.degree(n), reverse=True)[:candidate_limit]

    target_iso = isolated[:llm_limit]
    CHUNK_SIZE = 10  # 한 번의 LLM 호출에 담을 고립 노드 수

    print(f"\n  [전략 C: LLM]  고립 노드 {len(target_iso)}개 처리 (한 배치={CHUNK_SIZE}개)")
    print(f"  후보 노드 {len(top_conn)}개  |  모델: {LLM_MODEL_NAME}")

    G2    = copy.deepcopy(G)
    added = 0
    total_suggestions = []

    for chunk_start in range(0, len(target_iso), CHUNK_SIZE):
        chunk = target_iso[chunk_start:chunk_start+CHUNK_SIZE]
        iso_list  = [(nid, G.nodes[nid]) for nid in chunk]
        cand_list = [(nid, G.nodes[nid]) for nid in top_conn]

        user_prompt = _build_llm_prompt(iso_list, cand_list)
        print(f"\n  배치 {chunk_start//CHUNK_SIZE+1} / {(len(target_iso)-1)//CHUNK_SIZE+1}"
              f"  ({len(chunk)}개 노드)")

        t0  = time.time()
        raw = await _call_llm(_LLM_SYSTEM_PROMPT, user_prompt)
        print(f"  LLM 응답 ({time.time()-t0:.1f}초)")

        suggestions = _parse_llm_json(raw)
        valid = [s for s in suggestions
                 if isinstance(s, dict)
                 and s.get("confidence", 0) >= min_confidence
                 and s.get("isolated_id") in G.nodes
                 and s.get("candidate_id") in G.nodes]

        print(f"  제안 {len(suggestions)}개 → 유효 {len(valid)}개 (confidence≥{min_confidence})")
        for sug in valid:
            iso_n  = G.nodes[sug["isolated_id"]].get("entity_name",  sug["isolated_id"])
            cand_n = G.nodes[sug["candidate_id"]].get("entity_name", sug["candidate_id"])
            conf   = sug.get("confidence", 0)
            rel    = sug.get("relation", "관련")
            print(f"    [{conf:.2f}] {iso_n[:30]:<30} ─[{rel}]─> {cand_n[:30]}")

        total_suggestions.extend(valid)

        if not dry_run:
            for sug in valid:
                iso_id  = sug["isolated_id"]
                cand_id = sug["candidate_id"]
                rel     = sug.get("relation", "관련")
                desc    = sug.get("description", "LLM 제안 관계")
                conf    = float(sug.get("confidence", 0.5))
                G2.add_edge(iso_id, cand_id,
                            relation_name=rel,
                            keywords=rel,
                            description=desc,
                            weight=round(conf, 4),
                            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                added += 1

        # API 과부하 방지
        await asyncio.sleep(0.5)

    if dry_run:
        print(f"\n  [dry_run] 실제 수정은 수행하지 않습니다. (제안 총 {len(total_suggestions)}개)")
        return G, 0

    print(f"\n  엣지 추가 완료: {added}개")
    return G2, added


# ─────────────────────────────────────────────
# 전략 D: Re-link (소스 청크에서 관계 재추출)
# ─────────────────────────────────────────────

async def strategy_relink(
    G: nx.Graph,
    working_dir: str,
    llm_limit: int = 30,
    dry_run: bool = False,
) -> tuple[nx.Graph, int]:
    """
    고립 노드의 source_id 필드를 이용해 원본 청크 텍스트를 KV 스토어에서
    불러온 뒤, LLM에게 해당 엔티티와 관련된 관계를 재추출하도록 합니다.

    source_id 가 없거나 KV 스토어를 찾을 수 없는 경우 자동으로 스킵합니다.
    """
    # LightRAG KV 스토어 경로 탐색 (JSON 파일 기반)
    kv_candidates = [
        os.path.join(working_dir, f)
        for f in os.listdir(working_dir)
        if "chunk" in f.lower() and f.endswith(".json")
    ]
    if not kv_candidates:
        print("\n  [전략 D: Re-link]  청크 KV 스토어 파일 없음 — 스킵")
        return G, 0

    # 청크 ID → 텍스트 매핑 로드
    chunk_map: dict[str, str] = {}
    for kv_path in kv_candidates:
        try:
            with open(kv_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for cid, cval in data.items():
                    if isinstance(cval, dict):
                        text = cval.get("content", cval.get("text", ""))
                    else:
                        text = str(cval)
                    if text:
                        chunk_map[cid] = text
        except Exception:
            pass

    if not chunk_map:
        print("\n  [전략 D: Re-link]  청크 데이터 로드 실패 — 스킵")
        return G, 0

    isolated = get_isolated_nodes(G)
    print(f"\n  [전략 D: Re-link]  고립 노드 {len(isolated)}개 / 청크 {len(chunk_map)}개")

    SEP = "<SEP>"
    G2    = copy.deepcopy(G)
    added = 0

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
  },
  ...
]

규칙:
- confidence < 0.6 인 항목은 제외하세요.
- 반드시 JSON 배열만 출력하세요.
"""

    # 기존 노드 이름 목록 (LLM 제안과 매칭용)
    node_name_map: dict[str, str] = {}   # lower(name) → node_id
    for nid, attrs in G.nodes(data=True):
        name = (attrs.get("entity_name") or nid).strip()
        node_name_map[name.lower()] = nid

    processed = 0
    for nid in isolated[:llm_limit]:
        attrs    = G.nodes[nid]
        name     = attrs.get("entity_name", nid)
        src_ids  = (attrs.get("source_id") or "").split(SEP)
        src_ids  = [s.strip() for s in src_ids if s.strip() in chunk_map]
        if not src_ids:
            continue

        chunk_text = "\n---\n".join(chunk_map[s] for s in src_ids[:3])[:3000]
        user_prompt = f"[타겟 엔티티]\n{name}\n\n[텍스트]\n{chunk_text}"

        t0  = time.time()
        raw = await _call_llm(_RELINK_SYSTEM, user_prompt)
        suggestions = _parse_llm_json(raw)
        elapsed = time.time() - t0

        valid = [s for s in suggestions
                 if isinstance(s, dict) and s.get("confidence", 0) >= 0.6]

        print(f"  [{processed+1}/{min(len(isolated),llm_limit)}] {name[:35]:<35}"
              f"  제안 {len(valid)}개  ({elapsed:.1f}초)")

        for sug in valid:
            other_name = (sug.get("other_entity") or "").strip().lower()
            target_id  = node_name_map.get(other_name)
            if not target_id or target_id == nid:
                continue
            rel  = sug.get("relation", "관련")
            desc = sug.get("description", "")
            conf = float(sug.get("confidence", 0.6))
            print(f"    + {name[:25]} ─[{rel}]─> "
                  f"{G.nodes.get(target_id, {}).get('entity_name', target_id)[:25]}"
                  f"  (conf={conf:.2f})")
            if not dry_run:
                G2.add_edge(nid, target_id,
                            relation_name=rel,
                            keywords=rel,
                            description=desc,
                            weight=round(conf, 4),
                            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                added += 1

        processed += 1
        await asyncio.sleep(0.3)

    if dry_run:
        print(f"\n  [dry_run] 실제 수정은 수행하지 않습니다.")
        return G, 0

    print(f"\n  엣지 추가 완료: {added}개")
    return G2, added


# ─────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────

async def run(args: argparse.Namespace) -> None:
    working_dir = args.working_dir

    # 그래프 로드
    try:
        G, gpath = _load_graph(working_dir)
    except FileNotFoundError as e:
        print(f"\n  오류: {e}")
        sys.exit(1)

    print(f"\n  GraphML: {os.path.basename(gpath)}")
    print_stats(G, "변경 전")

    if args.detail:
        print_isolated_detail(G, limit=args.detail_limit)

    # --stats 만 요청한 경우 종료
    if not (args.prune or args.embed or args.llm or args.relink):
        return

    # 백업
    backup_path = _backup(gpath)
    print(f"\n  백업 저장: {os.path.basename(backup_path)}")

    total_changed = 0

    # 전략 A: Prune
    if args.prune:
        G, n = strategy_prune(
            G,
            min_desc_len=args.prune_min_desc,
            dry_run=args.dry_run,
        )
        total_changed += n

    # 전략 B: Embed
    if args.embed:
        G, n = await strategy_embed(
            G,
            threshold=args.embed_threshold,
            top_k=args.embed_top_k,
            dry_run=args.dry_run,
        )
        total_changed += n

    # 전략 C: LLM
    if args.llm:
        G, n = await strategy_llm(
            G,
            llm_limit=args.llm_limit,
            dry_run=args.dry_run,
            min_confidence=args.llm_min_confidence,
        )
        total_changed += n

    # 전략 D: Re-link
    if args.relink:
        G, n = await strategy_relink(
            G,
            working_dir=working_dir,
            llm_limit=args.relink_limit,
            dry_run=args.dry_run,
        )
        total_changed += n

    # 저장
    if not args.dry_run and total_changed > 0:
        _save_graph(G, gpath)
        print(f"\n  그래프 저장 완료: {os.path.basename(gpath)}")
        print_stats(G, "변경 후")
    elif args.dry_run:
        print("\n  [dry_run 모드] 파일 저장 없음")
    else:
        print("\n  변경 사항 없음")


# ─────────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="entity_solution.py",
        description="고립 노드(degree=0) 해소 유틸리티",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # 공통
    p.add_argument("--working-dir", default=WORKING_DIR, metavar="DIR",
                   help=f"LightRAG working_dir 경로 (기본: {WORKING_DIR})")
    p.add_argument("--dry-run", action="store_true",
                   help="변경 내용 미리보기만 출력 (실제 저장 안 함)")
    p.add_argument("--stats", action="store_true",
                   help="통계 출력만 하고 종료")
    p.add_argument("--detail", action="store_true",
                   help="고립 노드 상세 목록 출력")
    p.add_argument("--detail-limit", type=int, default=50, metavar="N",
                   help="상세 목록 최대 출력 수 (기본: 50)")

    # 전략 A
    p.add_argument("--prune", action="store_true",
                   help="[전략 A] 품질 미달 고립 노드 삭제")
    p.add_argument("--prune-min-desc", type=int, default=MIN_DESC_LEN, metavar="N",
                   help=f"삭제 기준 최소 설명 길이 (기본: {MIN_DESC_LEN}자)")

    # 전략 B
    p.add_argument("--embed", action="store_true",
                   help="[전략 B] 임베딩 유사도로 고립 노드 연결")
    p.add_argument("--embed-threshold", type=float, default=DEFAULT_EMBED_THRESHOLD,
                   metavar="F",
                   help=f"연결 최소 코사인 유사도 (기본: {DEFAULT_EMBED_THRESHOLD})")
    p.add_argument("--embed-top-k", type=int, default=DEFAULT_TOP_K, metavar="K",
                   help=f"노드 하나당 연결할 최대 후보 수 (기본: {DEFAULT_TOP_K})")

    # 전략 C
    p.add_argument("--llm", action="store_true",
                   help="[전략 C] LLM이 관계 제안 후 엣지 삽입")
    p.add_argument("--llm-limit", type=int, default=50, metavar="N",
                   help="LLM으로 처리할 최대 고립 노드 수 (기본: 50)")
    p.add_argument("--llm-min-confidence", type=float, default=0.5, metavar="F",
                   help="LLM 제안 최소 confidence (기본: 0.5)")

    # 전략 D
    p.add_argument("--relink", action="store_true",
                   help="[전략 D] 소스 청크에서 관계 재추출")
    p.add_argument("--relink-limit", type=int, default=30, metavar="N",
                   help="Re-link으로 처리할 최대 고립 노드 수 (기본: 30)")

    return p


if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()

    # --stats 플래그가 없어도 아무 전략도 지정 안 하면 stats 표시
    if not any([args.prune, args.embed, args.llm, args.relink]):
        args.stats = True

    asyncio.run(run(args))
