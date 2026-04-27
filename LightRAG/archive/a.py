import os
import html
import asyncio
import time
import numpy as np
import networkx as nx
from pyvis.network import Network
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc

try:
    import tiktoken
    _enc = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    _enc = None

# =============================================
# 설정
# =============================================
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Qdrant 로컬 서버 주소 (docker run -d -p 6333:6333 qdrant/qdrant)
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.join(_BASE_DIR, "lightrag_a")
MD_DIR = os.path.join(_BASE_DIR, "test_md")

os.makedirs(WORKING_DIR, exist_ok=True)

# =============================================
# 비용 / 타이밍 추적
# =============================================
_LLM_INPUT_COST  = 0.000150   # $0.150 / 1M tokens  (gpt-4o-mini input)
_LLM_OUTPUT_COST = 0.000600   # $0.600 / 1M tokens  (gpt-4o-mini output)
_EMB_COST        = 0.000020   # $0.020 / 1M tokens  (text-embedding-3-small)

# 페이즈별 누적 사용량
_usage = {
    "insert": {"llm_in": 0, "llm_out": 0, "emb": 0},
    "query":  {"llm_in": 0, "llm_out": 0, "emb": 0},
}
_phase    = "insert"   # 현재 페이즈 ("insert" | "query")
_call_log = []         # 현재 쿼리 내 호출 로그 (쿼리마다 초기화)


def _count(text: str) -> int:
    if _enc and text:
        return len(_enc.encode(str(text)))
    return len(str(text)) // 4


async def tracked_llm(prompt, system_prompt=None, history_messages=[], **kwargs):
    in_tok = _count(system_prompt or "") + _count(prompt)
    for m in (history_messages or []):
        in_tok += _count(m.get("content", ""))

    t = time.time()
    result = await gpt_4o_mini_complete(
        prompt, system_prompt=system_prompt,
        history_messages=history_messages, **kwargs
    )
    dur = time.time() - t
    out_tok = _count(result)

    _usage[_phase]["llm_in"]  += in_tok
    _usage[_phase]["llm_out"] += out_tok
    if _phase == "query":
        _call_log.append({"type": "llm", "in": in_tok, "out": out_tok, "sec": dur})
    return result


async def tracked_embed(texts, **kwargs):
    _fn = openai_embed.func if hasattr(openai_embed, "func") else openai_embed
    t = time.time()
    result = await _fn(texts, **kwargs)
    dur = time.time() - t
    tok = sum(_count(x) for x in (texts if isinstance(texts, list) else [texts]))

    _usage[_phase]["emb"] += tok
    if _phase == "query":
        _call_log.append({"type": "emb", "in": tok, "out": 0, "sec": dur})
    return result


def _phase_cost(phase: str) -> tuple:
    u = _usage[phase]
    i = u["llm_in"]  / 1000 * _LLM_INPUT_COST
    o = u["llm_out"] / 1000 * _LLM_OUTPUT_COST
    e = u["emb"]     / 1000 * _EMB_COST
    return i, o, e, i + o + e


def print_insert_cost():
    i, o, e, total = _phase_cost("insert")
    print(f"\n{'='*52}")
    print(f"[삽입(관계 형성) 크레딧]")
    print(f"  LLM 입력 : {_usage['insert']['llm_in']:>10,} tok  ${i:.4f}")
    print(f"  LLM 출력 : {_usage['insert']['llm_out']:>10,} tok  ${o:.4f}")
    print(f"  임베딩   : {_usage['insert']['emb']:>10,} tok  ${e:.4f}")
    print(f"  소계     :                   ${total:.4f}  (≈ ₩{total*1380:,.0f})")
    print(f"{'='*52}")


def print_total_cost():
    i1, o1, e1, t1 = _phase_cost("insert")
    i2, o2, e2, t2 = _phase_cost("query")
    total = t1 + t2
    print(f"\n{'='*52}")
    print(f"[전체 크레딧 합계]")
    print(f"  삽입     :  ${t1:.4f}")
    print(f"  쿼리     :  ${t2:.4f}")
    print(f"  합계     :  ${total:.4f}  (≈ ₩{total*1380:,.0f})")
    print(f"{'='*52}\n")

# =============================================
# 문서 삽입 (지식그래프 구축)
# =============================================
async def insert_documents(rag: LightRAG):
    """test_md/ 의 MD 파일을 원문 그대로 삽입 — 청킹은 LightRAG 내부에 맡김"""
    md_files = sorted([
        f for f in os.listdir(MD_DIR) if f.endswith(".md")
    ])
    if not md_files:
        print(f"❌ MD 파일을 찾을 수 없습니다: {MD_DIR}")
        return

    total = len(md_files)
    for idx, fname in enumerate(md_files, 1):
        fpath = os.path.join(MD_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            document = f.read().strip()

        if not document:
            print(f"  [{idx}/{total}] 건너뜀 (빈 파일): {fname}")
            continue

        t = time.time()
        print(f"  [{idx}/{total}] 삽입 중: {fname} ({len(document):,} chars)")
        try:
            await rag.ainsert(document)
            print(f"  [{idx}/{total}] 완료: {time.time()-t:.1f}초")
        except Exception as e:
            print(f"  [{idx}/{total}] 오류: {e}")

# =============================================
# 그래프 시각화
# =============================================
def visualize_graph(
    graphml_path: str = None,
    output_html: str = "./lightrag_a/knowledge_graph.html",
    max_nodes: int = 200,
):
    """
    LightRAG가 생성한 GraphML 파일을 pyvis 인터랙티브 HTML로 시각화합니다.

    Args:
        graphml_path: GraphML 파일 경로 (None이면 WORKING_DIR에서 자동 탐색)
        output_html:  출력 HTML 파일 경로
        max_nodes:    시각화할 최대 노드 수 (너무 크면 느림)
    """
    # GraphML 파일 탐색
    if graphml_path is None:
        candidates = [
            os.path.join(WORKING_DIR, f)
            for f in os.listdir(WORKING_DIR)
            if f.endswith(".graphml")
        ]
        if not candidates:
            print("❌ GraphML 파일을 찾을 수 없습니다. 문서 삽입을 먼저 실행하세요.")
            return
        graphml_path = max(candidates, key=os.path.getmtime)

    print(f"[시각화] GraphML 로딩: {graphml_path}")
    G = nx.read_graphml(graphml_path)
    print(f"  노드 수: {G.number_of_nodes()}, 엣지 수: {G.number_of_edges()}")

    # 너무 큰 그래프는 degree 기준 상위 노드만 추출
    if G.number_of_nodes() > max_nodes:
        print(f"  노드가 {max_nodes}개 초과 → 연결수 상위 {max_nodes}개 노드만 표시")
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_ids = [n for n, _ in top_nodes]
        G = G.subgraph(top_node_ids).copy()

    # ── pyvis 네트워크 설정 ──────────────────────────────
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#0f0f1a",          # 다크 배경
        font_color="#e0e0e0",
        directed=G.is_directed(),
        notebook=False,
    )

    # 물리 엔진 옵션 (Barnes-Hut 알고리즘으로 자연스러운 레이아웃)
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "springLength": 150,
          "springConstant": 0.04,
          "damping": 0.09
        },
        "minVelocity": 0.75
      },
      "edges": {
        "smooth": { "type": "dynamic" },
        "color": { "inherit": "both" },
        "width": 1.5,
        "font": { "size": 10, "color": "#aaaaaa", "strokeWidth": 0 }
      },
      "nodes": {
        "shape": "dot",
        "font": { "size": 13, "color": "#ffffff" },
        "borderWidth": 2
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true,
        "tooltipDelay": 200
      }
    }
    """)

    # ── 노드 색상 팔레트 (엔티티 타입별 색상 자동 부여) ──
    type_colors = {}
    palette = [
        "#4fc3f7", "#81c784", "#ffb74d", "#e57373",
        "#ba68c8", "#4db6ac", "#f06292", "#aed581",
        "#ff8a65", "#90a4ae",
    ]

    def get_color(entity_type: str) -> str:
        if entity_type not in type_colors:
            type_colors[entity_type] = palette[len(type_colors) % len(palette)]
        return type_colors[entity_type]

    # ── 노드 추가 ──────────────────────────────────────
    for node_id, attrs in G.nodes(data=True):
        label = attrs.get("entity_name", attrs.get("id", str(node_id)))
        etype = attrs.get("entity_type", "UNKNOWN")
        desc  = attrs.get("description", "")
        color = get_color(etype)
        degree = G.degree(node_id)

        net.add_node(
            str(node_id),
            label=str(label)[:40],                  # 너무 긴 레이블 자름
            title=f"<b>[{etype}]</b> {label}<br/>{desc[:200]}",
            color={
                "background": color,
                "border": "#ffffff",
                "highlight": {"background": "#ffffff", "border": color},
            },
            size=max(10, min(40, degree * 3)),       # degree에 비례한 노드 크기
            font={"color": "#ffffff", "size": 13},
        )

    # ── 엣지 추가 ──────────────────────────────────────
    for src, dst, attrs in G.edges(data=True):
        relation  = attrs.get("relation_name", attrs.get("relation", ""))
        try:
            weight = float(attrs.get("weight", 1.0))
        except (ValueError, TypeError):
            weight = 1.0
        keywords  = attrs.get("keywords", "")

        net.add_edge(
            str(src),
            str(dst),
            title=f"{relation}<br/>keywords: {keywords}",
            label=str(relation)[:25] if relation else "",
            width=max(1.0, min(5.0, weight * 2)),
            color={"opacity": 0.7},
        )

    # ── 범례 (엔티티 타입 → 색상) ────────────────────
    legend_html = "<div style='position:fixed;top:10px;right:10px;background:#1a1a2e;padding:12px;border-radius:8px;font-family:sans-serif;font-size:12px;color:#fff;z-index:999;'>"
    legend_html += "<b>Entity Types</b><br/>"
    for etype, color in type_colors.items():
        legend_html += f"<span style='display:inline-block;width:12px;height:12px;background:{color};border-radius:50%;margin-right:5px;'></span>{html.escape(etype)}<br/>"
    legend_html += "</div>"

    net.save_graph(output_html)

    # 범례를 HTML에 삽입
    with open(output_html, "r", encoding="utf-8") as f:
        html_content = f.read()
    html_content = html_content.replace("</body>", f"{legend_html}</body>")
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ 시각화 완료 → {os.path.abspath(output_html)}")
    print("   브라우저에서 위 파일을 열면 인터랙티브 지식그래프를 확인할 수 있습니다.")

# =============================================
# 쿼리 예시
# =============================================
async def run_queries(rag: LightRAG):
    global _phase, _call_log

    queries = [
        "리리카와 쎄레브렉스의 안전성 정보를 비교해줘",
        "쎄레브렉스의 주요 효능이 뭐야?",
        "기넥신 관련 상병코드 알려줘",
    ]

    _phase = "query"

    for query in queries:
        _call_log = []
        print(f"\n{'='*60}")
        print(f"[쿼리] {query}")
        print('='*60)

        t_total = time.time()
        result = await rag.aquery(query, param=QueryParam(mode="hybrid"))
        total_sec = time.time() - t_total

        print(result)

        # ── 단계별 타이밍 분석 ──────────────────────────────
        emb_calls = [c for c in _call_log if c["type"] == "emb"]
        llm_calls = [c for c in _call_log if c["type"] == "llm"]

        # LightRAG 쿼리 흐름: embed → (키워드추출 LLM) → 답변생성 LLM
        # 마지막 LLM 호출 = 답변 생성, 그 이전 = 검색 보조
        retrieval_llm = llm_calls[:-1] if len(llm_calls) > 1 else []
        gen_llm       = llm_calls[-1]  if llm_calls else None

        t_emb      = sum(c["sec"] for c in emb_calls)
        t_ret_llm  = sum(c["sec"] for c in retrieval_llm)
        t_gen      = gen_llm["sec"] if gen_llm else 0
        t_other    = total_sec - t_emb - t_ret_llm - t_gen  # 벡터/그래프 서칭 등

        _, _, _, q_cost = _phase_cost("query")
        print(f"\n  ┌─ [타이밍 분석] ───────────────────────────")
        print(f"  │  쿼리 임베딩       : {t_emb:.2f}초  ({len(emb_calls)}회)")
        print(f"  │  벡터·그래프 서칭  : {t_other:.2f}초  (DB 검색)")
        if retrieval_llm:
            print(f"  │  검색 보조 LLM     : {t_ret_llm:.2f}초  ({len(retrieval_llm)}회, 키워드 추출 등)")
        print(f"  │  답변 생성 LLM     : {t_gen:.2f}초  (컨텍스트 → 답변)")
        print(f"  │  ─────────────────────────────────────")
        print(f"  └─ 합계              : {total_sec:.2f}초")

# =============================================
# 메인 실행
# =============================================
async def main():
    global _phase
    t_start = time.time()
    print("🚀 LightRAG 지식그래프 구축 시작\n")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=tracked_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=tracked_embed,
        ),
        # ── Qdrant 로컬 서버 (docker run -d -p 6333:6333 qdrant/qdrant) ──
        vector_storage="QdrantVectorDBStorage",
        vector_db_storage_cls_kwargs={
            "collection_name": "lightrag_labq",
        },
        # ─────────────────────────────────────────────────────────────────
    )
    await rag.initialize_storages()

    # 1. 문서 삽입 & 지식그래프 생성
    _phase = "insert"
    t1 = time.time()
    await insert_documents(rag)
    elapsed_insert = time.time() - t1
    print(f"[시간] 삽입 합계: {elapsed_insert:.1f}초")
    print_insert_cost()

    # 2. 그래프 시각화
    t2 = time.time()
    visualize_graph()
    print(f"[시간] 시각화: {time.time()-t2:.1f}초")

    # 3. 쿼리 테스트
    t3 = time.time()
    await run_queries(rag)
    print(f"\n[시간] 쿼리 합계: {time.time()-t3:.1f}초")

    print(f"\n[총 소요 시간] {time.time()-t_start:.1f}초")
    print_total_cost()

if __name__ == "__main__":
    asyncio.run(main())