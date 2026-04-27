import os
import re
import json
import html
import asyncio
import time
import networkx as nx
from pyvis.network import Network
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc

# =============================================
# 설정
# =============================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

_BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR  = os.path.join(_BASE_DIR, "lightrag_a")
RAW_JSON     = os.path.join(_BASE_DIR, "parsed_md_output_h2.json")
JSON_PATH    = os.path.join(_BASE_DIR, "parsed_output.json")

os.makedirs(WORKING_DIR, exist_ok=True)

# =============================================
# JSON 전처리 → parsed_output.json 생성
# =============================================
_SK_FOOTER = re.compile(
    r'사원번호\s*:\s*\d+.*?이 문서는 SK케미칼 보안문서로서 외부반출을 금지합니다\.?',
    re.DOTALL,
)

_SKIP_HEADERS = {
    "Internal Use Only",
    "Open Access",
    "RESEARCH ARTICLE",
    "Competing interests",
    "Acknowledgements",
    "Author details",
    "References",
    "Root",
}


def _clean(content: str) -> str:
    return _SK_FOOTER.sub("", content).strip()


def _should_skip(header: str, content: str, level: int) -> bool:
    if header in _SKIP_HEADERS:
        return True
    if not header and not content:
        return True
    # content 없는 level≤1은 섹션 구분자일 뿐
    if level <= 1 and not content:
        return True
    return False


def preprocess() -> None:
    """parsed_md_output_h2.json 을 정제해 parsed_output.json 으로 저장."""
    with open(RAW_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    result = {}
    total_before = total_after = 0

    for filename, sections in raw.items():
        total_before += len(sections)

        # 1단계: 노이즈 제거 + footer 클리닝
        filtered = []
        for sec in sections:
            header  = sec.get("header", "").strip()
            level   = sec.get("level", 1)
            content = _clean(sec.get("content", ""))
            if _should_skip(header, content, level):
                continue
            filtered.append({"header": header, "level": level, "content": content})

        # 2단계: 연속된 동일 헤더 섹션 병합
        merged = []
        for sec in filtered:
            if (merged
                    and merged[-1]["header"] == sec["header"]
                    and merged[-1]["level"] == sec["level"]):
                sep = "\n\n" if merged[-1]["content"] and sec["content"] else ""
                merged[-1]["content"] += sep + sec["content"]
            else:
                merged.append(dict(sec))

        if merged:
            result[filename] = merged
            total_after += len(merged)
            print(f"  {filename}: {len(sections)}개 → {len(merged)}개 섹션")

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    removed = total_before - total_after
    print(f"\n✅ 전처리 완료: {total_before}개 → {total_after}개 섹션 ({removed}개 제거)")
    print(f"   저장 경로: {JSON_PATH}\n")


# =============================================
# 문서 삽입
# =============================================
async def insert_documents(rag: LightRAG) -> None:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        parsed = json.load(f)

    documents = []
    for filename, sections in parsed.items():
        for sec in sections:
            header  = sec.get("header", "").strip()
            level   = sec.get("level", 1)
            content = sec.get("content", "").strip()
            if not header and not content:
                continue
            parts = [f"[출처: {filename}]"]
            if header:
                parts.append(f"{'#' * level} {header}")
            if content:
                parts.append(content)
            documents.append("\n\n".join(parts))

    print(f"총 {len(documents)}개 섹션 삽입 시작...")
    await rag.ainsert(documents)
    print(f"✅ 삽입 완료 — {len(documents)}개 섹션")


# =============================================
# 그래프 시각화
# =============================================
def visualize_graph(
    graphml_path: str = None,
    output_html: str = None,
    max_nodes: int = 1000,
) -> None:
    if output_html is None:
        output_html = os.path.join(WORKING_DIR, "knowledge_graph.html")

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
    print(f"  노드: {G.number_of_nodes()}, 엣지: {G.number_of_edges()}")

    if G.number_of_nodes() > max_nodes:
        print(f"  노드 {max_nodes}개 초과 → 연결수 상위 {max_nodes}개만 표시")
        top = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        G = G.subgraph([n for n, _ in top]).copy()

    net = Network(
        height="900px", width="100%",
        bgcolor="#0f0f1a", font_color="#e0e0e0",
        directed=G.is_directed(), notebook=False,
    )
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

    type_colors: dict = {}
    palette = [
        "#4fc3f7", "#81c784", "#ffb74d", "#e57373",
        "#ba68c8", "#4db6ac", "#f06292", "#aed581",
        "#ff8a65", "#90a4ae",
    ]

    def get_color(etype: str) -> str:
        if etype not in type_colors:
            type_colors[etype] = palette[len(type_colors) % len(palette)]
        return type_colors[etype]

    for node_id, attrs in G.nodes(data=True):
        label  = attrs.get("entity_name", attrs.get("id", str(node_id)))
        etype  = attrs.get("entity_type", "UNKNOWN")
        desc   = attrs.get("description", "")
        color  = get_color(etype)
        degree = G.degree(node_id)
        net.add_node(
            str(node_id),
            label=str(label)[:40],
            title=f"<b>[{etype}]</b> {label}<br/>{desc[:200]}",
            color={"background": color, "border": "#ffffff",
                   "highlight": {"background": "#ffffff", "border": color}},
            size=max(10, min(40, degree * 3)),
            font={"color": "#ffffff", "size": 13},
        )

    for src, dst, attrs in G.edges(data=True):
        relation = attrs.get("relation_name", attrs.get("relation", ""))
        try:
            weight = float(attrs.get("weight", 1.0))
        except (ValueError, TypeError):
            weight = 1.0
        keywords = attrs.get("keywords", "")
        net.add_edge(
            str(src), str(dst),
            title=f"{relation}<br/>keywords: {keywords}",
            label=str(relation)[:25] if relation else "",
            width=max(1.0, min(5.0, weight * 2)),
            color={"opacity": 0.7},
        )

    legend_html = (
        "<div style='position:fixed;top:10px;right:10px;background:#1a1a2e;"
        "padding:12px;border-radius:8px;font-family:sans-serif;font-size:12px;"
        "color:#fff;z-index:999;'><b>Entity Types</b><br/>"
    )
    for etype, color in type_colors.items():
        legend_html += (
            f"<span style='display:inline-block;width:12px;height:12px;"
            f"background:{color};border-radius:50%;margin-right:5px;'></span>"
            f"{html.escape(etype)}<br/>"
        )
    legend_html += "</div>"

    net.save_graph(output_html)
    with open(output_html, "r", encoding="utf-8") as f:
        html_content = f.read()
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content.replace("</body>", f"{legend_html}</body>"))

    print(f"✅ 시각화 완료 → {os.path.abspath(output_html)}")


# =============================================
# 쿼리 예시
# =============================================
async def run_queries(rag: LightRAG) -> None:
    queries = [
        "선별급여 받는 약 5개 알려줘",
        "기넥신을 어떤 병원에 추천해야할까?",
        "기넥신 상병코드랑 부작용 알려줘."
    ]
    for query in queries:
        print(f"\n{'='*60}")
        print(f"[쿼리] {query}")
        print('='*60)
        result = await rag.aquery(query, param=QueryParam(mode="hybrid"))
        print(result)


# =============================================
# 메인 실행
# =============================================
async def main() -> None:
    t_start = time.time()
    print("🚀 LightRAG (b.py) 시작\n")

    # 0. 전처리
    if not os.path.isfile(JSON_PATH):
        print("=== [전처리] parsed_output.json 생성 중 ===")
        t0 = time.time()
        preprocess()
        print(f"[시간] 전처리: {time.time() - t0:.1f}초\n")
    else:
        print(f"[전처리 생략] {JSON_PATH} 이미 존재\n")

    # 1. LightRAG 초기화
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=openai_embed.func if hasattr(openai_embed, "func") else openai_embed,
        ),
        vector_storage="QdrantVectorDBStorage",
        vector_db_storage_cls_kwargs={
            "collection_name": "lightrag_labq_a",
        },
    )
    await rag.initialize_storages()

    # # 2. 문서 삽입
    print("=== [삽입] ===")
    t1 = time.time()
    await insert_documents(rag)
    print(f"[시간] 삽입: {time.time() - t1:.1f}초\n")

    # 3. 그래프 시각화
    print("=== [시각화] ===")
    t2 = time.time()
    visualize_graph()
    print(f"[시간] 시각화: {time.time() - t2:.1f}초\n")

    # 4. 쿼리 테스트
    print("=== [쿼리] ===")
    t3 = time.time()
    await run_queries(rag)
    print(f"[시간] 쿼리: {time.time() - t3:.1f}초\n")

    print(f"{'='*60}")
    print(f"[총 소요 시간] {time.time() - t_start:.1f}초")


if __name__ == "__main__":
    asyncio.run(main())
