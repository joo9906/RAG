"""
sql_test.py
===========
lightrag_c/ 의 실제 데이터를 PostgreSQL에 적재하고 쿼리를 시연한다.

  데이터 소스
    lightrag_c/kv_store_doc_status.json    → doc_status 테이블
    lightrag_c/kv_store_text_chunks.json   → text_chunks 테이블
    lightrag_c/graph_chunk_entity_relation.graphml
        노드(567개) → entities 테이블
        엣지(430개) → relations 테이블

  Qdrant  → 벡터 임베딩 (의미 유사도 검색)
  여기(PG) → 원문 텍스트·메타데이터·그래프 구조 (실제 내용 저장)

실행 전 준비
    pip install psycopg2-binary
    docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:16
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
import psycopg2

# Windows cp949 터미널에서 한글·특수문자 깨짐 방지
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from psycopg2.extras import RealDictCursor, execute_values

# ─────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────
import os
_BASE = os.path.dirname(os.path.abspath(__file__))
_LC   = os.path.join(_BASE, "lightrag_c")

GRAPHML_PATH    = os.path.join(_LC, "graph_chunk_entity_relation.graphml")
DOC_STATUS_PATH = os.path.join(_LC, "kv_store_doc_status.json")
CHUNKS_PATH     = os.path.join(_LC, "kv_store_text_chunks.json")

CONN = dict(host="localhost", port=5432, dbname="postgres",
            user="postgres", password="postgres")

NS = "http://graphml.graphdrawing.org/xmlns"   # GraphML 네임스페이스


# ─────────────────────────────────────────
# 1. 스키마
# ─────────────────────────────────────────
DDL = """
DROP TABLE IF EXISTS relations  CASCADE;
DROP TABLE IF EXISTS entities   CASCADE;
DROP TABLE IF EXISTS text_chunks CASCADE;
DROP TABLE IF EXISTS doc_status CASCADE;

-- 문서 처리 상태
CREATE TABLE doc_status (
    doc_id        TEXT PRIMARY KEY,
    file_path     TEXT,
    status        TEXT    NOT NULL DEFAULT 'pending',
    chunks_count  INT,
    content_length INT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- 텍스트 청크  (Qdrant point_id == chunk_id)
CREATE TABLE text_chunks (
    chunk_id      TEXT PRIMARY KEY,
    doc_id        TEXT REFERENCES doc_status(doc_id) ON DELETE CASCADE,
    content       TEXT NOT NULL,
    chunk_order   INT  NOT NULL DEFAULT 0,
    tokens        INT,
    file_path     TEXT
);
CREATE INDEX idx_chunks_doc   ON text_chunks(doc_id);
CREATE INDEX idx_chunks_order ON text_chunks(doc_id, chunk_order);

-- 엔티티 (지식 그래프 노드)  — 벡터는 Qdrant, 메타는 여기
CREATE TABLE entities (
    entity_name   TEXT PRIMARY KEY,
    entity_type   TEXT,
    description   TEXT,
    source_chunks TEXT[],   -- 출처 chunk_id 목록
    file_path     TEXT,
    created_at    BIGINT
);
CREATE INDEX idx_entities_type ON entities(entity_type);

-- 관계 (지식 그래프 엣지)
CREATE TABLE relations (
    id             SERIAL PRIMARY KEY,
    source_entity  TEXT REFERENCES entities(entity_name) ON DELETE CASCADE,
    target_entity  TEXT REFERENCES entities(entity_name) ON DELETE CASCADE,
    description    TEXT,
    keywords       TEXT,
    weight         FLOAT DEFAULT 1.0,
    source_chunks  TEXT[],
    file_path      TEXT,
    created_at     BIGINT,
    UNIQUE(source_entity, target_entity)
);
CREATE INDEX idx_rel_src ON relations(source_entity);
CREATE INDEX idx_rel_tgt ON relations(target_entity);
"""


# ─────────────────────────────────────────
# 2. GraphML 파싱 (networkx 없이)
# ─────────────────────────────────────────
def parse_graphml(path: str):
    """
    GraphML을 파싱해 (nodes, edges) 반환.
    LightRAG가 만든 키 매핑(d0~d13)을 자동으로 해석한다.
    """
    tree = ET.parse(path)
    root = tree.getroot()

    def tag(name):
        return f"{{{NS}}}{name}"

    # ── 키 매핑 읽기 (d0=entity_id, d1=entity_type, ...) ──
    key_map = {}   # id → attr.name
    for k in root.findall(tag("key")):
        key_map[k.attrib["id"]] = k.attrib.get("attr.name", k.attrib["id"])

    # <graph> 요소 탐색 (네임스페이스 있는 경우와 없는 경우 모두 처리)
    graph = root.find(tag("graph"))
    if graph is None:
        graph = root.find("graph")
    nodes, edges = [], []

    for node in graph.findall(tag("node")):
        obj = {"entity_name": node.attrib["id"]}
        for d in node.findall(tag("data")):
            field = key_map.get(d.attrib["key"], d.attrib["key"])
            obj[field] = (d.text or "").strip()
        nodes.append(obj)

    for edge in graph.findall(tag("edge")):
        obj = {
            "source_entity": edge.attrib["source"],
            "target_entity": edge.attrib["target"],
        }
        for d in edge.findall(tag("data")):
            field = key_map.get(d.attrib["key"], d.attrib["key"])
            obj[field] = (d.text or "").strip()
        edges.append(obj)

    return nodes, edges


# ─────────────────────────────────────────
# 3. 데이터 로드 + 적재
# ─────────────────────────────────────────
def load_and_insert(cur):

    # ── doc_status ──────────────────────────────────────
    with open(DOC_STATUS_PATH, encoding="utf-8") as f:
        raw_docs = json.load(f)

    doc_rows = []
    for doc_id, v in raw_docs.items():
        doc_rows.append((
            doc_id,
            v.get("file_path", ""),
            v.get("status", "processed"),
            v.get("chunks_count"),
            v.get("content_length"),
        ))

    execute_values(cur,
        "INSERT INTO doc_status (doc_id,file_path,status,chunks_count,content_length) VALUES %s",
        doc_rows,
    )
    print(f"  doc_status   : {len(doc_rows)}건 적재")

    # ── text_chunks ─────────────────────────────────────
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        raw_chunks = json.load(f)

    chunk_rows = []
    for chunk_id, v in raw_chunks.items():
        doc_id = v.get("full_doc_id", "")
        if doc_id not in {r[0] for r in doc_rows}:
            continue            # 고아 청크 스킵
        chunk_rows.append((
            chunk_id,
            doc_id,
            v.get("content", ""),
            v.get("chunk_order_index", 0),
            v.get("tokens"),
            v.get("file_path", ""),
        ))

    execute_values(cur,
        "INSERT INTO text_chunks (chunk_id,doc_id,content,chunk_order,tokens,file_path) VALUES %s",
        chunk_rows,
    )
    print(f"  text_chunks  : {len(chunk_rows)}건 적재")

    # ── GraphML 파싱 ────────────────────────────────────
    nodes, edges = parse_graphml(GRAPHML_PATH)
    print(f"  GraphML 파싱 : 노드 {len(nodes)}개, 엣지 {len(edges)}개")

    # ── entities ────────────────────────────────────────
    entity_rows = []
    entity_names = set()
    for n in nodes:
        name = n["entity_name"]
        if name in entity_names:
            continue
        entity_names.add(name)
        # source_id는 '<SEP>'으로 이어진 chunk_id 목록
        source_ids = [s.strip() for s in n.get("source_id", "").split("<SEP>") if s.strip()]
        entity_rows.append((
            name,
            n.get("entity_type", "unknown"),
            n.get("description", ""),
            source_ids,
            n.get("file_path", ""),
            int(n["created_at"]) if n.get("created_at", "").isdigit() else None,
        ))

    execute_values(cur,
        """INSERT INTO entities (entity_name,entity_type,description,source_chunks,file_path,created_at)
           VALUES %s ON CONFLICT (entity_name) DO NOTHING""",
        entity_rows,
    )
    print(f"  entities     : {len(entity_rows)}건 적재")

    # ── relations ───────────────────────────────────────
    # 양쪽 엔티티가 모두 entities에 있는 엣지만 삽입 (FK 오류 방지)
    relation_rows = []
    seen_pairs = set()
    for e in edges:
        src = e["source_entity"]
        tgt = e["target_entity"]
        pair = (src, tgt)
        if src not in entity_names or tgt not in entity_names:
            continue
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        source_ids = [s.strip() for s in e.get("source_id", "").split("<SEP>") if s.strip()]
        try:
            weight = float(e.get("weight", 1.0))
        except (ValueError, TypeError):
            weight = 1.0
        relation_rows.append((
            src, tgt,
            e.get("description", ""),
            e.get("keywords", ""),
            weight,
            source_ids,
            e.get("file_path", ""),
            int(e["created_at"]) if e.get("created_at", "").isdigit() else None,
        ))

    execute_values(cur,
        """INSERT INTO relations
             (source_entity,target_entity,description,keywords,weight,source_chunks,file_path,created_at)
           VALUES %s ON CONFLICT (source_entity,target_entity) DO NOTHING""",
        relation_rows,
    )
    print(f"  relations    : {len(relation_rows)}건 적재")
    print()


# ─────────────────────────────────────────
# 4. 실제 데이터 기반 쿼리 시연
# ─────────────────────────────────────────
def run_demo_queries(cur):

    # ── Q1. 문서별 청크 현황 ────────────────────────────
    sep()
    print("Q1. 문서별 청크 현황")
    sep()
    cur.execute("""
        SELECT d.file_path, d.status, d.chunks_count,
               COUNT(c.chunk_id) AS actual, SUM(c.tokens) AS total_tokens
        FROM doc_status d
        LEFT JOIN text_chunks c ON c.doc_id = d.doc_id
        GROUP BY d.doc_id, d.file_path, d.status, d.chunks_count
        ORDER BY d.file_path
    """)
    print(f"  {'파일':<45} {'상태':<12} {'청크':>4} {'토큰합':>8}")
    print(f"  {'─'*72}")
    for r in cur.fetchall():
        name = (r["file_path"] or "").split("/")[-1] or r["file_path"] or "-"
        print(f"  {name:<45} {r['status']:<12} {r['actual']:>4} {(r['total_tokens'] or 0):>8,}")
    print()

    # ── Q2. 엔티티 타입 분포 ────────────────────────────
    sep()
    print("Q2. 엔티티 타입 분포")
    sep()
    cur.execute("""
        SELECT entity_type, COUNT(*) AS cnt
        FROM entities
        GROUP BY entity_type
        ORDER BY cnt DESC
    """)
    for r in cur.fetchall():
        bar = "#" * min(r["cnt"], 40)
        print(f"  {r['entity_type']:<20} {r['cnt']:>4}  {bar}")
    print()

    # ── Q3. Qdrant 검색 결과 → 원문 텍스트 조회 ─────────
    sep()
    print("Q3. Qdrant 검색 결과(chunk_id) → 원문 텍스트 조회")
    print("    (Qdrant는 유사도 순위만 반환 → 원문은 여기서)")
    sep()
    # 실제 chunk_id 2개를 가져와서 시뮬레이션
    cur.execute("SELECT chunk_id FROM text_chunks LIMIT 2")
    sample_ids = [r["chunk_id"] for r in cur.fetchall()]
    cur.execute("""
        SELECT chunk_id, chunk_order, tokens,
               LEFT(content, 120) AS preview
        FROM text_chunks
        WHERE chunk_id = ANY(%s)
    """, (sample_ids,))
    for r in cur.fetchall():
        print(f"  chunk_id : {r['chunk_id']}")
        print(f"  order    : {r['chunk_order']}  |  tokens: {r['tokens']}")
        print(f"  내용     : {r['preview']}...")
        print()

    # ── Q4. 특정 엔티티 검색 ────────────────────────────
    sep()
    print("Q4. 엔티티 키워드 검색 (LLM이 추출한 키워드 → DB 탐색)")
    sep()
    cur.execute("""
        SELECT entity_name, entity_type,
               LEFT(description, 120) AS desc_preview,
               ARRAY_LENGTH(source_chunks, 1) AS chunk_refs
        FROM entities
        WHERE entity_name ILIKE ANY(ARRAY['%Lyrica%','%Celebrex%','%기넥신%','%리리카%'])
        ORDER BY entity_name
    """)
    for r in cur.fetchall():
        print(f"  [{r['entity_type']:<12}] {r['entity_name']}")
        print(f"   └ {r['desc_preview']}...")
        print(f"   └ 출처 청크 수: {r['chunk_refs'] or 0}")
        print()

    # ── Q5. 1홉 이웃 탐색 (NetworkX BFS → SQL JOIN) ─────
    sep()
    print("Q5. 1홉 이웃 탐색 — 'Lyrica'와 연결된 엔티티 Top 10")
    print("    (NetworkX 인메모리 BFS 대신 인덱스 기반 JOIN)")
    sep()
    cur.execute("""
        SELECT r.target_entity,
               e.entity_type,
               r.keywords,
               r.weight,
               LEFT(r.description, 80) AS desc_preview
        FROM relations r
        JOIN entities e ON e.entity_name = r.target_entity
        WHERE r.source_entity = 'Lyrica'
        ORDER BY r.weight DESC
        LIMIT 10
    """)
    rows = cur.fetchall()
    if rows:
        for r in rows:
            print(f"  Lyrica → [{r['entity_type']:<12}] {r['target_entity']:<25} (weight={r['weight']:.2f})")
            print(f"    키워드: {r['keywords']}")
            print(f"    설명  : {r['desc_preview']}...")
            print()
    else:
        print("  (결과 없음 — 엔티티명 대소문자 확인 필요)\n")

    # ── Q6. 2홉 재귀 탐색 ───────────────────────────────
    sep()
    print("Q6. 2홉 재귀 탐색 — 'Lyrica' 주변 연결망 (재귀 CTE)")
    print("    PostgreSQL 재귀 CTE = NetworkX 2-hop BFS와 동일한 결과")
    sep()
    cur.execute("""
        WITH RECURSIVE graph(entity, depth, path) AS (
            SELECT 'Lyrica'::TEXT, 0, ARRAY['Lyrica'::TEXT]
            UNION ALL
            SELECT r.target_entity,
                   g.depth + 1,
                   g.path || r.target_entity
            FROM graph g
            JOIN relations r ON r.source_entity = g.entity
            WHERE g.depth < 2
              AND NOT (r.target_entity = ANY(g.path))
        )
        SELECT entity, depth, ARRAY_LENGTH(path, 1) AS path_len
        FROM graph
        ORDER BY depth, entity
        LIMIT 20
    """)
    for r in cur.fetchall():
        indent = "    " * r["depth"]
        print(f"  {indent}[depth={r['depth']}] {r['entity']}")
    print()

    # ── Q7. 관계 weight Top 10 ──────────────────────────
    sep()
    print("Q7. 관계 강도(weight) 상위 10개 엣지")
    sep()
    cur.execute("""
        SELECT source_entity, target_entity, weight, keywords
        FROM relations
        ORDER BY weight DESC
        LIMIT 10
    """)
    print(f"  {'출처':<25} {'대상':<25} {'weight':>6}  키워드")
    print(f"  {'─'*80}")
    for r in cur.fetchall():
        print(f"  {r['source_entity']:<25} {r['target_entity']:<25} {r['weight']:>6.2f}  {r['keywords']}")
    print()

    # ── Q8. 청크 → 엔티티 역추적 ───────────────────────
    sep()
    print("Q8. 특정 청크에서 추출된 엔티티 역추적")
    print("    (Qdrant로 청크 찾은 뒤 → 그 청크 기반 엔티티 목록 조회)")
    sep()
    # 실제 청크 하나 가져오기
    cur.execute("SELECT chunk_id FROM text_chunks ORDER BY chunk_order LIMIT 1")
    row = cur.fetchone()
    if row:
        sample_chunk = row["chunk_id"]
        cur.execute("""
            SELECT entity_name, entity_type,
                   LEFT(description, 80) AS desc_preview
            FROM entities
            WHERE %s = ANY(source_chunks)
            ORDER BY entity_type, entity_name
            LIMIT 15
        """, (sample_chunk,))
        print(f"  청크 ID: {sample_chunk}")
        print(f"  이 청크에서 추출된 엔티티:")
        results = cur.fetchall()
        if results:
            for r in results:
                print(f"    [{r['entity_type']:<12}] {r['entity_name']}")
                print(f"      └ {r['desc_preview']}...")
        else:
            print("    (없음 — source_chunks 매핑 확인 필요)")
    print()

    # ── Q9. 전체 통계 ────────────────────────────────────
    sep()
    print("Q9. 전체 데이터 통계")
    sep()
    cur.execute("""
        SELECT
            (SELECT COUNT(*) FROM doc_status)   AS docs,
            (SELECT COUNT(*) FROM text_chunks)  AS chunks,
            (SELECT COUNT(*) FROM entities)     AS entities,
            (SELECT COUNT(*) FROM relations)    AS relations,
            (SELECT AVG(tokens) FROM text_chunks) AS avg_tokens
    """)
    r = cur.fetchone()
    print(f"  문서        : {r['docs']}개")
    print(f"  청크        : {r['chunks']}개  (Qdrant point 수와 동일)")
    print(f"  엔티티      : {r['entities']}개  (지식 그래프 노드)")
    print(f"  관계        : {r['relations']}개  (지식 그래프 엣지)")
    print(f"  평균 토큰   : {r['avg_tokens']:.0f} tok/청크")
    print()


def sep():
    print("=" * 62)


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
def main():
    print(f"PostgreSQL 연결 중... ({CONN['host']}:{CONN['port']})\n")
    conn = psycopg2.connect(**CONN)
    conn.autocommit = False

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:

            print("── 스키마 생성 (기존 테이블 초기화) ──")
            cur.execute(DDL)
            conn.commit()
            print("  완료\n")

            print("── lightrag_c 실제 데이터 적재 ──")
            load_and_insert(cur)
            conn.commit()

            print("── 쿼리 시연 ──\n")
            run_demo_queries(cur)

    except Exception as e:
        conn.rollback()
        print(f"\n[오류] {e}")
        import traceback; traceback.print_exc()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
