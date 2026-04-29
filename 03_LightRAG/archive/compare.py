"""
compare.py
==========
기존 방식(NetworkX + 파일 JSON) vs PostgreSQL 방식의 성능을 비교하고
결과를 result_compare.md 에 저장한다.

비교 항목
  1. 초기화 시간  — GraphML+JSON 전체 로딩 vs PostgreSQL 연결
  2. 엔티티 조회  — dict 탐색 vs SELECT WHERE
  3. 타입 필터    — 전체 스캔 vs 인덱스 스캔
  4. 1홉 탐색    — 인메모리 adjacency vs JOIN
  5. 2홉 탐색    — BFS vs 재귀 CTE
  6. 청크 조회   — dict 탐색 vs SELECT WHERE
  7. 역추적      — 전체 스캔 vs ANY(array) 인덱스
  8. 메모리 점유 — 로딩 후 RSS 증가량

실행 방법
  python compare.py
"""

import gc
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict, deque

import networkx as nx
import psutil
import psycopg2
from psycopg2.extras import RealDictCursor

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────
# 경로 / 설정
# ─────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_LC   = os.path.join(_BASE, "lightrag_c")

GRAPHML_PATH    = os.path.join(_LC, "graph_chunk_entity_relation.graphml")
CHUNKS_PATH     = os.path.join(_LC, "kv_store_text_chunks.json")
DOC_STATUS_PATH = os.path.join(_LC, "kv_store_doc_status.json")
OUTPUT_MD       = os.path.join(_BASE, "result_compare.md")

PG_CONN = dict(host="localhost", port=5432, dbname="postgres",
               user="postgres", password="postgres")

REPEAT      = 200    # 연산별 반복 횟수
TEST_ENTITY = "Lyrica"
NS          = "http://graphml.graphdrawing.org/xmlns"


# ─────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────
def tag(name):
    return f"{{{NS}}}{name}"


def bench(fn, n=REPEAT):
    """fn 을 n 회 실행해 (avg_ms, min_ms, max_ms) 반환."""
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return sum(times) / len(times), min(times), max(times)


def rss_mb():
    return psutil.Process().memory_info().rss / 1024 / 1024


# ─────────────────────────────────────────
# ── 접근 방식 A: NetworkX + 파일 JSON ────
# ─────────────────────────────────────────
class FileBackend:
    """GraphML + JSON KV 파일 기반 접근."""

    def __init__(self):
        self.G: nx.Graph = None
        self.chunks: dict = {}
        self.chunk_to_entities: dict = defaultdict(list)  # {chunk_id: [entity_name]}

    def load(self):
        """GraphML + 청크 JSON 전체 로딩."""
        # 1) GraphML → NetworkX 그래프
        self.G = nx.read_graphml(GRAPHML_PATH)

        # 2) 청크 JSON
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            self.chunks = json.load(f)

        # 3) 청크→엔티티 역인덱스 구성
        for nid, attrs in self.G.nodes(data=True):
            src_raw = attrs.get("source_id", "")
            for cid in src_raw.split("<SEP>"):
                cid = cid.strip()
                if cid:
                    self.chunk_to_entities[cid].append(nid)

    # ── 벤치 대상 연산들 ──────────────────
    def entity_lookup(self, name=TEST_ENTITY):
        return self.G.nodes.get(name)

    def entity_type_filter(self, etype="drug"):
        return [n for n, d in self.G.nodes(data=True) if d.get("entity_type") == etype]

    def hop1(self, name=TEST_ENTITY):
        return list(self.G.neighbors(name))

    def hop2(self, name=TEST_ENTITY):
        visited, result = {name}, []
        queue = deque([(name, 0)])
        while queue:
            node, depth = queue.popleft()
            result.append((node, depth))
            if depth < 2:
                for nb in self.G.neighbors(node):
                    if nb not in visited:
                        visited.add(nb)
                        queue.append((nb, depth + 1))
        return result

    def chunk_get(self, chunk_id):
        return self.chunks.get(chunk_id, {}).get("content", "")

    def chunk_entities(self, chunk_id):
        return self.chunk_to_entities.get(chunk_id, [])

    def full_pipeline(self, name=TEST_ENTITY):
        """엔티티 조회 → 1홉 탐색 → 청크 원문 수집"""
        entity = self.entity_lookup(name)
        neighbors = self.hop1(name)
        src_raw = self.G.nodes[name].get("source_id", "") if entity else ""
        chunk_ids = [c.strip() for c in src_raw.split("<SEP>") if c.strip()]
        texts = [self.chunk_get(cid) for cid in chunk_ids]
        return entity, neighbors, texts


# ─────────────────────────────────────────
# ── 접근 방식 B: PostgreSQL ──────────────
# ─────────────────────────────────────────
class PGBackend:
    """PostgreSQL 기반 접근."""

    def __init__(self):
        self.conn = None
        self._sample_chunk_id = None

    def load(self):
        self.conn = psycopg2.connect(**PG_CONN)
        self.conn.autocommit = True

    def _cur(self):
        return self.conn.cursor(cursor_factory=RealDictCursor)

    def entity_lookup(self, name=TEST_ENTITY):
        with self._cur() as cur:
            cur.execute(
                "SELECT entity_name, entity_type, description FROM entities WHERE entity_name = %s",
                (name,),
            )
            return cur.fetchone()

    def entity_type_filter(self, etype="drug"):
        with self._cur() as cur:
            cur.execute("SELECT entity_name FROM entities WHERE entity_type = %s", (etype,))
            return cur.fetchall()

    def hop1(self, name=TEST_ENTITY):
        with self._cur() as cur:
            cur.execute(
                "SELECT target_entity FROM relations WHERE source_entity = %s", (name,)
            )
            return [r["target_entity"] for r in cur.fetchall()]

    def hop2(self, name=TEST_ENTITY):
        with self._cur() as cur:
            cur.execute(
                """
                WITH RECURSIVE g(entity, depth, path) AS (
                    SELECT %s::TEXT, 0, ARRAY[%s::TEXT]
                    UNION ALL
                    SELECT r.target_entity, g.depth + 1, g.path || r.target_entity
                    FROM g JOIN relations r ON r.source_entity = g.entity
                    WHERE g.depth < 2 AND NOT (r.target_entity = ANY(g.path))
                )
                SELECT entity, depth FROM g ORDER BY depth
                """,
                (name, name),
            )
            return cur.fetchall()

    def chunk_get(self, chunk_id):
        with self._cur() as cur:
            cur.execute("SELECT content FROM text_chunks WHERE chunk_id = %s", (chunk_id,))
            row = cur.fetchone()
            return row["content"] if row else ""

    def chunk_entities(self, chunk_id):
        with self._cur() as cur:
            cur.execute(
                "SELECT entity_name FROM entities WHERE %s = ANY(source_chunks)", (chunk_id,)
            )
            return [r["entity_name"] for r in cur.fetchall()]

    def full_pipeline(self, name=TEST_ENTITY):
        with self._cur() as cur:
            cur.execute(
                "SELECT entity_name, entity_type, description, source_chunks FROM entities WHERE entity_name = %s",
                (name,),
            )
            entity = cur.fetchone()
            cur.execute(
                "SELECT target_entity FROM relations WHERE source_entity = %s", (name,)
            )
            neighbors = [r["target_entity"] for r in cur.fetchall()]
            chunk_ids = entity["source_chunks"] if entity and entity["source_chunks"] else []
            if chunk_ids:
                cur.execute("SELECT content FROM text_chunks WHERE chunk_id = ANY(%s)", (chunk_ids,))
                texts = [r["content"] for r in cur.fetchall()]
            else:
                texts = []
        return entity, neighbors, texts


# ─────────────────────────────────────────
# 공통 샘플 데이터 준비
# ─────────────────────────────────────────
def get_sample_chunk_id():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    return list(chunks.keys())[0]


# ─────────────────────────────────────────
# 벤치마크 실행
# ─────────────────────────────────────────
def run_benchmarks():
    sample_chunk = get_sample_chunk_id()
    results = {}

    # ── A. 파일 기반 ─────────────────────
    print("[A] 파일 기반(NetworkX) 초기화 중...")
    gc.collect()
    rss_before_a = rss_mb()
    t0 = time.perf_counter()
    fb = FileBackend()
    fb.load()
    init_a_ms = (time.perf_counter() - t0) * 1000
    rss_after_a = rss_mb()
    mem_a = rss_after_a - rss_before_a

    print(f"    초기화 완료 ({init_a_ms:.0f}ms, +{mem_a:.1f}MB)")

    ops_a = {
        "엔티티 조회":   lambda: fb.entity_lookup(),
        "타입 필터":     lambda: fb.entity_type_filter(),
        "1홉 탐색":      lambda: fb.hop1(),
        "2홉 탐색":      lambda: fb.hop2(),
        "청크 조회":     lambda: fb.chunk_get(sample_chunk),
        "청크→엔티티":  lambda: fb.chunk_entities(sample_chunk),
        "전체 파이프라인": lambda: fb.full_pipeline(),
    }

    bench_a = {}
    for name, fn in ops_a.items():
        avg, lo, hi = bench(fn)
        bench_a[name] = (avg, lo, hi)
        print(f"    {name:<16} avg={avg:.3f}ms  min={lo:.3f}ms  max={hi:.3f}ms")

    # ── B. PostgreSQL ─────────────────────
    print("\n[B] PostgreSQL 초기화 중...")
    gc.collect()
    rss_before_b = rss_mb()
    t0 = time.perf_counter()
    pg = PGBackend()
    pg.load()
    init_b_ms = (time.perf_counter() - t0) * 1000
    rss_after_b = rss_mb()
    mem_b = rss_after_b - rss_before_b

    print(f"    초기화 완료 ({init_b_ms:.0f}ms, +{mem_b:.1f}MB)")

    ops_b = {
        "엔티티 조회":   lambda: pg.entity_lookup(),
        "타입 필터":     lambda: pg.entity_type_filter(),
        "1홉 탐색":      lambda: pg.hop1(),
        "2홉 탐색":      lambda: pg.hop2(),
        "청크 조회":     lambda: pg.chunk_get(sample_chunk),
        "청크→엔티티":  lambda: pg.chunk_entities(sample_chunk),
        "전체 파이프라인": lambda: pg.full_pipeline(),
    }

    bench_b = {}
    for name, fn in ops_b.items():
        avg, lo, hi = bench(fn)
        bench_b[name] = (avg, lo, hi)
        print(f"    {name:<16} avg={avg:.3f}ms  min={lo:.3f}ms  max={hi:.3f}ms")

    return {
        "init_a_ms": init_a_ms, "mem_a": mem_a,
        "init_b_ms": init_b_ms, "mem_b": mem_b,
        "bench_a": bench_a, "bench_b": bench_b,
        "sample_chunk": sample_chunk,
        "nodes": fb.G.number_of_nodes(),
        "edges": fb.G.number_of_edges(),
        "chunks": len(fb.chunks),
    }


# ─────────────────────────────────────────
# Markdown 출력
# ─────────────────────────────────────────
def winner(a_ms, b_ms):
    if a_ms < b_ms * 0.9:
        return "A"
    elif b_ms < a_ms * 0.9:
        return "B"
    return "-"


def speedup(a_ms, b_ms):
    if b_ms == 0:
        return "-"
    ratio = a_ms / b_ms
    if ratio >= 1.1:
        return f"B가 {ratio:.1f}x 빠름"
    elif ratio <= 0.9:
        return f"A가 {1/ratio:.1f}x 빠름"
    return "비슷"


def write_markdown(r):
    a = r["bench_a"]
    b = r["bench_b"]
    ops = list(a.keys())

    from datetime import date
    today = date.today().strftime("%Y-%m-%d")

    lines = []
    def w(*args):
        lines.append(" ".join(str(x) for x in args))

    w(f"# LightRAG 스토리지 방식 비교: NetworkX(파일) vs PostgreSQL")
    w()
    w(f"측정일: {today}  |  반복 횟수: {REPEAT}회  |  대상 엔티티: `{TEST_ENTITY}`")
    w()
    w("## 데이터셋 규모")
    w()
    w("| 항목 | 수량 |")
    w("|------|------|")
    w(f"| 엔티티(노드) | {r['nodes']:,}개 |")
    w(f"| 관계(엣지) | {r['edges']:,}개 |")
    w(f"| 텍스트 청크 | {r['chunks']:,}개 |")
    w(f"| 샘플 청크 ID | `{r['sample_chunk']}` |")
    w()
    w("---")
    w()
    w("## 1. 초기화 비용")
    w()
    w("| 구분 | A. NetworkX + 파일 | B. PostgreSQL |")
    w("|------|-------------------|---------------|")
    w(f"| 초기화 시간 | {r['init_a_ms']:.0f} ms | {r['init_b_ms']:.0f} ms |")
    w(f"| 메모리 증가 | +{r['mem_a']:.1f} MB | +{r['mem_b']:.1f} MB |")
    w()
    w("> **A 방식**: GraphML(XML) 전체 파싱 → NetworkX 그래프 객체 + JSON 파일 읽기를 포함한 시간  ")
    w("> **B 방식**: TCP 소켓 연결 + 인증만 포함 (데이터는 이미 DB에 있음)")
    w()
    w("---")
    w()
    w("## 2. 연산별 응답 시간 (단위: ms, 평균/최솟값/최댓값)")
    w()
    w("| 연산 | A avg | A min | B avg | B min | 승자 | 배속 |")
    w("|------|------:|------:|------:|------:|------|------|")
    for op in ops:
        a_avg, a_min, a_max = a[op]
        b_avg, b_min, b_max = b[op]
        w = winner(a_avg, b_avg)
        sp = speedup(a_avg, b_avg)
        lines.append(
            f"| {op} | {a_avg:.3f} | {a_min:.3f} | {b_avg:.3f} | {b_min:.3f} | {w} | {sp} |"
        )
    w = lines.append  # reassign

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. 항목별 분석")
    lines.append("")

    analyses = {
        "엔티티 조회": (
            "딕셔너리 O(1) 해시 탐색",
            "B-tree 인덱스 O(log n) 탐색 + 네트워크 왕복",
            "그래프가 인메모리에 있으면 A가 유리하나, 규모가 커지면 메모리 부족으로 A 자체가 불가능해진다."
        ),
        "타입 필터": (
            "전체 노드 선형 스캔 O(n)",
            "entity_type 인덱스 스캔 O(k)",
            "엔티티 수가 많을수록 B의 인덱스 효과가 극대화된다. 수만 노드 이상에서 역전이 뚜렷해진다."
        ),
        "1홉 탐색": (
            "adjacency list 직접 접근 O(degree)",
            "source_entity 인덱스 JOIN O(log n + degree)",
            "인메모리라 A가 절대적으로 빠르다. 단, 그래프 로딩(수십초) 비용은 B에 없다."
        ),
        "2홉 탐색": (
            "BFS deque — O(V+E) 범위 내",
            "재귀 CTE — 인덱스 기반 레벨별 JOIN",
            "현재 규모(567 노드)에서는 A가 빠르다. 수만 노드 이상이면 인메모리 BFS가 메모리 한계에 걸린다."
        ),
        "청크 조회": (
            "딕셔너리 O(1) 해시",
            "PK 인덱스 O(log n)",
            "단순 조회는 인메모리 A가 항상 빠르다. 단, 대용량에서는 RAM 부족으로 A 사용 불가."
        ),
        "청크→엔티티": (
            "사전 구성된 역인덱스 dict O(1)",
            "GIN 인덱스 없이 ANY(array) 스캔",
            "GIN 인덱스를 추가하면 B가 대용량에서 A보다 빠를 수 있다. 현재는 인메모리 역인덱스 A가 유리."
        ),
        "전체 파이프라인": (
            "엔티티 조회 + 1홉 + 청크 수집 (모두 인메모리)",
            "3개 쿼리 순차 실행 (각각 네트워크 왕복)",
            "단일 사용자 기준 A가 빠르다. 동시 요청 N명 시 A는 메모리 N배 필요, B는 커넥션 풀로 공유."
        ),
    }

    for op, (desc_a, desc_b, comment) in analyses.items():
        if op not in a:
            continue
        a_avg = a[op][0]
        b_avg = b[op][0]
        lines.append(f"### {op}")
        lines.append("")
        lines.append(f"- **A 방식**: {desc_a}")
        lines.append(f"- **B 방식**: {desc_b}")
        lines.append(f"- **해석**: {comment}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## 4. 규모별 예상 특성 변화")
    lines.append("")
    lines.append("| 규모 | 노드 수 | A(NetworkX) | B(PostgreSQL) |")
    lines.append("|------|--------|-------------|---------------|")
    lines.append("| 현재 (소규모) | ~567 | 로딩 후 빠름 | 매 쿼리 네트워크 왕복 |")
    lines.append("| 중규모 | ~3만 | GraphML 로딩 30초+, RAM 1GB+ | 응답 일정 유지 |")
    lines.append("| 대규모 | ~10만 | RAM 3GB+, 로딩 수분 | 인덱스로 일정 유지 |")
    lines.append("| 초대규모 | ~50만+ | 사실상 불가 | 샤딩 검토 필요 |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. 결론")
    lines.append("")
    lines.append("| 기준 | 추천 방식 |")
    lines.append("|------|-----------|")
    lines.append("| 단일 사용자, 소규모 데이터 | A (NetworkX) |")
    lines.append("| 동시 사용자 다수 | B (PostgreSQL) |")
    lines.append("| 노드 1만개 이상 | B (PostgreSQL) |")
    lines.append("| 인프라 단순화 우선 | A (파일 기반) |")
    lines.append("| 장기 운영, 안정성 우선 | B (PostgreSQL) |")
    lines.append("")
    lines.append("> **현재 프로젝트 권장**: 문서 수가 수백 건 수준이면 NetworkX로 충분하다.  ")
    lines.append("> 문서가 수천 건을 넘거나 동시 사용자가 생기면 PostgreSQL 전환을 검토한다.")
    lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────
# 메인
# ─────────────────────────────────────────
def main():
    print("=" * 60)
    print("LightRAG 스토리지 방식 벤치마크")
    print(f"  데이터: {_LC}")
    print(f"  반복:   {REPEAT}회")
    print("=" * 60 + "\n")

    results = run_benchmarks()

    print(f"\n결과를 {OUTPUT_MD} 에 저장 중...")
    md = write_markdown(results)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"완료: {OUTPUT_MD}")


if __name__ == "__main__":
    main()
