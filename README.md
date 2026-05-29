# RAG 고도화 스터디

RAG(Retrieval-Augmented Generation) 파이프라인을 단계적으로 구축하고 고도화한 과정을 담은 레포지토리입니다.  
벡터 DB 세팅부터 하이브리드 검색, 지식 그래프, 에이전트 오케스트레이션까지 각 레이어를 직접 구현하며 정리했습니다.

## 파이프라인 전체 흐름

```
문서 전처리  →  벡터 DB 저장  →  검색 품질 개선  →  고급 검색  →  에이전트 오케스트레이션
06_Preprocess   01_Weaviate        02_history         03_LightRAG     05_Langgraph
                07_ElasticSearch   improve            04_OpenKB
```

---

## 디렉토리 구조

### `01_Weaviate/`
Weaviate 벡터 DB의 기본 세팅부터 시맨틱 검색까지 구현한 첫 번째 RAG 파이프라인.
[Weaviate 정리 블로그](https://joo9906.tistory.com/61)

- Docker Compose로 Weaviate 서버를 로컬에 구동
- Python 클라이언트(`weaviate-client v4`)로 컬렉션 생성, 문서 삽입, 벡터 검색 구현
- 문서를 임베딩해서 저장하고, 자연어 쿼리로 유사 문서를 검색하는 기본 RAG 흐름 확립
- `chatbot.py`: 검색된 컨텍스트를 LLM에 전달해 답변을 생성하는 챗봇 구현

### `02_history_improve/`
멀티턴 대화에서 발생하는 검색 품질 저하 문제를 히스토리 관리와 리랭킹으로 개선한 실험.

- **히스토리 관리**: 이전 대화 맥락을 현재 쿼리에 반영해 문맥 유지 검색 구현
- **CrossEncoder 리랭킹**: Bi-encoder로 1차 검색한 결과를 CrossEncoder로 재정렬해 정밀도 향상
- **병렬 처리**: 여러 검색 소스를 `asyncio`로 동시 호출해 지연 시간 단축
- 개선 전/후 결과를 실제 쿼리로 비교 분석한 리포트 포함 (`Rerank_개선 리포트.md`)

### `03_LightRAG/`
키워드/벡터 검색의 한계인 "관계 파악" 문제를 지식 그래프로 보완하는 GraphRAG 아키텍처 구현.
[LightRAG 정리 블로그](https://joo9906.tistory.com/114)

- **LightRAG**: 문서에서 엔티티(Entity)와 관계(Relation)를 자동 추출해 그래프로 구성
- 그래프 순회(Graph Traversal) + 벡터 검색을 결합한 하이브리드 검색으로 복잡한 질문 처리
- 대용량 문서 인제스천 시 성능을 측정하는 부하 테스트(`load_test.py`) 포함
- Docker로 Neo4j 등 그래프 DB와 연동하는 환경 구성

### `04_OpenKB/`
대량의 문서를 입력받아 서로 연결된 위키 형태의 지식 베이스를 자동으로 생성하는 파이프라인.
[OpenKB 정리 블로그](https://joo9906.tistory.com/115)

- 문서별 요약, 핵심 개념 추출, 문서 간 연결 관계를 LLM으로 자동화
- Obsidian 등 마크다운 뷰어와 바로 연동 가능한 구조로 출력
- 생성된 위키(`wiki/`)는 개념별, 출처별, 요약별로 계층 구조화

### `05_Langgraph/`
단순 검색-응답 구조를 넘어 조건 분기와 루프가 있는 복잡한 RAG 에이전트를 LangGraph로 구현.

- 노드(Node)와 엣지(Edge)로 에이전트 흐름을 그래프로 명시적으로 정의
- 검색 결과 품질을 판단해 재검색 여부를 결정하는 Self-RAG 패턴 실험
- 상태(State)를 기반으로 멀티스텝 추론 흐름 관리

### `06_Preprocess/`
RAG 품질의 기반이 되는 문서 전처리 모듈을 재사용 가능한 라이브러리 형태로 구성.

- PDF, Word, 마크다운, 텍스트 등 다양한 포맷 파싱 (`docling` 활용)
- 청킹 전략 구현: 고정 크기, 문장 기반, 의미 기반(Semantic Chunking)
- 청크 간 오버랩 조정, 메타데이터 보존 등 검색 품질에 직결되는 전처리 옵션 제공

### `07_ElasticSearch/`
키워드 검색과 벡터 검색을 ES 내부에서 결합하는 하이브리드 검색 파이프라인 구현.

- **Nori 형태소 분석기**: 한국어 BM25 키워드 검색을 위한 Nori 분석기 인덱스 설정
- **dense_vector + kNN**: 문서 임베딩을 ES에 저장하고 ANN(Approximate Nearest Neighbor) 검색 구현
- **RRF (Reciprocal Rank Fusion)**: ES 8.9+의 `rank.rrf`를 활용해 BM25 순위와 kNN 순위를 공식으로 합산
  ```
  RRF score = Σ 1 / (rank_constant + rank_i)   # rank_constant=60 권장
  ```
- **멀티소스 파이프라인**: ES 하이브리드 결과 + Weaviate BM25 결과를 병합 후 CrossEncoder 리랭킹 → LLM 응답
- **LangSmith 트레이싱**: `@traceable` 데코레이터로 RAG 전 구간(검색 → 리랭킹 → LLM) 추적

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| 벡터 DB | Weaviate, Elasticsearch (`dense_vector`) |
| 검색 | BM25 (Nori), kNN ANN, Hybrid RRF |
| 리랭킹 | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| 임베딩 | `paraphrase-multilingual-MiniLM-L12-v2` (384차원, 한/영) |
| 그래프 RAG | LightRAG, OpenKB |
| 오케스트레이션 | LangGraph |
| 문서 파싱 | Docling |
| LLM | OpenAI GPT-4o-mini |
| 트레이싱 | LangSmith |
| 인프라 | Docker Compose |
