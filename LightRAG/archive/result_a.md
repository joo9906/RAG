LightRAG 지식그래프 구축 시작

INFO: [] Loaded graph from C:\Users\happy\OneDrive\바탕 화면\labq\lightrag*kg\graph_chunk_entity_relation.graphml with 662 nodes, 501 edges
WARNING: Qdrant collection: lightrag_vdb_entities missing suffix. Pls add model_name to embedding_func for proper workspace data isolation.
WARNING: Qdrant collection: lightrag_vdb_relationships missing suffix. Pls add model_name to embedding_func for proper workspace data isolation.
WARNING: Qdrant collection: lightrag_vdb_chunks missing suffix. Pls add model_name to embedding_func for proper workspace data isolation.
INFO: [] Process 3304 KV load full_docs with 154 records
INFO: [] Process 3304 KV load text_chunks with 52 records
INFO: [] Process 3304 KV load full_entities with 6 records
INFO: [] Process 3304 KV load full_relations with 6 records
INFO: [] Process 3304 KV load entity_chunks with 662 records
INFO: [] Process 3304 KV load relation_chunks with 501 records
C:\Users\happy\OneDrive\바탕 화면\skch-aix-ce\venv\Lib\site-packages\lightrag\kg\qdrant_impl.py:583: UserWarning: Qdrant client version 1.15.1 is incompatible with server version 1.17.1. Major versions should match and minor version difference must not exceed 1. Set check_compatibility=False to skip version check.
self.\_client = QdrantClient(
INFO: Qdrant: Found legacy collection 'lightrag_vdb_entities' (namespace=entities, workspace=*)
INFO: [] Qdrant collection 'entities' initialized successfully
INFO: Qdrant: Found legacy collection 'lightrag*vdb_relationships' (namespace=relationships, workspace=*)
INFO: [] Qdrant collection 'relationships' initialized successfully
INFO: Qdrant: Found legacy collection 'lightrag*vdb_chunks' (namespace=chunks, workspace=*)
INFO: [] Qdrant collection 'chunks' initialized successfully
INFO: [] Process 3304 KV load llm_response_cache with 110 records
INFO: [] Process 3304 doc status load doc_status with 154 records
[시각화] GraphML 로딩: C:\Users\happy\OneDrive\바탕 화면\labq\lightrag_kg\graph_chunk_entity_relation.graphml
노드 수: 662, 엣지 수: 501
노드가 200개 초과 → 연결수 상위 200개 노드만 표시
✅ 시각화 완료 → C:\Users\happy\OneDrive\바탕 화면\labq\lightrag_kg\knowledge_graph.html
브라우저에서 위 파일을 열면 인터랙티브 지식그래프를 확인할 수 있습니다.

============================================================
[쿼리] 리리카와 쎄레브렉스의 안전성 정보를 비교해줘
============================================================
INFO: Embedding func: 8 new workers initialized (Timeouts: Func: 30s, Worker: 60s, Health Check: 75s)
INFO: Query nodes: 부작용, 사용 권장 사항, 임상 시험, 약리학적 특성 (top_k:40, cosine:0.2)
WARNING: Some nodes are missing, maybe the storage is damaged
INFO: Local query: 11 entites, 9 relations
INFO: Query edges: 리리카, 쎄레브렉스, 안전성 정보, 약물 비교 (top_k:40, cosine:0.2)
INFO: Global query: 12 entites, 8 relations
INFO: Raw search results: 22 entities, 16 relations, 0 vector chunks
INFO: After truncation: 22 entities, 16 relations
INFO: Selecting 14 from 14 entity-related chunks by vector similarity
INFO: Find no additional relations-related chunks from 16 relations
INFO: Round-robin merged chunks: 14 -> 14 (deduplicated 0)
WARNING: Rerank is enabled but no rerank model is configured. Please set up a rerank model or set enable_rerank=False in query parameters.
INFO: Final context: 22 entities, 16 relations, 14 chunks
INFO: Final chunks S+F/O: E2/1 E2/2 E2/3 E6/4 E1/5 E1/6 E2/7 E2/8 E2/9 E1/10 E4/11 E1/12 E3/13 E1/14
INFO: == LLM cache == Query cache hit, using cached response as query result
리리카와 쎄레브렉스의 안전성 정보를 비교하기 위해 각 약물의 특징 및 보고된 이상반응을 살펴보겠습니다.

### 리리카 (Pregabalin)

- **주요 사용 용도**: 신경병증성 통증, 뇌전증 보조요법, 섬유근육통 등.
- **보고된 이상반응**:
  - 일반적으로 보고되는 이상반응으로는 **졸음, 어지러움, 그리고 불면증**이 포함됩니다.
  - 약물 사용 시 **간 기능 지표**인 ALT, AST, ALP의 수치 상승이 동반될 수 있으며, 이는 간장애 환자에게 신중해야 함을 나타냅니다.
- **임부 및 수유부**: 임산부에게는 권장되지 않으며, 수유 중일 경우 수유를 피해야 합니다.

### 쎄레브렉스 (Celecoxib)

- **주요 사용 용도**: 염증 및 통증 완화, 주로 류머티즘 관절염 및 골관절염 환자에서 사용.
- **보고된 이상반응**:
  - 통상적으로 보고되는 이상반응으로는 **소화 불량, 구역, 그리고 혈압 상승** 등이 있습니다.
  - 사용 중 **위장관 출혈**의 위험이 있으며, 이는 NSAIDs 약물의 공통된 부작용입니다.
- **임부 및 수유부**: 임신 가능성 있는 여성 및 수유부에게는 주의가 필요합니다.

### 비교 요약

- **졸음 및 기타 CNS 관련 부작용**: 리리카는 졸음 및 어지러움을 유발할 가능성이 높으며, 이는 사용 중 환자에게 영향이 크다.
- **소화기계 이상반응**: 쎄레브렉스는 소화기계에서의 부작용이 더 빈번하며, 위장관 출혈의 위험이 있다.
- **간 기능 지표**: 리리카는 간 기능 지표에 영향을 줄 수 있으므로 간장애 환자에 대한 주의가 필요하다.

이런 내용을 바탕으로, 두 약물의 사용 시 각각의 안전성 정보를 고려해야 하며, 특히 환자의 상태와 위험 요소를 평가하는 것이 중요합니다. 리리카는 중추신경 계 이상반응, 반면 쎄레브렉스는 소화기계의 부작용이 더 주목받고 있습니다.

### References

- [1] 리리카 치료 정보
- [2] 쎄레브렉스 치료 정보

============================================================
[쿼리] 쎄레브렉스의 주요 상병코드는 무엇인가?
============================================================
INFO: Query nodes: 약물, 질병코드, 의약품, 세레브렉스, 치료방법 (top_k:40, cosine:0.2)
WARNING: Some nodes are missing, maybe the storage is damaged
INFO: Local query: 11 entites, 21 relations
INFO: Query edges: 쎄레브렉스, 상병코드, 주요정보 (top_k:40, cosine:0.2)
INFO: Global query: 12 entites, 9 relations
INFO: Raw search results: 21 entities, 27 relations, 0 vector chunks
INFO: After truncation: 21 entities, 27 relations
INFO: Selecting 12 from 12 entity-related chunks by vector similarity
INFO: Find no additional relations-related chunks from 27 relations
INFO: Round-robin merged chunks: 12 -> 12 (deduplicated 0)
WARNING: Rerank is enabled but no rerank model is configured. Please set up a rerank model or set enable_rerank=False in query parameters.
INFO: Final context: 21 entities, 27 relations, 12 chunks
INFO: Final chunks S+F/O: E4/1 E3/2 E3/3 E4/4 E1/5 E3/6 E3/7 E1/8 E1/9 E1/10 E2/11 E1/12
INFO: == LLM cache == Query cache hit, using cached response as query result
쎄레브렉스(세레브렉스)의 주요 상병코드는 다음과 같습니다:

- **R52.0 급성 통증 (Acute Pain)**: 성인의 급성 통증 완화를 위해 사용됩니다. 이 코드는 수술 후 통증이나 발치 후 통증 등의 치료와 관련이 있습니다.
- **M05–06 류마티스관절염 (Rheumatoid arthritis)**: 류마티스관절염의 치료에서도 사용되며, 이와 관련된 여러 하위 코드가 있습니다.

또한, 이와 관련된 정보는 리리카와 함께 통증 치료에서의 쎄레브렉스의 사용과 연관되어 있습니다. 쎄레브렉스는 이러한 상병코드에 따라 요양급여를 인정받기 위한 기준으로 활용됩니다.

### References

- [1] PAIN TREATMENT: LYRICA & CELEBREX IN PROGRESS
- [2] 쎄레브렉스 안전성 정보 요약

============================================================
[쿼리] 류마티스관절염 관련 상병코드를 알려줘
============================================================
INFO: Query nodes: Rheumatoid arthritis, ICD codes, 상병, 의학 (top_k:40, cosine:0.2)
WARNING: Some nodes are missing, maybe the storage is damaged
INFO: Local query: 10 entites, 13 relations
INFO: Query edges: 류마티스관절염, 상병코드, 의학적 정보 (top_k:40, cosine:0.2)
INFO: Global query: 0 entites, 0 relations
INFO: Raw search results: 10 entities, 13 relations, 0 vector chunks
INFO: After truncation: 10 entities, 13 relations
INFO: Selecting 4 from 4 entity-related chunks by vector similarity
INFO: Find no additional relations-related chunks from 13 relations
INFO: Round-robin merged chunks: 4 -> 4 (deduplicated 0)
WARNING: Rerank is enabled but no rerank model is configured. Please set up a rerank model or set enable_rerank=False in query parameters.
INFO: Final context: 10 entities, 13 relations, 4 chunks
INFO: Final chunks S+F/O: E2/1 E4/2 E4/3 E2/4
INFO: == LLM cache == Query cache hit, using cached response as query result
류마티스관절염은 주로 두 가지 유형으로 분류되며, 해당 상병코드는 다음과 같습니다:

- **M05 혈청검사양성 류마티스관절염** (Seropositive rheumatoid arthritis)
  - M05.80–05.89: 기타 혈청검사양성 류마티스관절염 (Other seropositive rheumatoid arthritis)
  - M05.90–05.99: 상세불명의 혈청검사양성 류마티스관절염 (Seropositive rheumatoid arthritis, unspecified)

- **M06 기타 류마티스관절염** (Other rheumatoid arthritis)
  - M06.00–06.09: 혈청검사음성 류마티스관절염 (Seronegative rheumatoid arthritis)
  - M06.80–06.89: 기타 명시된 류마티스관절염 (Other specified rheumatoid arthritis)
  - M06.90–06.99: 상세불명의 류마티스관절염 (Rheumatoid arthritis, unspecified)

이 상병코드는 류마티스관절염의 진단 및 치료에 활용됩니다.

### References

- [1] PAIN TREATMENT: LYRICA & CELEBREX IN PROGRESS
