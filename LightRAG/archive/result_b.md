🚀 LightRAG (b.py) 시작

[전처리 생략] C:\Users\happy\OneDrive\바탕 화면\labq\parsed_output.json 이미 존재

INFO: [] Loaded graph from C:\Users\happy\OneDrive\바탕 화면\labq\lightrag*new\graph_chunk_entity_relation.graphml with 990 nodes, 884 edges
WARNING: Qdrant collection: lightrag_vdb_entities missing suffix. Pls add model_name to embedding_func for proper workspace data isolation.
WARNING: Qdrant collection: lightrag_vdb_relationships missing suffix. Pls add model_name to embedding_func for proper workspace data isolation.
WARNING: Qdrant collection: lightrag_vdb_chunks missing suffix. Pls add model_name to embedding_func for proper workspace data isolation.
INFO: [] Process 3016 KV load full_docs with 116 records
INFO: [] Process 3016 KV load text_chunks with 127 records
INFO: [] Process 3016 KV load full_entities with 116 records
INFO: [] Process 3016 KV load full_relations with 112 records
INFO: [] Process 3016 KV load entity_chunks with 990 records
INFO: [] Process 3016 KV load relation_chunks with 884 records
C:\Users\happy\OneDrive\바탕 화면\skch-aix-ce\venv\Lib\site-packages\lightrag\kg\qdrant_impl.py:583: UserWarning: Qdrant client version 1.15.1 is incompatible with server version 1.17.1. Major versions should match and minor version difference must not exceed 1. Set check_compatibility=False to skip version check.
self.\_client = QdrantClient(
INFO: Qdrant: Found legacy collection 'lightrag_vdb_entities' (namespace=entities, workspace=*)
INFO: [] Qdrant collection 'entities' initialized successfully
INFO: Qdrant: Found legacy collection 'lightrag*vdb_relationships' (namespace=relationships, workspace=*)
INFO: [] Qdrant collection 'relationships' initialized successfully
INFO: Qdrant: Found legacy collection 'lightrag*vdb_chunks' (namespace=chunks, workspace=*)
INFO: [] Qdrant collection 'chunks' initialized successfully
INFO: [] Process 3016 KV load llm_response_cache with 269 records
INFO: [] Process 3016 doc status load doc_status with 116 records
=== [시각화] ===
[시각화] GraphML 로딩: C:\Users\happy\OneDrive\바탕 화면\labq\lightrag_new\graph_chunk_entity_relation.graphml
노드: 990, 엣지: 884
노드 200개 초과 → 연결수 상위 200개만 표시
✅ 시각화 완료 → C:\Users\happy\OneDrive\바탕 화면\labq\lightrag_new\knowledge_graph.html
[시간] 시각화: 0.2초

=== [쿼리] ===

============================================================
[쿼리] 리리카와 쎄레브렉스의 안전성 정보를 비교해줘
============================================================
INFO: Embedding func: 8 new workers initialized (Timeouts: Func: 30s, Worker: 60s, Health Check: 75s)
INFO: Query nodes: Lyrica, Celebrex, Side effects, Drug interactions, Patient safety (top_k:40, cosine:0.2)
WARNING: Some nodes are missing, maybe the storage is damaged
INFO: Local query: 23 entites, 47 relations
INFO: Query edges: Lyrica, Celebrex, Safety information comparison (top_k:40, cosine:0.2)
INFO: Global query: 22 entites, 21 relations
INFO: Raw search results: 37 entities, 53 relations, 0 vector chunks
INFO: After truncation: 37 entities, 53 relations
INFO: Selecting 25 from 25 entity-related chunks by vector similarity
INFO: Find no additional relations-related chunks from 53 relations
INFO: Round-robin merged chunks: 25 -> 25 (deduplicated 0)
WARNING: Rerank is enabled but no rerank model is configured. Please set up a rerank model or set enable_rerank=False in query parameters.
INFO: Final context: 37 entities, 53 relations, 20 chunks
INFO: Final chunks S+F/O: E5/1 E1/2 E5/3 E4/4 E6/5 E1/6 E1/7 E3/8 E2/9 E3/10 E2/11 E1/12 E3/13 E1/14 E1/15 E4/16 E2/17 E1/18 E2/19 E1/20
INFO: == LLM cache == Query cache hit, using cached response as query result
리리카(물리적 이름: Pregabalin)와 쎄레브렉스(Celecoxib)의 안전성 정보를 비교하면 다음과 같은 주요 사항이 있습니다.

### 리리카의 안전성 정보

- **가장 흔한 이상반응**: 리리카의 부작용으로는 어지러움과 졸음이 가장 흔하게 보고되며, 이러한 이상반응은 대개 경증 또는 중등증으로 나타납니다. (출처: [20])
- **사용 식이**: 리리카는 주로 섬유근육통, 신경통, 만성 통증 및 간질 부가치료에 사용됩니다.

### 쎄레브렉스의 안전성 정보

- **위험 요소**: 쎄레브렉스는 위장관계에 대한 위험이 있으며, 저용량 아스피린과 병용 시 위장관계 이상반응이 증가할 수 있습니다. 또한 심혈관계 위험이 있 으며, 중대한 심혈관계 혈전 반응, 심근경색증 및 뇌졸중의 위험을 증가시킬 수 있습니다. 이러한 반응은 치명적일 수 있습니다. (출처: [20])
- **이상반응 보고**: 또한 쎄레브렉스는 위장관계 이상반응 및 심혈관계 문제와 밀접한 연관이 있습니다.

### 요약

- **리리카**는 주로 CNS(중추신경계) 부작용(어지러움 및 졸음)가 주로 발생하는 반면, **쎄레브렉스**는 심혈관계와 위장관계와 관련된 위험이 주된 문제가 됩 니다. 두 약물 모두 특정 환자군에서 주의가 필요하지만, 그 부작용의 성격과 위험도는 다릅니다.

이처럼, 두 약물은 사용의 적응증뿐만 아니라, 안전성 프로필에서도 차이를 보입니다. 필요한 경우 상담을 통한 개인 맞춤형 치료가 중요합니다.

### References

- [20] LYRICA & CELEBREX 안전성 정보 md

============================================================
[쿼리] 쎄레브렉스의 주요 상병코드는 무엇인가?
============================================================
INFO: Query nodes: 의약품, 질병 코드, 제약 (top_k:40, cosine:0.2)
WARNING: Some nodes are missing, maybe the storage is damaged
INFO: Local query: 21 entites, 29 relations
INFO: Query edges: 쎄레브렉스, 주요 상병코드 (top_k:40, cosine:0.2)
INFO: Global query: 19 entites, 16 relations
INFO: Raw search results: 37 entities, 41 relations, 0 vector chunks
INFO: After truncation: 37 entities, 41 relations
INFO: Selecting 34 from 34 entity-related chunks by vector similarity
INFO: Find no additional relations-related chunks from 41 relations
INFO: Round-robin merged chunks: 34 -> 34 (deduplicated 0)
WARNING: Rerank is enabled but no rerank model is configured. Please set up a rerank model or set enable_rerank=False in query parameters.
INFO: Final context: 37 entities, 41 relations, 20 chunks
INFO: Final chunks S+F/O: E3/1 E5/2 E3/3 E2/4 E3/5 E5/6 E1/7 E8/8 E1/9 E4/10 E1/11 E1/12 E1/13 E1/14 E1/15 E1/16 E2/17 E1/18 E2/19 E2/20
INFO: == LLM cache == Query cache hit, using cached response as query result
쎄레브렉스(Celebrex)에 대한 주요 상병코드는 다음과 같습니다:

- 쎄레브렉스는 일반적으로 통증 및 염증 개선을 위해 사용되는 의약품으로, Painful Condition Code에 속하는 여러 질환에 처방됩니다. 이 약은 특히 중대한 심 혈관계 혈전 반응, 심근경색증 및 뇌졸중 등의 위험이 있는 환자에게 경과 관찰이 필요합니다.

추가적으로, 쎄레브렉스의 사용은 의학적 상태에 따라 달라질 수 있으며, 특정 상병코드는 환자의 질환에 따라 의료인이 결정합니다.

### References

- [1] 쎄레브렉스® 상병코드1
- [2] 요양급여 및 상병코드 안내 문구
- [3] LYRICA & CELEBREX
- [4] PAIN LESS
- [5] 쎄레브렉스 안전성 정보 요약

============================================================
[쿼리] 류마티스관절염 관련 상병코드를 알려줘
============================================================
INFO: Query nodes: 의학적 진단, 질병 코드, 코드 분류, 류머티즘 (top_k:40, cosine:0.2)
WARNING: Some nodes are missing, maybe the storage is damaged
INFO: Local query: 16 entites, 15 relations
INFO: Query edges: 류마티스관절염, 상병코드 (top_k:40, cosine:0.2)
INFO: Global query: 35 entites, 20 relations
INFO: Raw search results: 46 entities, 29 relations, 0 vector chunks
INFO: After truncation: 46 entities, 29 relations
INFO: Selecting 32 from 32 entity-related chunks by vector similarity
INFO: Find no additional relations-related chunks from 29 relations
INFO: Round-robin merged chunks: 32 -> 32 (deduplicated 0)
WARNING: Rerank is enabled but no rerank model is configured. Please set up a rerank model or set enable_rerank=False in query parameters.
INFO: Final context: 46 entities, 29 relations, 20 chunks
INFO: Final chunks S+F/O: E4/1 E9/2 E4/3 E14/4 E2/5 E1/6 E3/7 E4/8 E1/9 E2/10 E3/11 E3/12 E1/13 E1/14 E1/15 E1/16 E2/17 E1/18 E1/19 E1/20
INFO: == LLM cache == Query cache hit, using cached response as query result
류마티스관절염 관련 상병코드는 다음과 같습니다:

### M05–06 류마티스관절염

- **M05 혈청검사양성 류마티스관절염 (Seropositive Rheumatoid Arthritis)**:
  - **M05.80–05.89**: 기타 혈청검사양성 류마티스관절염 (Other seropositive rheumatoid arthritis)
  - **M05.90–05.99**: 상세불명의 혈청검사양성 류마티스관절염 (Seropositive rheumatoid arthritis, unspecified)

- **M06 기타 류마티스관절염 (Other Rheumatoid Arthritis)**:
  - **M06.00–06.09**: 혈청검사음성 류마티스관절염 (Seronegative rheumatoid arthritis)
  - **M06.80–06.89**: 기타 명시된 류마티스관절염 (Other specified rheumatoid arthritis)
  - **M06.90–06.99**: 상세불명의 류마티스관절염 (Rheumatoid arthritis, unspecified)

이러한 상병코드는 류마티스관절염의 다양한 형태와 관련된 질병의 분류를 위한 것입니다[출처: 20. KR-CELE-2025-00055*[문헌] Painless*상병코드.md].

### References

- [1] 20. KR-CELE-2025-00055*[문헌] Painless*상병코드.md
  [시간] 쿼리: 4.2초

============================================================
[총 소요 시간] 8.6초
