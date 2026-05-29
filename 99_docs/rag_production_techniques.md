# RAG 고도화 기법 정리

이 레포에서 다룬 내용 외에 실제 프로덕션 환경에서 사용되는 RAG 고도화 기법들을 정리한 문서입니다.  
각 기법이 어떤 문제를 해결하는지를 중심으로 서술합니다.

---

## 1. 청킹 전략 (Chunking Strategy)

> **문제**: 청크 크기와 방식이 검색 정확도에 직결된다. 고정 크기 청킹은 문장/단락 경계를 무시해 의미가 잘린다.

### Parent-Child Chunking
- 큰 단위(Parent)로 저장하되, 검색은 작은 단위(Child)로 수행
- 검색 정밀도는 높이면서 LLM에 넘기는 컨텍스트는 넓게 유지
- LangChain의 `ParentDocumentRetriever`로 구현 가능

### Semantic Chunking
- 임베딩 유사도가 급격히 떨어지는 지점을 청크 경계로 자동 감지
- 고정 크기보다 의미 단위가 유지되어 검색 품질 향상
- `semchunk` 라이브러리 또는 LangChain `SemanticChunker` 활용

### Hierarchical Chunking (계층 청킹)
- 문서 → 섹션 → 단락 → 문장 단위로 다층 인덱스 구성
- 질문 유형에 따라 적절한 단위로 검색

### Summary 
- 청킹된 문서의 앞단에 hierarchical 정보와 현재 문서의 요약을 넣을 경우 임베딩 성능의 향상 가능

---

## 2. 쿼리 변환 (Query Transformation)

> **문제**: 사용자 쿼리가 짧거나 모호할 때 검색 결과가 빈약해진다.

### HyDE (Hypothetical Document Embeddings)
- 질문에 대한 가상의 답변을 LLM으로 먼저 생성한 후, 그 임베딩으로 검색
- 질문 임베딩보다 답변 임베딩이 문서 임베딩 공간에 더 가깝다는 원리
```python
# 원리
hypothetical_answer = llm("다음 질문에 대한 답변을 작성해: " + query)
results = vector_search(embed(hypothetical_answer))
```

### Multi-Query Retrieval
- 하나의 질문을 LLM으로 여러 표현으로 확장(3~5개)해 각각 검색 후 병합
- 단일 표현의 어휘 불일치(Vocabulary Mismatch) 문제 완화

### Step-Back Prompting
- 구체적인 질문을 더 추상적인 상위 질문으로 변환해 검색
- "이 함수의 버그가 뭐야?" → "이 코드의 전체 동작 방식은?" 으로 변환 후 검색

### Query Decomposition
- 복합 질문을 여러 단순 질문으로 분해해 각각 검색 후 결과를 조합
- LangGraph의 조건 분기와 결합해 서브 쿼리를 병렬 처리

---

## 3. 검색 품질 개선 (Advanced Retrieval)

> **문제**: 단순 유사도 검색만으로는 노이즈 문서가 많이 포함되고 관련성 높은 문서를 놓친다.

### Contextual Compression
- 검색된 청크 전체가 아니라 질문과 관련된 부분만 추출해서 LLM에 전달
- 컨텍스트 길이를 줄여 LLM 비용 절감 + 정답 품질 향상
- LangChain `ContextualCompressionRetriever` 활용

### Ensemble Retriever (앙상블 리트리버)
- BM25, 벡터 검색, 키워드 검색 등 여러 리트리버 결과를 RRF로 합산
- 각 방식의 약점을 서로 보완하는 효과

### CRAG (Corrective RAG)
- 검색된 문서의 관련성 점수가 낮으면 웹 검색으로 보완 검색을 수행
- "검색 결과가 충분한가?"를 모델이 스스로 판단하는 Self-Correcting 패턴
```
검색 → 관련성 평가 → [충분] → 응답 생성
                  → [부족] → 웹 검색 보완 → 응답 생성
```

### Self-RAG
- 검색이 필요한지 여부, 검색 결과가 유용한지, 생성된 답변이 사실에 근거하는지를 모두 LLM이 판단
- 특수 토큰(`[Retrieve]`, `[Relevant]`, `[Supported]`)으로 흐름 제어
- 불필요한 검색을 줄이고 할루시네이션을 자체 검증

---

## 4. 평가 프레임워크 (Evaluation)

> **문제**: "검색이 잘 되는 것 같다"는 주관적 판단 대신 수치로 품질을 측정해야 한다.

### RAGAS 핵심 지표

| 지표 | 설명 | 측정 대상 |
|------|------|-----------|
| **Faithfulness** | 답변이 컨텍스트에 근거하는 비율 | 할루시네이션 탐지 |
| **Answer Relevancy** | 답변이 질문에 얼마나 관련 있는지 | 답변 품질 |
| **Context Recall** | 정답 도출에 필요한 정보가 검색됐는지 | 검색 완전성 |
| **Context Precision** | 검색된 청크 중 실제로 유용한 비율 | 검색 노이즈 |

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])
```

### LangSmith 평가
- `@traceable`로 파이프라인 전 구간 추적
- 실제 사용자 쿼리 기반으로 A/B 테스트 (예: 리랭킹 적용 전/후 비교)
- 이상 응답 자동 감지 및 알림 설정

---

## 5. 프로덕션 인프라

> **문제**: 개발 환경에서 잘 동작해도 실제 트래픽에서는 비용, 지연, 안정성 문제가 발생한다.

### Semantic Caching (시맨틱 캐싱)
- 동일하거나 유사한 쿼리가 반복될 때 LLM 재호출 없이 캐시된 답변 반환
- 키워드 일치가 아닌 임베딩 유사도로 캐시 히트 판단
- `GPTCache`, Redis + 벡터 검색으로 구현

### 비동기 병렬 처리
- ES 검색, Weaviate 검색, 리랭킹을 `asyncio.gather`로 동시 실행
- I/O 바운드 작업(API 호출, DB 검색)에서 지연 시간 대폭 감소
```python
es_docs, wv_docs = await asyncio.gather(
    es_search_async(query),
    weaviate_search_async(query)
)
```

### 스트리밍 응답 (Streaming)
- LLM 응답을 생성되는 즉시 클라이언트에 전달 (체감 응답 속도 개선)
- FastAPI + SSE(Server-Sent Events) 또는 WebSocket으로 구현

### 인덱스 관리 전략
- **증분 인덱싱**: 변경된 문서만 재인덱싱 (전체 재구축 방지)
- **버전 관리**: 인덱스 별칭(Alias)을 활용해 무중단 인덱스 교체
- **샤딩 전략**: 문서 수/검색 부하에 따른 ES 샤드 설계

---

## 6. 임베딩 고도화

> **문제**: 범용 임베딩 모델은 도메인 특화 용어나 문서 구조를 잘 표현하지 못한다.

### 도메인 특화 임베딩 파인튜닝
- 도메인 내 (질문, 관련 문서) 쌍을 구성해 Bi-encoder를 파인튜닝
- `sentence-transformers` 라이브러리의 `MultipleNegativesRankingLoss` 활용
- 파인튜닝 전/후 NDCG, MRR 지표로 성능 비교

### Late Interaction (ColBERT)
- 쿼리와 문서의 토큰 단위 상호작용을 검색 시점에 계산 (MaxSim 연산)
- Bi-encoder의 속도와 Cross-encoder의 정확도 사이 균형점
- `RAGatouille` 라이브러리로 구현 가능

---

## 7. 멀티모달 RAG

> **문제**: 실무 문서에는 텍스트 외에 표, 이미지, 차트가 포함되어 있어 텍스트만 처리하면 정보 손실이 크다.

### 표(Table) 처리
- 표를 마크다운 또는 HTML로 변환 후 별도 인덱싱
- 표 전체를 하나의 청크로 유지 (분리 시 의미 손실)

### 이미지/차트 처리
- Vision LLM으로 이미지를 텍스트 설명으로 변환 후 인덱싱
- 이미지 임베딩(CLIP 등)을 직접 저장하는 멀티모달 벡터 DB 활용

### PDF 고품질 파싱
- `docling`(이 레포에서 사용 중): 레이아웃 인식 기반 파싱
- `unstructured`: 표/이미지/텍스트를 요소별로 분리 추출

---

## 8. 보안 및 비용 관리

### Prompt Injection 방어
- 사용자 입력을 시스템 프롬프트와 명확히 분리
- 검색된 문서에 악의적 명령어가 포함될 수 있으므로 컨텍스트도 검증

### 비용 최적화
- **Prompt Caching**: Anthropic/OpenAI의 프롬프트 캐싱으로 반복 시스템 프롬프트 비용 절감
- **모델 라우팅**: 단순 질문은 소형 모델, 복잡한 추론만 대형 모델로 라우팅
- **청크 수 조정**: 리랭킹 후 LLM에 전달하는 청크 수를 최소화 (top-3~4가 적정)

---

## 단계별 적용 로드맵

```
지금 이 레포 수준
    │
    ├── 1단계: 평가 체계 구축  ← 가장 먼저 (없으면 개선 여부를 모름)
    │          RAGAS 지표 측정, LangSmith 대시보드
    │
    ├── 2단계: 쿼리 변환      ← 구현 난이도 낮고 효과 큼
    │          HyDE 또는 Multi-Query
    │
    ├── 3단계: Agentic RAG   ← LangGraph 이미 있으므로 자연스러운 확장
    │          CRAG 패턴 (검색 품질 자가 판단 + 재검색)
    │
    ├── 4단계: 인프라 고도화
    │          Semantic Caching, 비동기 처리, 스트리밍
    │
    └── 5단계: 도메인 파인튜닝
               임베딩 파인튜닝, ColBERT 도입
```
