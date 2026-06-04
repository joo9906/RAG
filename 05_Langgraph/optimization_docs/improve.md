# 05_Langgraph 프로젝트 학습 및 개선 가이드

> 목적: "돌아가는 것"에서 "이해하는 것"으로  
> 방향: 새로 짜는 것보다 현재 코드를 끝까지 파악하고 실험하기

---

## STEP 1 — 일단 실제로 돌려보기

아무것도 이해하기 전에 먼저 눈으로 확인한다.

```bash
# 1. 인프라 실행
docker compose up -d

# 2. 서비스 확인
docker compose ps   # 3개 모두 Up 상태인지
curl http://localhost:9200   # ES 응답
curl http://localhost:8080/v1/.well-known/ready   # Weaviate 응답
redis-cli ping   # Redis PONG

# 3. 문서 인덱싱
python ingest.py

# 4. 챗봇 실행
python main.py
```

### 테스트 질문 목록

아래 순서대로 던지면서 파이프라인 각 단계가 실제로 동작하는지 확인한다.

```
1. "PER이 뭐야?"
   → [확인] 라우팅이 rag로 됐는가
   → [확인] 멀티쿼리 3개가 생성됐는가 (출력에서 확인)
   → [확인] 검증 결과가 grounded로 나왔는가

2. "PER이 뭐야?" (동일 질문 재입력)
   → [확인] 캐시 히트가 됐는가 (⚡ 캐시 히트 표시)
   → [확인] 첫 번째보다 빠르게 응답됐는가

3. "삼성전자 현재 주가 알려줘"
   → [확인] 라우팅이 stock으로 됐는가
   → [확인] MCP 도구가 실행됐는가

4. "ETF가 뭔지 설명하고 삼성전자 주가도 알려줘"
   → [확인] 어떤 라우팅 결정이 나오는가 (rag? stock?)
   → 현재 파이프라인의 한계가 드러나는 질문
```

---

## STEP 2 — 코드 파일별 이해 점검

각 파일을 열고 아래 질문에 답할 수 있는지 확인한다.  
답 못하는 항목이 다음에 공부할 지점이다.

### `cache/redis_cache.py`

- [ ] `cosine similarity`를 왜 쓰는가? dot product나 euclidean distance와 뭐가 다른가?
- [ ] threshold가 0.8이면 어느 정도 비슷한 질문까지 히트하는가? 0.9와 0.7은 어떻게 다른가?
- [ ] `_pack()` / `_unpack()`에서 `base64`를 쓰는 이유는?
- [ ] Redis가 꺼진 상태에서 질문하면 어떻게 되는가? 코드에서 어디서 처리하는가?
- [ ] 24시간(86400초) TTL이 지나면 어떻게 되는가?

**실험**: Redis를 `docker compose stop redis`로 내리고 질문해보기. 에러 없이 동작하는가?

---

### `graph/state.py`

- [ ] `Annotated[list, add_messages]`에서 `add_messages`가 하는 일이 뭔가?
  - 일반 `list`로 바꾸면 어떻게 달라지는가?
- [ ] 노드 함수가 `{"messages": [...]}` 를 반환할 때 기존 메시지가 덮어써지는가, 추가되는가?
- [ ] `correction_count`가 초기화되지 않으면 처음 노드 진입 시 `state.get("correction_count", 0)`은 왜 안 터지는가?

---

### `graph/nodes.py`

**`_rrf()` 함수**
- [ ] `k=60`이 뭘 의미하는가? k를 1로 바꾸면 어떻게 되는가?
- [ ] 같은 문서가 두 개의 쿼리에서 각각 1위로 나오면 RRF 점수는 얼마인가? 직접 계산해보기.
- [ ] `doc_map.setdefault(key, doc)` — `setdefault`를 왜 쓰는가? `doc_map[key] = doc`과 다른 점은?

**`cache_check_node`**
- [ ] 캐시 히트 시 `correction_count`가 초기화되지 않는다. 문제가 되는 상황이 있는가?

**`query_expander_node`**
- [ ] LLM이 JSON 배열 대신 다른 형식으로 응답하면 어떻게 처리되는가? 코드에서 찾기.
- [ ] `corrected_query`가 있을 때 `original_query` 대신 쓰는 이유는?

**`multi_retrieval_node`**
- [ ] `asyncio.gather()`와 `asyncio.to_thread()`를 함께 쓰는 이유는?
- [ ] `_weaviate_search()`와 `_es_search()`가 실패해도 전체 파이프라인이 안 죽는다. 어디서 처리하는가?

**`hallucination_checker_node`**
- [ ] 문서가 없을 때(`if not docs`) grounded로 처리하는 이유는? 이게 올바른 판단인가?
- [ ] JSON 파싱에 실패하면 왜 grounded로 fallback하는가? hallucinated로 하면 어떻게 될까?

**`self_correction_node`**
- [ ] `correction_count >= 2`일 때 루프를 멈추는 조건이 어디 있는가? (`graph/builder.py`에서 찾기)
- [ ] 2번 self-correction을 다 써도 여전히 hallucinated면 어떤 답변이 나오는가?

---

### `graph/builder.py`

- [ ] `_after_cache`가 `END`를 반환할 때 LangGraph는 어떻게 처리하는가?
- [ ] `self_correction` → `query_expander` 엣지가 루프를 만드는데, LangGraph가 무한루프를 허용하는가?
- [ ] `stock_node`는 `AdvancedRAGState`를 받는데 원래 `MessagesState`용으로 만들어졌다. 왜 호환되는가?

---

## STEP 3 — 작은 실험들

코드를 읽는 것보다 직접 바꿔보는 게 이해를 빠르게 만든다.

### 실험 1: RRF k값 조정

`graph/nodes.py`의 `_rrf()` 호출에서 k값을 바꿔보기.

```python
# 현재
reranked = _rrf(ranked_lists, top_k=5)

# 실험 A: k=1 (극단적으로 작게)
reranked = _rrf(ranked_lists, top_k=5, k=1)

# 실험 B: k=1000 (극단적으로 크게)
reranked = _rrf(ranked_lists, top_k=5, k=1000)
```

같은 질문에 대해 검색 결과 순위가 어떻게 달라지는지 확인.

---

### 실험 2: 캐시 threshold 조정

`graph/nodes.py` 상단의 `_cache` 초기화 부분:

```python
# 현재
_cache = SemanticCache(redis_url=_REDIS_URL, threshold=0.8)

# 낮춰보기
_cache = SemanticCache(redis_url=_REDIS_URL, threshold=0.5)
```

"PER이 뭐야?"를 캐시한 뒤 "주가수익비율이 뭐야?"로 질문해보기.  
0.8에서는 캐시 미스? 0.5에서는 캐시 히트?

---

### 실험 3: hallucination fallback 변경

`hallucination_checker_node`의 JSON 파싱 실패 fallback을 바꿔보기.

```python
# 현재: 실패 시 grounded 처리
except (json.JSONDecodeError, ValueError, IndexError):
    verdict = "grounded"

# 변경: 실패 시 hallucinated 처리
except (json.JSONDecodeError, ValueError, IndexError):
    verdict = "hallucinated"
    reason = "JSON 파싱 실패 — 보수적으로 hallucinated 처리"
```

self-correction이 더 자주 발동하는가? 답변 품질이 올라가는가 내려가는가?

---

### 실험 4: Self-correction 최대 횟수 조정

`graph/builder.py`:

```python
# 현재
_MAX_CORRECTIONS = 2

# 0으로 바꾸면 self-correction을 아예 안 함
_MAX_CORRECTIONS = 0
```

같은 질문에 대해 답변 품질 차이가 있는가?  
응답 속도 차이는 얼마나 나는가?

---

## STEP 4 — AI 없이 혼자 짜보기 (구성요소 단위)

전체를 다시 짜는 게 아니라, **하나씩** 골라서 빈 파일에 AI 없이 짜본다.  
막히는 지점이 실제 이해의 한계다.

### 과제 A: `SemanticCache.get()` 재구현 (난이도: 중)

`cache/redis_cache.py`를 닫고, 빈 파일에 아래 스펙만 보고 구현해보기.

```
- Redis에서 모든 캐시 엔트리 ID 목록을 가져온다
- 각 엔트리의 임베딩을 가져와 쿼리 임베딩과 코사인 유사도를 계산한다
- 가장 높은 유사도가 threshold 이상이면 해당 답변을 반환한다
- Redis가 없으면 (None, 0.0)을 반환한다
```

막히는 지점: numpy 배열 직렬화, asyncio Redis 클라이언트 사용법, 코사인 유사도 공식

---

### 과제 B: `_rrf()` 재구현 (난이도: 하)

수식만 보고 짜기.

```
RRF(d) = Σ 1 / (k + rank(d, list_i))

- 여러 개의 ranked list를 받는다
- 각 list에서 문서의 순위(0-based)를 찾는다
- 모든 list에서의 점수를 합산한다
- 내림차순 정렬 후 top_k개를 반환한다
- 반환되는 각 문서에 rrf_score 필드를 추가한다
```

---

### 과제 C: `router_node` 개선 (난이도: 중)

현재 라우터는 "rag" / "stock" 둘 중 하나만 고른다.  
"ETF가 뭔지 설명하고 삼성전자 주가도 알려줘" 같은 질문을 처리하지 못한다.

AI 없이 "hybrid" 라우팅을 추가해보기.  
- `route_decision`이 "hybrid"일 때 어떻게 처리할지 설계부터 해보기
- `graph/builder.py`에서 어디를 고쳐야 하는지 파악하기
- 실제로 구현하지 않아도 설계 문서만 작성해도 충분

---

## STEP 5 — 더 깊이 파고들 개념

현재 프로젝트와 직결된 것들만. 추상적 공부가 아니라 코드와 연결해서.

| 개념 | 연결된 코드 | 공부하면 알게 되는 것 |
|------|------------|-------------------|
| Cosine Similarity vs Dot Product | `cache/redis_cache.py` `_cosine()` | 왜 임베딩 비교에 코사인을 쓰는가 |
| asyncio `gather` vs `to_thread` | `graph/nodes.py` `multi_retrieval_node` | CPU-bound vs IO-bound 차이 |
| LangGraph StateGraph 내부 동작 | `graph/builder.py` 전체 | 노드 간 상태 전파 방식 |
| BM25 알고리즘 | `vector_store/elasticsearch_store.py` | 키워드 검색이 벡터 검색과 다른 이유 |
| LLM JSON 출력 신뢰성 문제 | `graph/nodes.py` hallucination_checker | 왜 structured output을 쓰는가 |
| Redis Sorted Set vs Hash | `cache/redis_cache.py` | 현재 구현의 O(n) 탐색 한계 |

---

## 체크리스트 요약

```
□ docker compose 3개 서비스 정상 실행 확인
□ ingest.py 실행 후 문서 인덱싱 확인
□ 4개 테스트 질문 모두 실행해보기
□ 동일 질문 재입력 시 캐시 히트 확인
□ Redis 내리고 파이프라인 동작 확인
□ _rrf() 함수 직접 손으로 계산해보기
□ RRF k값 실험
□ threshold 실험
□ SemanticCache.get() AI 없이 재구현
□ 모르는 개념 1개 골라서 공부
```

---

> 이 파일의 모든 항목이 끝났을 때, 현재 프로젝트를 "이해한 것"이다.  
> 그 다음에 새로운 프로젝트를 시작하면 처음부터 짜도 방향이 생긴다.
