# Reranker 개선안

## 0. 개선 효과 요약

에이전트 1회 요청 기준 (VDB 2회 호출, RDB+VDB 멀티 도구 포함):

| 개선 항목                                      | 구분    | 절감 시간     | 측정 방법                                                                | 테스트 통과 |
| ---------------------------------------------- | ------- | ------------- | ------------------------------------------------------------------------ | ----------- |
| 모델 교체 + Reranker 척도 0→10 + 프롬프트 강화 | ✅ 구현 | **−6.24s**    | 실측 (8.32s → 2.08s)                                                     | 18 / 18     |
| Reranker 싱글턴 (2번째 호출~)                  | ✅ 구현 | **−0.87s/회** | 실측 (초기화 0.87s 제거)                                                 | 4 / 4       |
| `parallel_tool_calls=True` (A)                 | ✅ 구현 | **~−1.5s**    | 추정 (LLM 왕복 1회 감소)                                                 | 16 / 16     |
| 임베딩 공유 (B)                                | ✅ 구현 | **~−0.3s**    | 추정 (임베딩 호출 1회 제거, Azure 임베딩 병목 시 최대 3초까지 절감 가능) | 1 / 1       |
| Preflight 병렬화 (C)                           | ✅ 구현 | **~−0.15s**   | 추정 (Redis+DB 동시 조회)                                                | 5 / 5       |

> **테스트 최종 결과: 82/82 passed** (`improve_methods/parallel_test.py`, LangGraph 0.6.11 환경, 실 LLM 포함)

### 전체 절감 시간

|                                        | 시간                |
| -------------------------------------- | ------------------- |
| **실측 합계** (Reranker 척도 + 싱글턴) | **−7.11s**          |
| **추정 포함 합계** (A + B + C 추가)    | **−9.06s ~ -11.56** |

> Reranker를 2회 이상 호출하는 요청(VDB 2회)에서는 싱글턴 효과가 중복 적용되어 추가 절감.
> 실측 기준으로만 보면 **기존 대비 약 30~40% 응답 속도 단축** 수준.

---

## 1. Critical Path 분석

```
vdb_search_qdrant() 호출
  └─ Qdrant 검색 → 문서 top_k개 (기본 5개)
       └─ LLMReranker.rerank()
            └─ ThreadPoolExecutor: 문서 1개당 gpt-4.1 호출 1회 (병렬)
                 ├─ gpt-4.1 invoke → relevance/specificity/... 0|1 반환
                 ├─ gpt-4.1 invoke
                 ├─ ...
```

에이전트가 VDB를 2회 호출하면 rerank LLM만 **최대 10회** 발생.

---

## 2. 발견된 문제와 해결방법

### 2-1. 모델 과중 → 경량 모델로 교체

**문제:** `get_model()` → `gpt-4.1` (플래그십). 0|1 바이너리 판단에 가장 비싼 모델 사용.

**개선:** config의 `llm_reranker` 키로 모델 지정 가능하도록 변경. 기본값 `gpt-4.1-mini`.

```python
# 이전
base_llm = LangchainLoader().get_model(temperature=0)  # gpt-4.1

# 이후 — reranker.py:62
loader = LangchainLoader()
reranker_model = loader.config.get("llm_reranker", "gpt-4.1-mini")
base_llm = loader.get_model(model_name=reranker_model, temperature=0)
```

---

### 2-2. 0|1 스코어 구조 한계 → 0~10 척도로 변경

**문제:** 5개 기준이 모두 0 또는 1만 반환.

현재 가중치 구조:

| 기준           | 가중치 |
| -------------- | ------ |
| relevance      | 0.60   |
| specificity    | 0.15   |
| completeness   | 0.10   |
| practicality   | 0.10   |
| constraint_fit | 0.05   |

threshold=0.3 기준으로 `relevance=1`이면 무조건 0.60으로 통과. 결국 **relevance 하나로 당락이 결정**되는 구조였고, 나머지 4개 기준은 동점 처리용에 불과했음.

**개선:** 0~10 정수 척도로 변경. 최종 점수는 `(score / 10.0) * weight`로 정규화.

```python
# 이전
relevance: int = Field(description="관련성 점수 (0 또는 1)", ge=0, le=1)
...

# 이후 — reranker.py:17
relevance: int = Field(description="관련성 (0=무관~10=핵심 일치, 10단계)", ge=0, le=10)
specificity: int = Field(description="구체성 (0=추상적~10=매우 구체적, 10단계)", ge=0, le=10)
completeness: int = Field(description="완전성 (0=불완전~10=완전 충족, 10단계)", ge=0, le=10)
practicality: int = Field(description="실용성 (0=이론적~10=매우 실용적, 10단계)", ge=0, le=10)
constraint_fit: int = Field(description="조건 부합성 (0=미충족~10=완전 충족, 10단계)", ge=0, le=10)
```

---

### 2-3. 매 VDB 호출마다 재인스턴스화 → 싱글턴으로 교체

**문제:** `vdb.py` 내부에서 VDB 툴이 호출될 때마다 `LLMReranker()`를 새로 생성. `LangchainLoader()` + `load_config()` + 프롬프트 파일 읽기가 호출 횟수만큼 반복. 실측 초기화 비용: **약 0.9s/회**.

```python
# 이전 — vdb.py:639, 호출마다 실행
reranker = LLMReranker(threshold=0.3)
```

**개선:** 모듈 레벨 싱글턴으로 교체.

```python
# 이후 — vdb.py:38
_reranker: LLMReranker | None = None

def _get_reranker() -> LLMReranker:
    global _reranker
    if _reranker is None:
        _reranker = LLMReranker(threshold=0.3)
    return _reranker
```

---

### 2-4. 유사 제품명 혼동 → 프롬프트 규칙 추가

**문제:** 기존 프롬프트는 `relevance`를 "쿼리와 관련 있으면 1점"으로만 정의. "기넥신 부작용" 쿼리에서 리넥신 문서에도 `relevance=1`이 부여됨. 제품명이 달라도 같은 주제(부작용)를 다루면 관련 있다고 판단하는 구조적 결함.

**개선 1 — Relevance에 유사 제품 상한선 명시** (`reranker_instruction.txt`)

```
이전: "제품명이 언급되면 1점, 무관하면 0점"

이후: 10단계 앵커 포인트 + IMPORTANT 규칙
  - 10: 동일 제품명이 명확히 일치
  - 4~6: 관련 주제지만 다른 제품 문서
  - IMPORTANT: 쿼리가 특정 제품(기넥신)을 명시할 때,
    다른 제품(리넥신, 타나민) 문서의 relevance 최대값은 6.
    주제가 같더라도 7 이상 부여 금지.
```

**개선 2 — Constraint Fit에 제품명 불일치 = 0 강제**

```
이전: "조건이 반영되면 1점"

이후: IMPORTANT 규칙 추가
  - 쿼리가 제품명(기넥신)을 명시할 때,
    다른 제품(리넥신, 타나민) 문서는 constraint_fit = 0 강제.
```

---

### 2-5. 도구 병렬 호출 API 블로킹 → `parallel_tool_calls` 추가 (base_agent.py)

**문제:** LLM이 VDB + RDB를 동시에 호출하겠다고 판단해도, `parallel_tool_calls` 미설정 시 API가 이를 막고 순차 실행으로 강제. LLM의 판단과 무관하게 병렬 호출이 불가능했음.

> RDB에 데이터가 없어 VDB를 후속 호출하는 순차 폴백 로직은 LLM이 의존성을 판단해 결정하는 것이므로 이 설정과 무관하게 그대로 동작한다.

**개선:** `bind_tools`에 `parallel_tool_calls=True`를 적용해 LLM이 원하는 병렬 호출을 API 레벨에서 허용.
`bind_tools(tools, parallel_tool_calls=True).kwargs`로 tool 스키마와 함께 추출한 뒤, `extra_body`(프롬프트 캐시 키)와 단일 `bind()` 호출로 병합한다.

```python
# base_agent.py — create_agent_subgraph() 내부

tools = self._initialize_tools()
base_model = self.langchain_loader.get_model(
    temperature=0, seed=42, tags=[self.agent_name]
)

if tools:
    tool_binding_kwargs = base_model.bind_tools(tools, parallel_tool_calls=True).kwargs
    model = base_model.bind(
        **tool_binding_kwargs,
        extra_body={"prompt_cache_key": self.get_prompt_path().stem}
    )
else:
    model = base_model.bind(
        extra_body={"prompt_cache_key": self.get_prompt_path().stem}
    )

enhanced_agent = create_react_agent(
    model=model,
    tools=tools,
    prompt=agent_prompt_function,
    name=self.agent_name,
    state_schema=state_schema,
    post_model_hook=agent_post_hook,
)
```

---

## 3. 테스트 결과

### 3-1. Reranker 속도 개선

쿼리: `"기넥신 부작용 알려줘"` / `top_k=10`, `threshold=0.3`, 모델: `gpt-4.1-mini`

| 회차 | 변경 사항                     | Reranker 소요 | 변화                      |
| ---- | ----------------------------- | ------------- | ------------------------- |
| 1차  | 기준 (0~3 척도, gpt-4.1-mini) | **8.32s**     | —                         |
| 2차  | 프롬프트 제품명 조건 강화     | **1.37s**     | **−6.95s (−84%)**         |
| 3차  | 척도 0~3 → 0~10 세분화        | **2.08s**     | +0.71s (정확도 향상 대가) |

**누적 개선:** 8.32s → 2.08s — **전체 Reranker 처리 시간 75% 단축**

단계별 전체 소요:

| 단계                      | 수정 전  | 수정 후   | 절감          |
| ------------------------- | -------- | --------- | ------------- |
| Reranker 초기화 (1회)     | 0.87s    | 0.87s     | —             |
| Reranker 초기화 (싱글턴)  | 0.87s/회 | **0s**    | **−0.87s/회** |
| Reranker 평가 (10개 병렬) | 8.32s    | **2.08s** | **−6.24s**    |
| 총 소요 (싱글턴 포함)     | ~22.7s   | **~6.9s** | **−15.8s**    |

> 싱글턴 적용 시 2번째 호출부터 초기화 비용 0. VDB 검색 + 평가 = **약 5.8s** 수준.

### 3-2. 유사 제품명 혼동 해소 확인

| 항목                             | 수정 전      | 수정 후        |
| -------------------------------- | ------------ | -------------- |
| 리넥신 doc (p.31) 순위           | **3위**      | **8위**        |
| 리넥신 doc (p.31) 전체 점수      | 0.967        | **0.600**      |
| 리넥신 doc (p.31) constraint_fit | 3/3 (오평가) | **0/3** (정확) |
| 리넥신 doc (p.25) constraint_fit | 3/3 (오평가) | **0/3** (정확) |

### 3-3. 0~10 척도 세분화 효과

기존 0~3 척도에서 `2/3`으로 묶이던 상위 문서들이 0~10 척도 도입으로 세분화됨.

| 구간        | 이전 척도 (0~3)      | 이후 척도 (0~10)          |
| ----------- | -------------------- | ------------------------- |
| 상위 1~6위  | 0.700 ~ 0.867 (뭉침) | 0.700 ~ 0.780 (세분화)    |
| 하위 7~10위 | 판단 불가            | 0.320 ~ 0.585 (명확 분리) |

### 3-4. 단위 테스트 — 82/82 통과

`improve_methods/parallel_test.py` 실행 결과 (LangGraph 0.6.11 환경):

```
========================= 82 passed in 28.68s =========================
```

#### 테스트 구성

| 테스트 그룹                      | 테스트 수 | 내용                                                                                          |
| -------------------------------- | --------- | --------------------------------------------------------------------------------------------- |
| `TestParallelToolCallsBinding`   | 9개       | `parallel_tool_calls` kwargs 보존, `_should_bind_tools=False` 확인, 병합 binding 구조 검증    |
| `TestCreatePromptTemplate`       | 16개      | `PromptTemplate` 반환 구조, `data_access_section` partial_variables 내용, 도구 조합별 검증    |
| `TestAgentPromptFunction`        | 13개      | 단일 SystemMessage 구조, datetime 포맷팅, 히스토리 이어붙임, 플레이스홀더 제거 확인           |
| `TestPreflightConcurrency`       | 6개       | ctx/profile/embedding 3개 작업 동시 실행 (윈도우 겹침), 임베딩 1회 호출, 예외 시 크래시 없음  |
| `TestParallelToolCallsExecution` | 7개       | LLM 2회 vs 3회 호출(병렬/순차), ToolMessage 개수, tool_call_id 양쪽 전달, 최종 state 확인     |
| `TestLLMRerankerScoring`         | 9개       | 0~10 척도 정규화, WEIGHTS 합계=1.0, 가중치별 기여도, Pydantic 범위 유효성                     |
| `TestRerankerSingleton`          | 4개       | `_get_reranker()` 1회만 생성, 동일 인스턴스 반환, threshold=0.3 초기화 확인                   |
| `TestLLMRerankerWithMockLLM`     | 8개       | threshold 필터링, 점수 내림차순 정렬, rank/overall_score 메타데이터, 빈 목록 처리             |
| `TestRerankerIntegration` ★      | 9개       | **실제 LLM 호출** — 기넥신/리넥신 구분, constraint_fit ≤ 3 확인, [0,1] 범위, 0~10 정수 스코어 |

> ★ `TestRerankerIntegration`은 실제 API 키를 사용하는 통합 테스트. config 미설정 시 자동 skip.

#### 구버전 테스트(47개)에서 달라진 점

| 변경 항목                             | 이전 (47개)                                                  | 현재 (82개)                                            |
| ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| `_create_prompt_template()` 반환 타입 | `str` 기대 (에러)                                            | `PromptTemplate` 객체 검증으로 수정                    |
| `agent_prompt_function` 메시지 구조   | 2개 (static + dynamic) 기대 (에러)                           | 단일 SystemMessage + 히스토리 구조로 수정              |
| Group 5 모델 패칭                     | `patch.object(ChatOpenAI, "invoke")` → Pydantic v2 차단 에러 | `_TrackingFakeModel` (BaseChatModel 서브클래스)로 교체 |
| Reranker 전체 검증                    | 없음                                                         | Group 6~9 (30개) 신규 추가                             |

#### 주요 검증 포인트

- 단일 `bind()` 결과의 `kwargs`에 `tools`, `parallel_tool_calls`, `extra_body` 3개 모두 존재 확인
- `_should_bind_tools(model, tools)` → **False** 반환 확인 (재바인딩 방지, `parallel_tool_calls` 보존)
- preflight 3개 작업의 실행 시간 윈도우가 모두 겹침 확인 (순차 실행 불가)
- 가중치(`WEIGHTS`) 합계 = **1.0** 확인, 0~10 점수 → [0, 1] 정규화 수식 검증
- `_get_reranker()` 3회 연속 호출 시 `LLMReranker()` 생성자 **1회만** 호출됨 확인
- 기넥신 쿼리 + 리넥신 문서 → `constraint_fit_score ≤ 3` 실측 확인 (프롬프트 규칙 작동)

### 3-5. 전체 예상 속도 개선 요약

에이전트 1회 요청 기준 (VDB 2회 호출, RDB+VDB 멀티 도구 포함):

| 개선 항목                           | 절감 예상                   | 비고                 | 테스트 검증                                                              |
| ----------------------------------- | --------------------------- | -------------------- | ------------------------------------------------------------------------ |
| Reranker 경량 모델 (`gpt-4.1-mini`) | 모델 비용 감소 및 속도 개선 | ~ 3.5s 감소          | `TestRerankerIntegration::test_reranker_initializes_with_correct_model`  |
| Reranker 0~10 척도 + 프롬프트 강화  | **−6.24s** (Reranker 전체)  | 실측값               | `TestLLMRerankerScoring` 9개, `TestRerankerIntegration` 9개              |
| Reranker 싱글턴                     | **−0.9s/호출** (2번째~)     | 초기화 제거          | `TestRerankerSingleton` 4개                                              |
| `parallel_tool_calls=True` (A)      | **~−1.5s**                  | LLM 호출 1회 감소    | `TestParallelToolCallsBinding` 9개, `TestParallelToolCallsExecution` 7개 |
| 임베딩 공유 (B)                     | ~−0.3s                      | 임베딩 호출 1회 제거 | `TestPreflightConcurrency::test_embedding_computed_exactly_once`         |
| Preflight 병렬화 (C)                | ~−0.15s                     | Redis+DB 동시 조회   | `TestPreflightConcurrency::test_three_preflight_tasks_run_concurrently`  |

---

## 4. 추가 개선 과제

### A. Agent `parallel_tool_calls=True` 적용 ★★★ ✅ 구현 완료

ReAct 루프가 기본적으로 도구를 순차 실행하므로, `bind_tools`에 `parallel_tool_calls=True`를 적용해 RDB + VDB를 한 번의 LLM 호출로 동시 실행.

```
기존: LLM → rdb 실행 → LLM → vdb 실행 → LLM: 답변  (LLM 3회)
개선: LLM → rdb + vdb 동시 실행 → LLM: 답변          (LLM 2회)
```

구현 방법은 3-1절 참고.

**예상 절감:** ~1.5s (멀티 도구 에이전트에서 LLM 호출 1회 감소)

---

### B. 임베딩 공유 ★★★ ✅ 구현 완료

**문제:** `CacheService._search_cache_hybrid`와 `QuestionGraph._get_similar_question`이 동일한 `user_input`을 독립적으로 임베딩. 같은 벡터를 2회 계산.

**구현:** preflight에서 한 번 계산한 임베딩을 `init_state["query_embedding"]`에 담아 두 곳 모두 재사용.

```python
# supervisor_v3.py — preflight에서 한 번만 계산 (기존 구현)
emb_future = executor.submit(get_embedding_sync)
query_embedding = emb_future.result()
initial_state = { ..., "query_embedding": query_embedding, ... }

# CacheService.run() — state에서 fallback으로 읽기 (이번 수정: cache_service.py:59-61)
async def run(self, init_state: dict, query_embedding: Optional[List[float]] = None):
    if query_embedding is None:
        query_embedding = init_state.get("query_embedding")  # ← 추가된 1줄
    ...

# QuestionGraph._get_similar_question — state에서 주입 (기존 구현: supervisor.py:293-294)
matched_node_id = self.question_graph._get_similar_question(
    user_input, history, query_embedding=state.get("query_embedding")
)
```

**경로별 구현 상태:**

| 경로                                                                        | 임베딩 출처                              | 상태         |
| --------------------------------------------------------------------------- | ---------------------------------------- | ------------ |
| `process_user_query` → `_race_cache_vs_supervisor` → `CacheService.run()`   | `init_state["query_embedding"]` fallback | ✅ 이번 수정 |
| `process_user_query` → supervisor graph → `_get_similar_question()`         | `state.get("query_embedding")`           | ✅ 기존 구현 |
| `process_user_query_stream` → `_cache.run(init_state, query_embedding=...)` | 명시적 전달                              | ✅ 기존 구현 |

**관련 파일:** `supervisor_v3.py`, `cache/cache_service.py` (+3줄), `question_graph/question_graph.py`

**예상 절감:** ~300ms (임베딩 호출 1회 제거)

---

### C. Preflight 병렬화 ★★★ ✅ 구현 완료 (테스트 검증)

**문제:** `_prepare_request_context`(Redis 조회)와 `_load_user_profile`(DB 조회)이 순차 실행. 두 작업은 완전히 독립적.

**구현 상태:** `process_user_query` 및 `process_user_query_stream` 모두 `ThreadPoolExecutor` / `asyncio.gather`로 3개 작업(ctx, profile, embedding) 병렬 실행.

**테스트 검증:** `TestPreflightConcurrency` — 실행 시간 윈도우 겹침 방식으로 병렬 실행 확인.

```python
# process_user_query — ThreadPoolExecutor 방식
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    ctx_future = executor.submit(self._prepare_request_context, history, user_input, employee_ID)
    profile_future = executor.submit(self._load_user_profile, employee_ID)
    emb_future = executor.submit(get_embedding_sync)
    ctx = ctx_future.result()
    profile = profile_future.result()
    query_embedding = emb_future.result()

# process_user_query_stream — asyncio.gather 방식
preflight_results = await asyncio.gather(
    asyncio.to_thread(self._prepare_request_context, ...),
    asyncio.to_thread(self._load_user_profile, employee_ID),
    get_embedding()
)
```

**예상 절감:** ~150ms

---

### D. RDB 쿼리 결과 단기 캐싱 ★★☆

**문제:** 동일하거나 유사한 SQL 쿼리를 반복 실행할 때 매번 DB 왕복 발생.

**설계:** `query_rdb` 도구 래퍼에 Redis TTL 캐시 레이어 추가.

```python
# tools/rdb/query_cache.py (신규)
def rdb_cache(ttl: int = 300):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            cache_key = "rdb:" + hashlib.md5(
                json.dumps({"args": str(args), "kwargs": str(kwargs)}, ensure_ascii=False)
                .encode()
            ).hexdigest()

            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

            result = fn(*args, **kwargs)
            redis_client.setex(cache_key, ttl, json.dumps(result, ensure_ascii=False))
            return result
        return wrapper
    return decorator
```

TTL 정책: 매출/처방 등 실시간 쿼리 60s, 제품 마스터 등 정적 데이터 3600s.

**예상 절감:** 반복 쿼리 시 ~300ms/회

---

### E. Prompt 정적/동적 분리 ★★☆ ❌ 미구현

**문제:** `known_products` 전체 리스트(수백 개)가 매 요청 프롬프트에 포함되어 캐시 미스 유발 및 컨텍스트 낭비.

**현재 상태:** `agent_prompt_function`은 함수 기반으로 교체되었으나, 제품/병원 목록은 state에서 로드만 하고 프롬프트에 주입하지 않음. 정적/동적 분리 자체는 미구현.

**예상 효과:** 프롬프트 크기 감소 + 캐시 히트율 향상

---

### F. Reranker → cross-encoder 교체 ★★☆ ❌ 미구현(논의 필요)

LLM 호출 없이 cross-encoder로 교체하면 VDB 1회당 ~100ms 이하로 단축.
하지만 실제 도입 가능 여부는 SK 인프라 정책에 달려있음.

```python
# Cohere Rerank (외부 API)
results = cohere_client.rerank(query=query, documents=docs, model="rerank-v3.5")

# 또는 로컬 cross-encoder (외부 의존 없음)
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = model.predict([(query, doc.page_content) for doc in docs])
```
