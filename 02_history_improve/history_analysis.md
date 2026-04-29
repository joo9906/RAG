# History 관리 개선 분석

## 개요

대화가 길어질수록 히스토리 토큰이 누적되어 모델 컨텍스트(128K)를 초과할 위험이 있다.
120K를 기준선으로 삼아, 초과 시 **누적 요약(Rolling Summary)** 방식으로 압축한다.

---

## As-Is / To-Be 비교

| 구분 | 내용 |
|------|------|
| **As-Is** | `ContextEnhancer.__call__`이 `history_messages`를 받아 LLM으로 질문 재구성 → `_analyze_question_and_history()`로 `analyzed_question` 생성 → 재구성된 질문(대명사 치환)만 그래프에 전달 (`all_messages = [current_message]`) → 히스토리 원문은 그래프에 전달되지 않음 |
| **현재 브랜치** (`refactor/history`) | LLM 리라이팅 제거 → 히스토리를 그래프에 직접 전달 (`all_messages = history_messages + [current_message]`) → `ContextEnhancer`에서 `history` 파라미터 제거, 제품/병원명 감지만 수행 → **히스토리 토큰 제한 없음** → 대화가 길어질수록 컨텍스트 초과 위험 |
| **To-Be (구현 완료)** | 120K를 히스토리 토큰 기준으로 설정 → 초과 시 최신 80K 토큰 분량 보존, 나머지를 LLM으로 요약 → 요약본은 다음 요약 시 통합되어 재요약 (rolling summary) → user/assistant 쌍 경계 유지, 타임아웃 처리 추가 |

---

## To-Be 상세

### Rolling Summary 동작 방식

```
요약 v1 = summarize(messages 1~N)
요약 v2 = summarize(요약 v1 + messages N+1~M)
요약 v3 = summarize(요약 v2 + messages M+1~K)
```

### 토큰 기준 상수

```python
MAX_HISTORY_TOKENS = 120_000  # 128K 모델 대비 보수적 기준
KEEP_TOKENS_BUDGET = 80_000   # 요약 후 보존할 최대 토큰 (구: SUMMARIZE_RATIO 0.3 방식 대체)
MIN_KEEP_MESSAGES  = 2        # 최소 보존 메시지 수 (쌍 보정 후 안전망)
SUMMARIZE_TIMEOUT  = 30.0     # 요약 LLM 호출 타임아웃 (초)
```

> **변경 포인트**: 이전 설계의 `SUMMARIZE_RATIO = 0.3` (오래된 30% 슬라이스)에서
> `KEEP_TOKENS_BUDGET = 80_000` 기반 역방향 탐색으로 교체되었다.
> 비율 기반 방식은 메시지 길이가 불균일할 때 잔여 토큰이 예측 불가하다는 문제가 있었다.

### HistoryManager 클래스 설계 (구현 기준)

| 멤버 | 설명 |
|------|------|
| `summary: str` | 누적 요약본 |
| `recent_messages: List[BaseMessage]` | 아직 요약되지 않은 최근 메시지 |
| `build(history_messages)` | 히스토리 수신 후 토큰 초과 시 `_summarize_history()` 호출 |
| `_find_split_index()` | 뒤에서부터 `KEEP_TOKENS_BUDGET`을 채운 뒤 user/assistant 쌍 경계로 보정한 분할 인덱스 반환 |
| `_summarize_history()` | summary + recent 토큰 합산 → 초과 시 `_find_split_index()`로 범위 결정 후 `_summarize()` 호출 |
| `_build_summary_prompt(messages)` | 요약 프롬프트 문자열 생성 (기존 summary 유무에 따라 통합 요약 헤더 추가) |
| `_summarize(messages)` | 비동기 LLM 호출 (30s 타임아웃). 실패·타임아웃 시 `""` 반환하여 기존 요약 유지 |
| `get_history()` | 그래프에 전달할 메시지 구성 (요약 있으면 `SystemMessage`로 prepend) |
| `_count_tokens(messages)` | 메시지 리스트의 토큰 합산 |
| `_count_text_tokens(text)` | 문자열 단독 토큰 카운팅 (summary 토큰 계산용) |
| `_format_for_summary(messages)` | 메시지 리스트를 `[사용자] / [어시스턴트]` 형식 텍스트로 변환 |

### 요약 프롬프트 핵심 조건

- 제품명, 병원명, 수치(매출/처방수), 결정사항 반드시 보존
- 단순 인사/확인 메시지는 생략
- 이전 요약이 있으면 첫 줄에 `[통합 요약]` 명시

### 토큰 카운팅

- `tiktoken cl100k_base` 인코더 사용 (프로젝트 의존성에 이미 포함)
- 메시지당 content 토큰 + role·포맷 오버헤드 4토큰 계상
- `_summarize_history` 호출 시 `summary 토큰 + recent 토큰` 합산하여 판단

---

## 영향 범위

| 파일 | 변경 내용 |
|------|----------|
| `src/supervisors/v3/history_manager.py` | **신규 생성** — HistoryManager 클래스 |
| `src/supervisors/v3/supervisor_v3.py` | `_prepare_request_context()` 헬퍼 추가, 중복 코드 제거, 버그 수정 |

---

## 발견된 문제점 및 조치

#### 1. 토큰 제한 없음 (핵심 문제) → 해결

- `supervisor_v3.py:951` / `supervisor_v3.py:1299`에서 히스토리 길이 제한 없이 그대로 전달
- `HistoryManager` 도입으로 120K 초과 시 자동 압축

#### 2. 동일 로직 두 곳 중복 → 해결

- `process_user_query_stream`과 `process_user_query` 양쪽에 정규화·Redis 로딩·ContextEnhancer 호출 코드가 ~30줄씩 복붙
- `_prepare_request_context()` 헬퍼로 통합

#### 3. `context_result` NameError 버그 → 수정

- 리팩터링 후 `process_user_query`의 `initial_state`에 제거된 변수 참조가 잔존
- `context_result["original_input"]` → `original_input`으로 수정

#### 4. tiktoken 미사용 → 해결

- 의존성은 `requirements.txt`에 있었으나 코드에서 사용하지 않았음
- `HistoryManager`에서 처음으로 사용

#### 5. ContextEnhancer 스레드 안전성 (미해결)

- `supervisor.py:493~512` — `self.product_list`를 임시 교체/복원하는 방식
- 싱글턴 인스턴스에서 동시 요청 시 race condition 가능
- history 개선 범위 밖, 별도 대응 필요

#### 6. AbstractSupervisor 인터페이스 변경 없음

- `_prepare_request_context`와 `HistoryManager`는 supervisor 내부 구현
- `abstract_supervisor.py:22`의 `history: List[dict]` 시그니처 그대로 유지

---

## HistoryManager 메서드 설명

### `_count_tokens(messages)` / `_count_text_tokens(text)`

`tiktoken cl100k_base`로 토큰을 합산한다.

- `_count_tokens`: 메시지 리스트 전체 합산. 각 메시지의 content 토큰 + 4토큰(role·포맷) 계상
- `_count_text_tokens`: `summary` 문자열 단독 카운팅용 분리 유틸

### `_format_for_summary(messages)`

메시지 리스트를 `[사용자] / [어시스턴트] / [시스템]` 레이블 형식 텍스트로 변환한다. `_build_summary_prompt()`의 내부 헬퍼.

### `_build_summary_prompt(messages)`

요약 프롬프트 문자열을 생성한다. `_summarize()`에서 분리된 메서드.

- `self.summary`가 있으면 이전 요약을 포함한 통합 요약 헤더(`[통합 요약]` 명시) 구성
- 없으면 일반 요약 프롬프트 구성

### `_summarize(messages)`

**비동기 LLM 호출로 messages를 요약**한다.

- `asyncio.wait_for()`로 `SUMMARIZE_TIMEOUT`(30s) 제한
- 타임아웃: `asyncio.TimeoutError` 처리 → 경고 로그 후 `""` 반환
- 실패(`ValueError`, `TypeError`, `RuntimeError`): 에러 로그 후 `""` 반환
- 호출 성공 시 요약 결과의 토큰 수를 INFO 로그로 출력
- **반환값이 `""`(빈 문자열)이면 호출부(`_summarize_history`)에서 기존 `self.summary`를 유지**

> **변경 포인트**: 이전 설계에서는 실패 시 원문을 그대로 반환하는 fallback이었으나,
> 현재는 `""` 반환 후 호출부가 `if new_summary:` 조건으로 기존 요약을 유지한다.
> 원문 반환 fallback은 요약 토큰이 오히려 늘어나는 문제가 있었다.

### `_find_split_index()`

**뒤에서부터 `KEEP_TOKENS_BUDGET`을 채운 뒤 user/assistant 쌍 경계로 보정한 분할 인덱스를 반환**한다.

- `recent_messages` 끝에서부터 역방향으로 순회하며 토큰 누적
- `KEEP_TOKENS_BUDGET` 초과 직전 인덱스를 `split_idx`로 결정
- `split_idx`가 쌍 중간(non-human)이면 `human` 메시지 바로 앞으로 당김 (쌍 완결 보장)
- `len(recent_messages) - MIN_KEEP_MESSAGES`를 상한으로 두어 최소 2개 메시지 보존
- **반환값 ≤ 0이면 요약할 메시지가 없음**

> **변경 포인트**: 이전 설계의 `SUMMARIZE_RATIO = 0.3`(오래된 30% 슬라이스)을 대체.
> 비율 방식은 메시지 길이가 불균일할 때 잔여 토큰을 예측할 수 없었고,
> 쌍 경계가 깨져 요약 품질이 저하되는 문제도 있었다.

### `_summarize_history()`

`summary + recent` 토큰 합계로 **120K 초과 여부를 확인**한 뒤 `_summarize()`를 호출한다.

- `summary_tokens + recent_tokens` 합산 → `get_history()`가 실제로 반환하는 토큰량 기준으로 판단
- 초과 시 `_find_split_index()`로 분할 위치 결정 (구: `SUMMARIZE_RATIO` 슬라이스)
- `split_idx <= 0`이면 요약할 메시지가 없으므로 경고 후 반환
- `_summarize()` 반환값이 있을 때만 `self.summary` 갱신 (빈 문자열이면 기존 유지)

---

## supervisor_v3.py 변경 사항

### 1. import 추가

```python
from supervisors.v3.history_manager import HistoryManager
```

### 2. `_prepare_request_context()` 헬퍼 추가

`process_user_query_stream`과 `process_user_query` 두 메서드에서 완전히 동일하게 복붙되어 있던 코드를 단일 메서드로 통합했다.

| 단계 | 내용 |
|------|------|
| ① 정규화 | `history: List[dict]`의 `role` 값을 소문자로 변환 후 `convert_to_messages()` 적용 |
| ② Redis 로드 | `load_products_from_rdb()` / `load_hospitals_from_rdb()`로 제품·병원 목록 로드 |
| ③ ContextEnhancer | 제품명·병원명 감지 (`detected_products`, `detected_hospitals` 등 추출) |
| ④ HistoryManager | `build()` → `get_history()`로 토큰 압축 후 현재 메시지를 append해 `all_messages` 구성 |

반환값: `all_messages`, `known_products`, `known_hospitals`, `original_input`, `detected_company_products`, `detected_competitive_products`, `detected_hospitals`

### 3. 중복 코드 제거

두 진입점에서 각각 ~30줄씩 차지하던 블록을 아래 9줄로 교체했다.

```python
ctx = self._prepare_request_context(history, user_input, employee_ID)
all_messages                  = ctx["all_messages"]
known_products                = ctx["known_products"]
known_hospitals               = ctx["known_hospitals"]
original_input                = ctx["original_input"]
detected_company_products     = ctx["detected_company_products"]
detected_competitive_products = ctx["detected_competitive_products"]
detected_hospitals            = ctx["detected_hospitals"]
```

### 4. 버그 수정 — `context_result` NameError

리팩터링 과정에서 `process_user_query`의 `initial_state`에 이미 제거된 변수 참조가 잔존해 런타임 `NameError`가 발생할 수 있었다.

```python
# 수정 전 — context_result는 리팩터링으로 이미 사라진 변수
"query_for_rdb": context_result["original_input"],

# 수정 후
"query_for_rdb": original_input,
```

---

## 테스트 결과

외부 의존성(LLM)을 `unittest.mock`으로 격리한 단위 테스트를 실제로 실행한 결과다.
테스트 파일: `history_test.py` (프로젝트 루트)

### 테스트 환경

| 항목 | 값 |
|------|-----|
| Python | 3.11.9 |
| pytest | 9.0.1 |
| 비동기 백엔드 | anyio (asyncio) |
| tiktoken | `cl100k_base` 인코더 |
| 모델 mock | `AsyncMock` (응답 content 고정) |
| 임계값 | 실제 상수 대신 `_M_TOK`(메시지당 토큰) 배수로 소규모 설정 (`patch.multiple` 사용) |

> `_CONTENT = "hello " * 10` → `_C_TOK = 11`, `_M_TOK = 15` (content + 4 오버헤드)
> `_MAX = _M_TOK × 8 = 120`, `_KEEP = _M_TOK × 4 = 60`, `_KEEP_CORR = _M_TOK × 3 + 1 = 46`

---

### 실행 결과

```
pytest history_test.py -v

collected 8 items

history_test.py::test_tc01_no_summarize_when_under_budget        PASSED
history_test.py::test_tc02_summarize_and_system_message_prepended PASSED
history_test.py::test_tc03_pair_boundary_correction              PASSED
history_test.py::test_tc04_min_keep_messages_respected           PASSED
history_test.py::test_tc05_timeout_keeps_existing_summary        PASSED
history_test.py::test_tc06_api_error_keeps_existing_summary      PASSED
history_test.py::test_tc07_rolling_summary_includes_integration_header PASSED
history_test.py::test_tc08_no_summarizable_messages_when_only_min_keep PASSED

8 passed, 1 warning in 5.08s
```

---

### TC-01. 토큰 미초과 — 요약 미발생

```python
msgs = _msgs(3)   # 6개 메시지 × 15 토큰 = 90 < _MAX(120)
await mgr.build(msgs)

model.ainvoke.assert_not_called()     # _summarize() 미호출
assert mgr.summary == ""
assert mgr.get_history() == msgs      # 원본 그대로 반환
```

**PASSED**

---

### TC-02. 토큰 초과 — 요약 발생 및 SystemMessage prepend

```python
msgs = _msgs(5)   # 10개 메시지 × 15 토큰 = 150 > _MAX(120)
await mgr.build(msgs)

model.ainvoke.assert_called_once()
assert mgr.summary == "[요약 결과]"
assert mgr.get_history()[0].type == "system"
assert "[이전 대화 요약]" in mgr.get_history()[0].content
```

**PASSED**

---

### TC-03. user/assistant 쌍 경계 보정

```python
# KEEP_CORR = 46 → 역방향 탐색이 A3 인덱스에서 끊김 (AI 타입)
# → 보정으로 H3 인덱스(human)로 이동
msgs = _msgs(5)   # [H0,A0,H1,A1,H2,A2,H3,A3,H4,A4]

with _patch(KEEP_TOKENS_BUDGET=_KEEP_CORR):
    await mgr.build(msgs)

assert mgr.recent_messages[0].type == "human"   # 보존 시작이 항상 Human
assert model.ainvoke.called
```

**PASSED**

---

### TC-04. MIN_KEEP_MESSAGES 하한 보장

```python
# KEEP = 1개 분량(15 토큰)만 허용해도 최소 2개 보존
with _patch(KEEP_TOKENS_BUDGET=_M_TOK):
    await mgr.build(_msgs(5))

assert len(mgr.recent_messages) >= _MIN   # _MIN = 2
```

**PASSED**

---

### TC-05. 요약 타임아웃 — 기존 summary 유지

```python
model.ainvoke = AsyncMock(side_effect=asyncio.TimeoutError())
mgr.summary = "기존 요약"

with _patch(SUMMARIZE_TIMEOUT=0.001):
    await mgr.build(_msgs(5))

assert mgr.summary == "기존 요약"   # 변경 없음
```

**PASSED**

---

### TC-06. 요약 API 실패 — 기존 summary 유지

```python
model.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))
mgr.summary = "기존 요약"
await mgr.build(_msgs(5))

assert mgr.summary == "기존 요약"   # 변경 없음
```

**PASSED**

---

### TC-07. Rolling Summary — 통합 요약 헤더 포함 여부

```python
mgr.summary = "1차 요약본"
await mgr.build(_msgs(5))

prompt_content = model.ainvoke.call_args[0][0][0].content
assert "1차 요약본" in prompt_content
assert "[통합 요약]" in prompt_content
assert mgr.summary == "[통합 요약] 두 번째 요약"
```

**PASSED**

---

### TC-08. `split_idx <= 0` — 요약 대상 없음

```python
# 2개 메시지(MIN_KEEP), KEEP 예산이 전체보다 커서 split_idx=0
msgs = _msgs(1)   # [H0, A0]
total = _M_TOK * 2

with _patch(MAX_HISTORY_TOKENS=total - 1, KEEP_TOKENS_BUDGET=total + 100):
    await mgr.build(msgs)

model.ainvoke.assert_not_called()   # _summarize() 미호출
assert mgr.summary == ""
assert len(mgr.recent_messages) == 2
```

**PASSED**

---

### 결과 요약

| TC | 시나리오 | 결과 |
|----|---------|------|
| TC-01 | 토큰 미초과 → 요약 미발생 | **PASS** |
| TC-02 | 토큰 초과 → 요약 + SystemMessage prepend | **PASS** |
| TC-03 | user/assistant 쌍 경계 보정 | **PASS** |
| TC-04 | MIN_KEEP_MESSAGES 하한 보장 | **PASS** |
| TC-05 | 타임아웃 → 기존 summary 유지 | **PASS** |
| TC-06 | API 실패 → 기존 summary 유지 | **PASS** |
| TC-07 | Rolling Summary 통합 헤더 확인 | **PASS** |
| TC-08 | split_idx ≤ 0 → 요약 대상 없음 처리 | **PASS** |
| | **8 / 8** | **전체 통과** |

---

## 알려진 한계 (기본 구현 범위 내 허용)

| # | 항목 | 설명 |
|---|------|------|
| 1 | 단일 패스 요약 | `_summarize_history`는 1회만 실행. 보존 후에도 잔여 토큰이 120K를 초과하는 극단적 경우 재요약 없음. 실제 발생 가능성은 낮음 (`KEEP_TOKENS_BUDGET=80K`로 보존 범위가 명시적으로 제한되므로 이전 설계 대비 위험 감소) |
| 2 | SystemMessage 이중 삽입 | `get_history()`의 summary SystemMessage와 supervisor 프롬프트 SystemMessage가 공존. GPT-4.1 허용 범위이나, 문제 발생 시 `HumanMessage` prefix 방식으로 전환 필요 |
| 3 | 세션 간 요약 미지속 | `HistoryManager`는 요청마다 새로 생성. 요약이 Redis 등에 저장되지 않아 다음 요청에서 처음부터 재계산 |
