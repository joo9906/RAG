# Total Test Results

**실행 명령:** `venv/Scripts/python total_test.py all`
**결과:** 65 tests passed — `Ran 65 tests in 8.027s — OK`

---

## 테스트 스위트별 결과

### 1. LLMReranker — 가중치 계산 (`TestLLMRerankerWeightedScore`)

| 테스트 | 설명 | 결과 |
|--------|------|------|
| `test_perfect_score` | 모든 기준 10점 → 가중합 1.0 | ok |
| `test_zero_score` | 모든 기준 0점 → 0.0 | ok |
| `test_only_relevance` | 관련성(weight=0.6)만 10점, 나머지 0 → 0.6 | ok |
| `test_missing_key_treated_as_zero` | 없는 키 → 0으로 처리 | ok |
| `test_half_score` | 모든 기준 5점 → 0.5 | ok |
| `test_weights_sum_to_one` | 가중치 합 = 1.0 | ok |

### 2. LLMReranker — rerank() 동작 (`TestLLMRerankerRerank`)

| 테스트 | 설명 | 결과 |
|--------|------|------|
| `test_empty_docs_returns_empty` | 빈 문서 리스트 → 빈 리스트 반환 | ok |
| `test_no_prompt_returns_original` | 프롬프트 미로드 → 원본 반환 | ok |
| `test_threshold_filters_low_score` | threshold 0.5 / score 0.2 → 문서 제거 | ok |
| `test_threshold_passes_high_score` | threshold 0.3 / score 0.8 → 문서 통과 | ok |
| `test_sorted_by_score_descending` | score 0.74 > 0.56 > 0.38 순으로 정렬 | ok |
| `test_rank_metadata_added` | rank 메타데이터 1부터 순서대로 추가 | ok |
| `test_overall_score_metadata_added` | overall_score 메타데이터 추가 (score=0.620) | ok |
| `test_search_query_defaults_to_user_query` | search_query 미전달 시 user_query로 대체 | ok |
| `test_llm_error_doc_filtered` | LLM 오류 → 점수=0, threshold 미만 제거 | ok [*] |
| `test_sub_scores_stored_in_metadata` | 5개 세부 점수 metadata 저장 (score=0.685) | ok |
| `test_multiple_docs_all_pass` | 5개 문서 모두 threshold 이상 → 5개 반환 | ok |
| `test_original_docs_not_modified_on_no_prompt` | 프롬프트 없을 때 원본 doc 참조 동일 | ok |

> [*] `test_llm_error_doc_filtered`: 테스트 중 `[ERROR] 문서 1 평가 실패: RuntimeError` 로그 출력됨 — **의도된 동작** (에러 핸들링 경로 검증)

### 3. HistoryManager — 토큰 카운팅 (`TestHistoryManagerTokenCount`)

| 테스트 | 설명 | 결과 |
|--------|------|------|
| `test_empty_messages` | 빈 메시지 리스트 → 0 | ok |
| `test_single_message_token_overhead` | 메시지당 4토큰 오버헤드 포함 | ok |
| `test_multiple_messages` | 여러 메시지 토큰 합산 | ok |
| `test_empty_content` | 빈 content → 오버헤드(4)만 | ok |

### 4. HistoryManager — build/get_history (`TestHistoryManagerBuild`)

| 테스트 | 설명 | 결과 |
|--------|------|------|
| `test_build_stores_messages` | build() → recent_messages 저장 (17토큰 계산) | ok |
| `test_get_history_no_summary` | 요약 없을 때 recent_messages 그대로 반환 | ok |
| `test_get_history_with_summary` | 요약 있을 때 SystemMessage prepend | ok |
| `test_empty_history_no_summary` | 빈 히스토리 → 빈 반환 (0토큰) | ok |
| `test_build_replaces_previous_messages` | build() 재호출 시 이전 메시지 교체 | ok |

### 5. HistoryManager — 요약 트리거 (`TestHistoryManagerSummarize`)

| 테스트 | 설명 | 결과 |
|--------|------|------|
| `test_under_limit_no_summary` | 21토큰 < 120K → 요약 안 함 | ok |
| `test_over_limit_triggers_summarize` | 120,001토큰 > 120K → model.invoke 호출 | ok |
| `test_summary_stored_after_trigger` | 요약 후 self.summary 저장 (3개 메시지 처리) | ok |
| `test_oldest_messages_removed_after_summarize` | 상위 30% (10개 중 3개) 메시지 제거 | ok |
| `test_summarize_llm_failure_fallback` | LLM 실패 → 원문 fallback 반환 | ok [*] |
| `test_rolling_summary_integrates_previous` | 두 번째 요약 시 이전 요약 통합 (7토큰) | ok |
| `test_build_summary_prompt_no_previous` | 이전 요약 없을 때 프롬프트에 '통합' 없음 | ok |
| `test_format_for_summary_role_labels` | role 레이블 한국어 변환 | ok |

> [*] `test_summarize_llm_failure_fallback`: `[ERROR] 요약 실패: LLM 연결 실패` 로그 출력 — **의도된 동작**

### 6. ContextEnhancer — 초기화 (`TestContextEnhancerInit`)

| 테스트 | 설명 | 결과 |
|--------|------|------|
| `test_product_list_set` | product_list 초기화 확인 | ok |
| `test_hospital_list_set` | hospital_list 초기화 확인 | ok |
| `test_model_stored` | model 저장 확인 | ok |

### 7. ContextEnhancer — 감지 (`TestContextEnhancerDetection`)

| 테스트 | 설명 | 결과 |
|--------|------|------|
| `test_detect_product_exact_match` | '아세리스' 정확 매칭 → ['아세리스'] | ok |
| `test_detect_product_case_insensitive` | 'acerys' → 'ACERYS' 대소문자 무관 감지 | ok |
| `test_detect_product_not_in_text` | 텍스트에 없는 제품 → [] | ok |
| `test_detect_hospital` | '스마트정형외과의원' 감지 | ok |
| `test_detect_competitive_product` | 경쟁제품 감지 | ok |
| `test_no_match_returns_empty` | 매칭 없음 → 빈 리스트 | ok |

### 8. ContextEnhancer — 질문 분석 (`TestContextEnhancerAnalyzeQuestion`)

| 테스트 | 설명 | 결과 |
|--------|------|------|
| `test_no_history_returns_original` | 히스토리 없음 → original_input 반환 | ok |
| `test_no_model_returns_original` | 모델 없음 → original_input 반환 | ok |
| `test_with_history_calls_llm` | '거기 방문 계획' → '스마트정형외과의원 방문 계획' 변환 | ok |
| `test_llm_error_returns_original` | LLM 오류 → original_input fallback | ok [*] |

> [*] `test_llm_error_returns_original`: `[ERROR] Error in question and history analysis: LLM 오류` 로그 출력 — **의도된 동작**

### 9. ContextEnhancer — `__call__` (`TestContextEnhancerCall`)

| 테스트 | 설명 | 결과 |
|--------|------|------|
| `test_returns_required_keys` | 필수 키 반환 확인 | ok |
| `test_original_input_preserved` | original_input 보존 | ok |
| `test_no_history_analyzed_equals_original` | 히스토리·모델 없으면 analyzed_question == original | ok |
| `test_detected_products_populated` | '아세리스' 제품 감지 → detected_products populated | ok |
| `test_product_list_override_per_call` | 호출별 product_list('임시제품') 오버라이드 후 원복 | ok |
| `test_hospital_list_override_per_call` | 호출별 hospital_list('임시병원') 오버라이드 후 원복 | ok |

### 10. `_build_enhanced_input` (`TestBuildEnhancedInput`)

| 테스트 | 설명 | 결과 |
|--------|------|------|
| `test_original_input_always_included` | original user input 항상 포함 | ok |
| `test_products_appended_when_detected` | 감지된 제품 섹션 추가 | ok |
| `test_no_products_section_absent` | 제품 없으면 섹션 미포함 | ok |
| `test_competitive_products_appended` | 경쟁제품 섹션 추가 | ok |
| `test_history_section_when_analyzed_differs` | analyzed_question != original 시 history 섹션 포함 | ok |
| `test_no_history_section_when_same` | analyzed_question == original 시 history 섹션 없음 | ok |
| `test_full_combination` | 전체 섹션 조합 | ok |

### 11. `_prepare_request_context` 통합 (`TestPrepareRequestContext`)

| 테스트 | 설명 | 결과 |
|--------|------|------|
| `test_returns_required_keys` | 필수 키 반환 확인 (2 products, 1 hospital 로드) | ok |
| `test_history_role_normalized_to_lowercase` | history role 소문자 정규화 | ok |
| `test_context_enhancer_receives_history` | ContextEnhancer가 history_messages 수신 확인 | ok |
| `test_analyzed_question_in_final_message` | analyzed_question이 최종 all_messages에 포함 | ok |

---

## 주의사항

`[ERROR]` 로그가 출력되는 3개 테스트는 **에러 처리(graceful degradation) 경로를 검증하는 의도된 동작**입니다:

| 테스트 | 출력 로그 | 의미 |
|--------|-----------|------|
| `test_llm_error_doc_filtered` | `[ERROR] 문서 1 평가 실패: RuntimeError` | Reranker LLM 오류 시 문서 필터링 검증 |
| `test_summarize_llm_failure_fallback` | `[ERROR] 요약 실패: LLM 연결 실패` | HistoryManager LLM 실패 시 원문 fallback 검증 |
| `test_llm_error_returns_original` | `[ERROR] Error in question and history analysis` | ContextEnhancer LLM 오류 시 original_input 반환 검증 |

---

## 실행 방법

```bash
venv/Scripts/python total_test.py            # 전체 (65개)
venv/Scripts/python total_test.py rerank     # Reranker (17개)
venv/Scripts/python total_test.py history    # HistoryManager (17개)
venv/Scripts/python total_test.py context    # ContextEnhancer (19개)
venv/Scripts/python total_test.py prepare    # _prepare_request_context (11개)
```
