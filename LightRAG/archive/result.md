# process.py 실행 결과 및 개선 이력

작성일: 2026-04-24  
파일명: `process.py`

---

## 0. 사용법

```bash
# 전체 파이프라인 (삽입 + 시각화 + 통계 + 쿼리)
python process.py

# 단일 질문
python process.py -q 기넥신 효능 알려줘

# 삽입 건너뛰고 쿼리만
python process.py --skip-insert -q 질문

# 4개 모드 비교
python process.py --mode-compare 기넥신 효능 알려줘

# 배치 쿼리 (파일에서 질문 읽어 batch_result.md 저장)
python process.py --batch queries.txt

# 그래프 통계만
python process.py --stats

# 캐시 내용 확인
python process.py --show-cache

# 캐시 끄고 쿼리
python process.py --no-cache -q 질문

# 캐시 유사도 임계값 조정 (기본 0.92)
python process.py --cache-threshold 0.85 -q 질문

# LLM 모델 변경
python process.py --llm 4o -q 질문

# 청크 사이즈 변경
python process.py --chunk-size 700 --chunk-overlap 100
```

---

## 1. 삽입 결과 (실측)

문서 6개 기준 실측 데이터. 가상환경: `skch-aix-ce/venv`

```
========================================================================================================
                                    [ 삽입 토큰 사용량 요약 ]
========================================================================================================
  파일명                                        |  LLM |     입력tok |    출력tok |  임베딩tok |    비용($) |   시간
  ─────────────────────────────────────────────────────────────────────────────────────────────────────
  20. KR-CELE-2025-00055_[문헌] Painless_상병코드. |    6 |    29,477 |    4,867 |     6,183 |   0.00815 |  49.5초
  3_7_2 '24년 하반기 신입사원_편두통(수벡스)...md   |   18 |    84,013 |    8,579 |    13,969 |   0.01957 |  59.3초
  OPC_102_EPSILON study(초기 파킨슨병)...md        |   10 |    46,112 |    4,472 |     7,288 |   0.01055 |  32.5초
  기넥신 PSS 가이드북.md                           |   56 |   258,058 |   28,952 |    44,870 |   0.06191 | 163.0초
  레밋치브로셔 RMC-HD04-202304-01.md               |   22 |   104,984 |   13,216 |    19,229 |   0.02618 | 110.1초
  프로맥_소장점막손상효과.md                         |   23 |   103,224 |   12,267 |    17,948 |   0.02518 |  84.5초
  ─────────────────────────────────────────────────────────────────────────────────────────────────────
  합계                                          |  135 |   625,868 |   72,353 |   109,487 |   0.15153 | 498.9초
========================================================================================================
  LLM 입력  :   625,868 tok  $0.09388
  LLM 출력  :    72,353 tok  $0.04341
  임베딩    :   109,487 tok  $0.01423
  소계      :             $0.15153  (= ₩209)
  소요 시간 : 498.9초 (약 8분 19초)
========================================================================================================
```

→ 파일별 LLM 호출 횟수·토큰·비용·시간이 실시간 출력되며 마지막에 합계 테이블로 정리된다.  
→ 기넥신 가이드북이 56회 LLM 호출로 전체 비용의 40%를 차지 — 문서 길이에 비례.

---

## 2. 초기화 로그 해석

```
INFO: Loaded graph from ...graph_chunk_entity_relation.graphml with N nodes, N edges
```

→ 이전 실행에서 저장된 GraphML을 로드. 최초 실행 시에는 새로 생성.

```
INFO: Qdrant collection: lightrag_vdb_entities_text_embedding_3_large_2048d
INFO: Qdrant collection: lightrag_vdb_relationships_text_embedding_3_large_2048d
INFO: Qdrant collection: lightrag_vdb_chunks_text_embedding_3_large_2048d
```

→ 컬렉션명에 모델명·차원 접미사 포함 (버전 관리용).

```
INFO: Process NNNNN KV load full_docs with 6 records
INFO: Process NNNNN KV load text_chunks with 66 records
INFO: Process NNNNN KV load entity_chunks with 679 records
INFO: Process NNNNN KV load relation_chunks with 556 records
```

→ JSON KV 스토어 로드. `full_docs` 6건, 청크 66개, 엔티티 679개, 관계 556개.

```
INFO: LLM func: 4 new workers initialized
INFO: Embedding func: 8 new workers initialized
```

→ LLM 비동기 워커 4개, 임베딩 워커 8개 생성.

---

## 3. 설정 정보

```
[모델] gpt-4o-mini  (in=$0.15/1M  out=$0.60/1M)
[캐시] N건 로드됨 (dim=2048)
[청크] size=1000tok  overlap=150tok
[임베딩] text-embedding-3-large  dim=2048  $0.130/1M
```

| 항목        | 값                               | 위치                                |
| ----------- | -------------------------------- | ----------------------------------- |
| LLM         | gpt-4o-mini                      | CONFIG `LLM_MODEL`                  |
| 임베딩      | text-embedding-3-large, 2048차원 | CONFIG `EMB_MODEL / EMB_DIM`        |
| 청크        | 1000토큰 / 150 오버랩            | CONFIG `CHUNK_SIZE / CHUNK_OVERLAP` |
| 캐시 임계값 | 0.92 (코사인 유사도)             | `CACHE_SIMILARITY_THRESHOLD`        |
| 언어        | Korean                           | `addon_params["language"]`          |

---

## 4. 시각화

```
[시각화] GraphML 로딩: ...graph_chunk_entity_relation.graphml
  노드: 679, 엣지: 556
  노드 200개 초과 -> 상위 200개만 표시
  시각화 완료 -> ...knowledge_graph.html
[시간] 시각화: 0.1초
```

→ 연결도 상위 200개 노드만 렌더링 (pyvis 성능 제한).  
→ 생성 파일: `lightrag_before_chunk_test/knowledge_graph.html`

---

## 5. 그래프 통계 (실측)

```
노드: 679개  |  엣지: 556개
연결 컴포넌트: 195개  |  고립 노드: 133개
평균 연결도: 1.6  |  최대 연결도: 27
```

→ 노드 679개 중 133개(19.6%)가 고립 노드.  
→ 평균 연결도 1.6, 최대 27(날푸라핀염산염).

### 엔티티 타입 분포

| 타입         | 수  | 의미                         |
| ------------ | --- | ---------------------------- |
| concept      | 220 | 추상 개념 (MCI, 인지기능 등) |
| content      | 70  | 문서·논문 내용               |
| person       | 70  | 인물 (의사, 연구자 등)       |
| data         | 65  | 수치·데이터                  |
| organization | 28  | 기관·병원                    |
| UNKNOWN      | 26  | 추출 시 타입 미지정          |
| condition    | 24  | 질환 상태                    |
| event        | 24  | 임상 이벤트 등               |
| product      | 20  | 약품·제품                    |
| method       | 20  | 투여법·검사법                |
| drug         | 14  | 의약품                       |
| symptom      | 13  | 증상                         |
| …            | …   | …                            |

### 연결도 상위 10 엔티티

| 엔티티                | 타입    | 연결수 | 의미                      |
| --------------------- | ------- | ------ | ------------------------- |
| 날푸라핀염산염        | drug    | 27     | 레밋치 주성분 — 허브 노드 |
| 기넥신                | content | 21     | 핵심 약품                 |
| Polaprezinc           | drug    | 19     | 프로맥 주성분             |
| 수벡스정              | content | 17     | 편두통 약품               |
| Ginkgo Biloba Extract | content | 15     | 기넥신 성분               |
| Dementia              | concept | 14     | 치매                      |
| 레밋치 구강붕해정     | content | 14     | 제품명                    |
| Placebo               | product | 13     | 임상 대조군               |
| Opicapone             | drug    | 13     | 파킨슨 약물               |
| Ginkgo Biloba         | concept | 11     | 은행나무 추출 개념        |

---

## 6. 쿼리 결과 샘플

### 캐시 히트 예시

```
[임베딩] 12tok | 0.86초
** 캐시 히트 ** (유사도: 1.0000)
[캐시 히트] 절약 $0.00455 | 0.86초
```

→ 이전 동일 질문이 캐시에 저장되어 LLM 호출 없이 즉시 반환.

### 신규 LLM 호출 예시 (타이밍 분석)

```
  ┌─ [타이밍 분석]
  │  쿼리 임베딩      : 0.19초  (1회)   ← 질문 벡터화
  │  벡터·그래프 서칭 : 0.27초          ← Qdrant + GraphML 탐색
  │  검색 보조 LLM    : 1.96초  (1회)   ← 키워드 추출용 내부 LLM
  │  답변 생성 LLM    : 20.76초         ← 최종 답변 생성 (전체의 ~90%)
  │  ──────────────────────────────────
  │  합계             : 23.18초
  └─ 비용             : $0.00439  (= ₩6)
```

---

## 7. 개선 이력

### 7-1. 오류 수정 (2026-04-24)

| 항목                 | 이전                                             | 이후                               |
| -------------------- | ------------------------------------------------ | ---------------------------------- |
| `qdrant-client` 버전 | 1.15.1 (서버 1.17.1과 불일치)                    | 1.17.1 (버전 일치)                 |
| 컬렉션 접미사        | 없음 → `WARNING: missing suffix`                 | `model_name=EMB_MODEL` 추가로 해결 |
| 레거시 컬렉션        | `lightrag_vdb_entities` 등 3개 잔존              | 마이그레이션 후 삭제 완료          |
| Rerank 경고          | `WARNING: Rerank is enabled but no rerank model` | `enable_rerank=False` 설정         |

### 7-2. 그래프 품질 최적화 (2026-04-24)

`process.py` `_build_rag()` 에 아래 3개 파라미터 추가.

```python
entity_extract_max_gleaning=2,   # 기본 1 → 모호한 청크 재추출 횟수 증가
force_llm_summary_on_merge=3,    # 기본 8 → 엔티티 머지 시 LLM 요약 더 자주 발동
addon_params={
    "language": "Korean",        # 기본 English → 한국어 문서 추출 정확도 향상
},
```

| 파라미터                      | 기본값    | 변경값   | 기대 효과                                    |
| ----------------------------- | --------- | -------- | -------------------------------------------- |
| `entity_extract_max_gleaning` | 1         | 2        | 누락 엔티티 감소 (비용 소폭 증가)            |
| `force_llm_summary_on_merge`  | 8         | 3        | 같은 엔티티의 중복 설명 적극 통합            |
| `language`                    | `English` | `Korean` | 한국어 기준 엔티티 추출 → UNKNOWN·other 감소 |

→ **적용 기준**: 데이터를 밀고 재삽입할 때부터 반영됨.

---

## 8. 종합 해석

| 항목             | 값                       | 평가                    |
| ---------------- | ------------------------ | ----------------------- |
| 문서 수          | 6개                      | 소규모 데이터셋         |
| 청크 수          | 66개                     | 평균 11청크/문서        |
| KG 노드          | 679개                    | 재삽입 후 정합성 개선   |
| KG 엣지          | 556개                    | 이전(532)보다 연결 증가 |
| 고립 노드 비율   | 19.6% (133/679)          | 이전(28%)보다 개선      |
| 평균 연결도      | 1.6                      | 이전(1.4)보다 향상      |
| 삽입 총비용      | $0.15153 (₩209)          | 문서 6개 기준           |
| 쿼리 1건 비용    | ~$0.0044 (₩6)            | 캐시 미스 기준          |
| 핵심 허브 엔티티 | 날푸라핀염산염 (연결=27) | 레밋치 주성분           |
