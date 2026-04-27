# total_process_batch.py 사용 가이드

LightRAG 파이프라인에 OpenAI Batch API를 통합하여 LLM 비용을 최대 50% 절감하는 통합 스크립트입니다.

---

## 목차

1. [개요](#1-개요)
2. [사전 준비](#2-사전-준비)
3. [CONFIG 설정](#3-config-설정)
4. [전체 파이프라인 구조](#4-전체-파이프라인-구조)
5. [실행 모드별 사용법](#5-실행-모드별-사용법)
   - [5-1. 전체 파이프라인 (기본)](#5-1-전체-파이프라인-기본)
   - [5-2. 배치 제출만 하고 나중에 재개](#5-2-배치-제출만-하고-나중에-재개)
   - [5-3. 삽입 건너뛰고 힐링만](#5-3-삽입-건너뛰고-힐링만)
   - [5-4. 쿼리](#5-4-쿼리)
   - [5-5. 통계 / 시각화](#5-5-통계--시각화)
   - [5-6. 실시간 모드 (배치 없이)](#5-6-실시간-모드-배치-없이)
6. [힐링 전략 상세](#6-힐링-전략-상세)
7. [전체 CLI 옵션 레퍼런스](#7-전체-cli-옵션-레퍼런스)
8. [비용 구조 및 절감 계산](#8-비용-구조-및-절감-계산)
9. [내부 구조 설명](#9-내부-구조-설명)
10. [자주 묻는 질문](#10-자주-묻는-질문)

---

## 1. 개요

### Batch API란?

OpenAI Batch API는 LLM 요청을 즉시 처리하지 않고 JSONL 파일로 묶어 비동기로 제출하면, 최대 24시간 내에 처리 결과를 돌려주는 기능입니다. **비용이 실시간 대비 50% 저렴**합니다.

```
실시간 API  →  즉시 응답, 높은 단가
Batch API   →  최대 24시간 소요, 50% 단가
```

### 적용 범위

| 단계                  | 처리 방식              | 이유                       |
| --------------------- | ---------------------- | -------------------------- |
| 삽입: 엔티티 추출     | **배치**               | 청크별로 독립적, 순서 무관 |
| 삽입: gleaning/merge  | 실시간                 | 이전 추출 결과에 의존      |
| 힐링 전략 A (Prune)   | 실시간                 | LLM 불필요                 |
| 힐링 전략 B (Embed)   | 실시간 (Embedding API) | 빠른 처리, 배치 불필요     |
| 힐링 전략 C (LLM)     | **배치**               | 고립 노드별 독립 요청      |
| 힐링 전략 D (Re-link) | **배치**               | 고립 노드별 독립 요청      |
| 쿼리                  | 실시간                 | 즉시 결과 필요             |

---

## 2. 사전 준비

### 디렉터리 구조

```
after_quit/
├── total_process_batch.py   ← 이 파일
├── lightrag_before_chunk_test/   ← WORKING_DIR (자동 생성)
│   ├── graph_chunk_entity_relation.graphml
│   ├── kv_store_*.json
│   ├── query_cache.json
│   ├── batch_state.json     ← 배치 상태 저장 (자동 생성)
│   └── knowledge_graph.html
├── docs/
│   └── guide.md
└── ../../
    ├── .env.json            ← API 키
    └── test_md/             ← 입력 문서
        ├── 약품A.md
        └── 약품B.md
```

### `.env.json` 형식

```json
{
  "openai_api_key": "sk-..."
}
```

### 필수 패키지

```bash
pip install lightrag openai networkx pyvis numpy tiktoken
```

Qdrant 벡터 DB도 실행 중이어야 합니다:

```bash
# Docker로 실행
docker run -p 6333:6333 qdrant/qdrant
```

---

## 3. CONFIG 설정

파일 상단 **CONFIG 블록만 수정**하면 전체 파이프라인이 맞춰 동작합니다.

```python
# [1] 데이터 경로
ENV_JSON_PATH     = "../../.env.json"        # OpenAI API 키 파일
WORKING_DIR       = ".../lightrag_before_chunk_test"  # 그래프 저장 위치
MD_DIR            = ".../test_md"            # 입력 MD 파일 폴더
QDRANT_URL        = "http://localhost:6333"
QDRANT_COLLECTION = "lightrag_before_chunk_test"

# [2] LLM 모델
LLM_MODEL = "mini"   # "mini" = gpt-4o-mini | "4o" = gpt-4o

# [3] 임베딩 모델
EMB_MODEL = "text-embedding-3-large"
EMB_DIM   = 2048

# [4] 청크 설정 - 따로 청킹 전략을 활용할 경우 청킹이 불필요하므로 빼야함
CHUNK_SIZE    = 1000   # 토큰 단위
CHUNK_OVERLAP = 150

# [5] 쿼리 캐시
CACHE_SIMILARITY_THRESHOLD = 0.85   # 캐시 히트 코사인 유사도 임계값. 적정 수치는 직접 테스트 하며 찾아봐야 할 것 같음

# [6] 힐링 기본값 - 해당 수치들을 넘기지 못할 경우 적용 X
HEAL_PRUNE_MIN_DESC  = 10    # Prune: 설명 최소 길이
HEAL_EMBED_THRESHOLD = 0.75  # Embed: 코사인 유사도 임계값
HEAL_EMBED_TOP_K     = 2     # Embed: 노드당 최대 연결 수
HEAL_LLM_LIMIT       = 50    # LLM: 처리할 고립 노드 최대 수
HEAL_LLM_MIN_CONF    = 0.5   # LLM: 신뢰도 최소값
HEAL_RELINK_LIMIT    = 30    # Re-link: 처리할 노드 최대 수

# [7] 배치 API 설정
BATCH_POLL_INTERVAL  = 60    # 폴링 간격 (초)
BATCH_COMPLETION_WIN = "24h" # OpenAI 배치 완료 창
```

---

## 4. 전체 파이프라인 구조

```
[원본 파일 파싱] → [MD 파일] → 청크 분할 → 엔티티 추출 프롬프트 생성
                                ↓
                     ┌──────────────────────┐
                     │   OpenAI Batch API   │  ← 최대 24시간
                     │  (JSONL 업로드/폴링)   │
                     └──────────────────────┘
                                ↓
              BatchCachedLLM (캐시 재주입) → LightRAG ainsert()
                                ↓
               GraphML 저장 (노드 + 엣지)
                                ↓
              힐링: Prune → Embed → LLM배치 → Re-link배치
                                ↓
                   HTML 시각화 / 통계 출력
                                ↓
                          쿼리 (실시간)
```

### 두 단계 삽입 방식

LightRAG의 삽입은 "엔티티 추출 → gleaning → merge" 순서로 진행되며, gleaning/merge는 추출 결과에 의존하므로 실시간으로 유지됩니다. 엔티티 추출만 배치로 미리 처리하고, 실제 삽입 시 캐시에서 결과를 꺼내 씁니다.

```
[Phase 1] 배치 제출
  모든 청크 → entity extraction 프롬프트 → Batch API 제출

[Phase 2] 삽입 재실행
  LightRAG ainsert() 호출
  └─ entity extraction 요청 → BatchCachedLLM → 캐시 히트 (비용 없음)
  └─ gleaning/merge 요청   → 실시간 API (비율 낮음)
```

---

## 5. 실행 모드별 사용법

### 5-1. 전체 파이프라인 (기본)

배치 삽입 → 배치 힐링 → 시각화 → 통계 → 기본 쿼리 순서로 전부 실행합니다.

```bash
python total_process_batch.py
```

완료까지 폴링하며 대기합니다. 배치 처리 시간(수 분~수 시간)이 걸릴 수 있습니다.

---

### 5-2. 배치 제출만 하고 나중에 재개

터미널을 닫아도 되는 워크플로우입니다.

**Step 1: 제출만 하고 종료**

```bash
python total_process_batch.py --submit-only
```

출력 예:

```
  [Batch Insert] MD 파일 3개 청크 분할 중 ...
  [Batch] 배치 ID: batch_abc123  상태: validating
  [Batch Insert] 제출 완료. 배치 ID: batch_abc123
  완료 후 재개하려면:
    python total_process_batch.py --resume insert
  상태 저장 → lightrag_before_chunk_test/batch_state.json
```

**Step 2: (선택) 상태 확인**

```bash
python total_process_batch.py --batch-status
```

출력 예:

```
  [insert]  batch_id=batch_abc123
    제출: 2026-04-27T15:30:00
    상태: in_progress  완료: 45/120
```

**Step 3: 완료 후 재개**

```bash
# 삽입만 재개
python total_process_batch.py --resume insert

# 힐링만 재개
python total_process_batch.py --resume heal

# 삽입 + 힐링 모두 재개
python total_process_batch.py --resume all
```

재개 시 `batch_state.json`에 저장된 `batch_id`로 폴링을 재시작합니다.

---

### 5-3. 삽입 건너뛰고 힐링만

그래프가 이미 구축된 경우 힐링만 실행합니다.

```bash
# 기본 힐링 (A+B+C, 배치 포함)
python total_process_batch.py --heal

# 전략 D(Re-link)까지 포함한 전체 힐링
python total_process_batch.py --heal-all

# 특정 전략만
python total_process_batch.py --heal-prune          # 전략 A만
python total_process_batch.py --heal-embed          # 전략 B만
python total_process_batch.py --heal-llm            # 전략 C만
python total_process_batch.py --heal-relink         # 전략 D만

# 힐링 전략 C+D를 Batch API로 실행
python total_process_batch.py --heal --batch-heal

# 건식 실행 (그래프 변경 없이 결과 미리보기)
python total_process_batch.py --heal --dry-run

# 고립 노드 목록 상세 출력 후 힐링
python total_process_batch.py --heal --isolated-detail

# 힐링 파라미터 조정
python total_process_batch.py --heal \
  --embed-threshold 0.80 \   # 임베딩 유사도 임계값 높이기
  --embed-top-k 3 \          # 노드당 최대 연결 수 늘리기
  --llm-limit 100 \          # LLM 처리 노드 수 늘리기
  --llm-min-confidence 0.6   # 신뢰도 기준 높이기
```

#### 힐링 전략 요약

| 플래그          | 전략       | 방법                                  | 배치 여부 |
| --------------- | ---------- | ------------------------------------- | --------- |
| `--heal-prune`  | A: Prune   | 설명 없거나 짧은 UNKNOWN 노드 삭제    | 불필요    |
| `--heal-embed`  | B: Embed   | 임베딩 코사인 유사도로 유사 노드 연결 | 불필요    |
| `--heal-llm`    | C: LLM     | LLM에게 관계 제안 요청                | **배치**  |
| `--heal-relink` | D: Re-link | 소스 청크에서 관계 재추출             | **배치**  |

`--heal` = A+B+C 실행 (D 제외)  
`--heal-all` = A+B+C+D 전부 실행

---

### 5-4. 쿼리

기존 그래프에 질문합니다. 항상 실시간으로 처리됩니다.

```bash
# 단건 쿼리 (기본 hybrid 모드)
python total_process_batch.py -q 기넥신 누구한테 써?

# 모드 지정
python total_process_batch.py -q 외과에 어떤 약을 추천할까 --mode local
python total_process_batch.py -q 류마티스 관절염 약 --mode global

# 모든 모드(naive/local/global/hybrid) 비교
python total_process_batch.py --mode-compare 기넥신 적응증

# 파일에서 여러 쿼리 실행 (결과 → batch_result.md 저장)
python total_process_batch.py --batch queries.txt

# 쿼리 캐시 끄기
python total_process_batch.py -q 기넥신 --no-cache

# 쿼리 캐시 유사도 임계값 조정 (기본 0.92)
python total_process_batch.py -q 기넥신 --cache-threshold 0.85

# 저장된 쿼리 캐시 목록 확인
python total_process_batch.py --show-cache
```

**쿼리 모드 설명:**

| 모드     | 설명                           | 적합한 질문 유형    |
| -------- | ------------------------------ | ------------------- |
| `naive`  | 벡터 검색만 사용               | 단순 유사도 검색    |
| `local`  | 엔티티 주변 로컬 그래프 탐색   | 특정 개체 상세 정보 |
| `global` | 전체 그래프 커뮤니티 요약 활용 | 종합적 개요         |
| `hybrid` | local + global 결합 (기본값)   | 일반적인 질문       |

**배치 쿼리 파일 형식 (`queries.txt`):**

```
# '#'으로 시작하는 줄은 주석
기넥신 누구한테 영업할까?
외과에 어떤 약을 추천할까
류마티스 관절염에 도움이 되는 약
```

---

### 5-5. 통계 / 시각화

```bash
# 그래프 통계 출력
python total_process_batch.py --stats

# HTML 시각화 생성 (knowledge_graph.html)
python total_process_batch.py --visualize

# 시각화 노드 수 제한 (기본 1000)
python total_process_batch.py --visualize --max-nodes 500
```

**통계 출력 예시:**

```
================================================================
  그래프 통계  [최종]
================================================================
  노드: 762개  |  엣지: 2,341개
  컴포넌트: 45개  |  고립 노드: 162개 (21.3%)
  평균 연결도: 6.14  |  최대 연결도: 87

  [타입 분포]
    Concept                  210개  ##############################
    Person                   185개  ###########################
    Organization             140개  ####################
    ...

  [연결도 상위 10]
    기넥신                   [Artifact    ] 연결=87
    ...
================================================================
```

---

### 5-6. 실시간 모드 (배치 없이)

배치 API를 사용하지 않고 `total_process.py`와 동일하게 동작합니다.

```bash
python total_process_batch.py --no-batch
```

단건 테스트나 소규모 문서(1~2개)에서 즉시 결과가 필요할 때 사용합니다.

---

## 6. 힐링 전략 상세

힐링은 그래프에서 **고립 노드 (degree=0, 엣지가 없는 노드)** 를 해소하는 과정입니다.

### 전략 A: Prune (삭제)

**조건:**

- 설명(`description`)이 비어 있는 노드
- 타입이 `UNKNOWN`이고 설명이 `--prune-min-desc`(기본 10)자 미만인 노드

**효과:** 노이즈 제거. 그래프 크기가 줄어듦.

```bash
python total_process_batch.py --heal-prune --prune-min-desc 20
```

---

### 전략 B: Embed (유사도 연결)

임베딩 코사인 유사도를 계산하여 의미적으로 유사한 고립 노드를 기존 노드에 연결합니다.

**파라미터:**

- `--embed-threshold` (기본 0.75): 연결 최소 유사도
- `--embed-top-k` (기본 2): 노드당 최대 연결 수

```bash
python total_process_batch.py --heal-embed \
  --embed-threshold 0.80 \
  --embed-top-k 3
```

**추가되는 엣지 속성:**

```
relation_name = "semantic_similarity"
weight        = 코사인 유사도 값
```

---

### 전략 C: LLM 관계 제안 (Batch API)

고립 노드와 연결 후보 노드 목록을 LLM에게 보내 관계를 제안받습니다. 전략 C+D가 Batch API로 처리됩니다.

**파라미터:**

- `--llm-limit` (기본 50): 처리할 고립 노드 최대 수
- `--llm-min-confidence` (기본 0.5): 관계 신뢰도 최소값

```bash
python total_process_batch.py --heal-llm \
  --llm-limit 100 \
  --llm-min-confidence 0.6
```

LLM 응답 형식 (JSON 배열):

```json
[
  {
    "isolated_id": "노드ID",
    "candidate_id": "후보노드ID",
    "relation": "처방 대상",
    "confidence": 0.85,
    "description": "근거 설명"
  }
]
```

---

### 전략 D: Re-link 소스 재추출 (Batch API)

고립 노드의 `source_id`로 원본 청크를 찾아 해당 텍스트에서 관계를 재추출합니다. `--heal-all` 또는 `--heal-relink`로 활성화합니다.

**파라미터:**

- `--relink-limit` (기본 30): 처리할 노드 최대 수

```bash
python total_process_batch.py --heal-relink --relink-limit 50
```

---

## 7. 전체 CLI 옵션 레퍼런스

### 공통 옵션

| 옵션                | 기본값 | 설명                |
| ------------------- | ------ | ------------------- |
| `--llm {mini\|4o}`  | `mini` | 사용할 LLM 모델     |
| `--chunk-size N`    | `1000` | 청크 토큰 크기      |
| `--chunk-overlap N` | `150`  | 청크 오버랩 토큰 수 |

### 배치 제어

| 옵션                           | 설명                                  |
| ------------------------------ | ------------------------------------- |
| `--no-batch`                   | 배치 API 비활성화, 실시간 모드로 전환 |
| `--submit-only`                | 배치 제출 후 즉시 종료 (폴링 안 함)   |
| `--resume {insert\|heal\|all}` | 이전에 제출한 배치 재개               |
| `--poll-interval N`            | 배치 폴링 간격 (기본 60초)            |
| `--batch-status`               | 저장된 배치 상태 조회 후 종료         |

### 삽입 제어

| 옵션            | 설명                                   |
| --------------- | -------------------------------------- |
| `--skip-insert` | 삽입 단계 건너뜀 (그래프 이미 있을 때) |

### 힐링 옵션

| 옵션                     | 설명                                          |
| ------------------------ | --------------------------------------------- |
| `--heal`                 | 힐링 실행 (전략 A+B+C)                        |
| `--heal-all`             | 힐링 실행 (전략 A+B+C+D)                      |
| `--heal-prune`           | 전략 A(Prune)만 실행                          |
| `--heal-embed`           | 전략 B(Embed)만 실행                          |
| `--heal-llm`             | 전략 C(LLM)만 실행                            |
| `--heal-relink`          | 전략 D(Re-link)만 실행                        |
| `--batch-heal`           | 힐링 C+D를 Batch API로 실행 (`--heal`과 함께) |
| `--dry-run`              | 그래프 변경 없이 결과 미리보기                |
| `--isolated-detail`      | 힐링 전 고립 노드 목록 상세 출력              |
| `--prune-min-desc N`     | Prune 기준: 설명 최소 길이 (기본 10)          |
| `--embed-threshold F`    | Embed 기준: 코사인 유사도 임계값 (기본 0.75)  |
| `--embed-top-k N`        | Embed: 노드당 최대 연결 수 (기본 2)           |
| `--llm-limit N`          | LLM 처리 노드 최대 수 (기본 50)               |
| `--llm-min-confidence F` | LLM 신뢰도 최소값 (기본 0.5)                  |
| `--relink-limit N`       | Re-link 처리 노드 최대 수 (기본 30)           |

### 쿼리 옵션

| 옵션                                    | 설명                                      |
| --------------------------------------- | ----------------------------------------- |
| `-q / --query WORD...`                  | 단건 쿼리 실행                            |
| `--mode {naive\|local\|global\|hybrid}` | 쿼리 모드 (기본 hybrid)                   |
| `--mode-compare WORD...`                | 4가지 모드 모두 비교 실행                 |
| `--batch FILE`                          | 파일에서 다건 쿼리 실행, 결과를 MD로 저장 |
| `--no-cache`                            | 쿼리 캐시 비활성화                        |
| `--cache-threshold F`                   | 캐시 히트 유사도 임계값 (기본 0.92)       |
| `--show-cache`                          | 저장된 쿼리 캐시 목록 출력                |

### 통계 / 시각화

| 옵션            | 설명                            |
| --------------- | ------------------------------- |
| `--stats`       | 그래프 통계 출력 후 종료        |
| `--visualize`   | HTML 시각화 생성 후 종료        |
| `--max-nodes N` | 시각화 최대 노드 수 (기본 1000) |

---

## 8. 비용 구조 및 절감 계산

### 모델별 단가

| 모델        | 실시간 입력 | 실시간 출력 | 배치 입력 | 배치 출력 |
| ----------- | ----------- | ----------- | --------- | --------- |
| gpt-4o-mini | $0.150/1M   | $0.600/1M   | $0.075/1M | $0.300/1M |
| gpt-4o      | $2.500/1M   | $10.000/1M  | $1.250/1M | $5.000/1M |

### 최종 비용 요약 출력

```
========================================================
[전체 비용 최종 요약]
  모델: gpt-4o-mini
  삽입 (실시간 LLM)  : $0.00120  (= ₩166)
  삽입 (배치 API)    : $0.07540  (= ₩10,405)
  힐링 (실시간 LLM)  : $0.00021  (= ₩29)
  힐링 (배치 API)    : $0.00320  (= ₩442)
  쿼리               : $0.00580  (= ₩800)
  ────────────────────────────────────────────────
  합계               : $0.08581  (= ₩11,842)
  배치 절감          : $0.07860  (= ₩10,847)  (48% 절감)
  총 소요 시간       : 3842.5초
========================================================
```

### 삽입 파일별 요약 출력

```
  파일명                                     | 실시간 |  배치 |   입력(실시간) |   입력(배치) |     비용($) |   시간
  약품A.md                                   |      3 |    12 |         4,200 |       18,000 |    0.00740 |  62.3초
  약품B.md                                   |      2 |     8 |         2,800 |       12,000 |    0.00491 |  41.1초
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────
  합계                                       |      5 |    20 |         7,000 |       30,000 |    0.01231 | 103.4초

  배치 처리 비율: 80%  (실시간 $0.00420 + 배치 $0.00811)
```

---

## 9. 내부 구조 설명

### BatchJobManager

OpenAI Batch API의 전체 생명주기를 관리합니다.

```
add_request(custom_id, messages)  → 요청 목록에 추가
submit(description)               → JSONL 업로드 + 배치 생성 → batch_id 반환
check_status(batch_id)            → 현재 상태 조회
poll_until_done(batch_id)         → 완료될 때까지 폴링 → {custom_id: 응답} 반환
save_state(type, batch_id)        → batch_state.json에 저장 (재개용)
load_state(type)                  → batch_state.json에서 로드
clear_state(type)                 → 완료 후 상태 삭제
```

### BatchCachedLLM

배치 결과를 LightRAG의 `llm_model_func`에 주입하는 어댑터입니다.

```
캐시 키 = MD5(user_prompt)
캐시 히트 → 배치 단가로 비용 집계, 즉시 반환
캐시 미스 → 실시간 API 폴백 (gleaning/merge 등)
```

### 상태 파일 (`batch_state.json`)

```json
{
  "insert": {
    "batch_id": "batch_abc123",
    "submitted_at": "2026-04-27T15:30:00",
    "req_count": 120,
    "hash_map": { "cid_0001": "md5hash...", ... },
    "md_files": ["약품A.md", "약품B.md"]
  },
  "heal_c": {
    "batch_id": "batch_def456",
    "submitted_at": "2026-04-27T16:00:00"
  }
}
```

---

## 10. 자주 묻는 질문

**Q. 배치가 24시간이 지났는데 expired 됐다고 합니다.**

`--resume`으로 재개하면 `RuntimeError: 배치가 expired 상태로 종료` 오류가 납니다. 이 경우 처음부터 다시 제출해야 합니다.

```bash
python total_process_batch.py  # 재제출
```

---

**Q. `--submit-only`로 제출했는데 `batch_state.json`이 없어졌습니다.**

`WORKING_DIR` 위치에 생성됩니다. 경로를 확인하세요.

```bash
python total_process_batch.py --batch-status
```

---

**Q. 삽입은 이미 됐는데 힐링만 다시 하고 싶습니다.**

```bash
python total_process_batch.py --heal --skip-insert
```

또는 힐링만 단독 실행합니다.

```bash
python total_process_batch.py --heal-all
```

---

**Q. 배치를 쓰지 않고 즉시 결과를 보고 싶습니다.**

```bash
python total_process_batch.py --no-batch
```

`total_process.py`와 동일하게 동작합니다.

---

**Q. 쿼리 캐시가 히트되지 않아 비용이 계속 나갑니다.**

`--cache-threshold`를 낮추면 더 유사한 기존 쿼리를 히트로 인식합니다.

```bash
python total_process_batch.py -q "질문" --cache-threshold 0.85
```

저장된 캐시 목록을 보려면:

```bash
python total_process_batch.py --show-cache
```

---

**Q. 그래프가 너무 커서 시각화가 느립니다.**

```bash
python total_process_batch.py --visualize --max-nodes 300
```

상위 연결도 300개 노드만 표시합니다.

---

**Q. gpt-4o로 품질을 높이고 싶습니다.**

```bash
python total_process_batch.py --llm 4o
```

또는 CONFIG에서 `LLM_MODEL = "4o"`로 변경합니다. 배치 단가: 입력 $1.250/1M, 출력 $5.000/1M.
