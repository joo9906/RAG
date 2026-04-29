# total_process.py 사용 가이드

LightRAG + OpenAI Batch API 통합 파이프라인입니다. 문서를 Knowledge Graph로 변환하고, LLM 비용을 최대 50% 절감합니다.

---

## 목차

1. [개요](#1-개요)
2. [디렉터리 구조 및 사전 준비](#2-디렉터리-구조-및-사전-준비)
3. [CONFIG 설정](#3-config-설정)
4. [입력 데이터 형식 (JSONL)](#4-입력-데이터-형식-jsonl)
5. [전체 파이프라인 구조](#5-전체-파이프라인-구조)
6. [실행 모드별 사용법](#6-실행-모드별-사용법)
7. [힐링 전략 상세](#7-힐링-전략-상세)
8. [Retrieve-only 사용법 (LLM 답변 없이 검색만)](#8-retrieve-only-사용법-llm-답변-없이-검색만)
9. [Langfuse 옵저버빌리티](#9-langfuse-옵저버빌리티)
10. [전체 CLI 옵션 레퍼런스](#10-전체-cli-옵션-레퍼런스)
11. [비용 구조 및 출력 예시](#11-비용-구조-및-출력-예시)
12. [내부 구조 설명](#12-내부-구조-설명)
13. [자주 묻는 질문](#13-자주-묻는-질문)

---

## 1. 개요

### LightRAG란?

LightRAG는 텍스트에서 엔티티(노드)와 관계(엣지)를 추출해 **Knowledge Graph**를 구축하고, 쿼리 시 그래프 탐색 + 벡터 검색을 결합해 답변을 생성하는 RAG 프레임워크입니다.

- **그래프 저장**: GraphML 파일 (`working_dir/graph_chunk_entity_relation.graphml`)
- **벡터 저장**: Qdrant (청크 및 엔티티 임베딩)

### Batch API 적용 범위

OpenAI Batch API는 LLM 요청을 JSONL로 묶어 비동기 제출하면 최대 24시간 내에 처리, **비용 50% 절감**합니다.

| 단계 | 처리 방식 | 이유 |
|------|-----------|------|
| 삽입: 엔티티 추출 | **배치** | 청크별로 독립적, 순서 무관 |
| 삽입: gleaning/merge | 실시간 | 이전 추출 결과에 의존 |
| 힐링 전략 A (Prune) | 실시간 | LLM 불필요 |
| 힐링 전략 B (Embed) | 실시간 (Embedding API) | 빠른 처리 |
| 힐링 전략 C (LLM) | **배치** 또는 실시간 | `--no-batch` 여부에 따라 |
| 힐링 전략 D (Re-link) | **배치** 또는 실시간 | `--no-batch` 여부에 따라 |
| 쿼리 | 실시간 | 즉시 결과 필요 |

---

## 2. 디렉터리 구조 및 사전 준비

### 디렉터리 구조

```
03_LightRAG/
├── total_process.py             ← 메인 스크립트
├── .env.json                    ← API 키 및 서버 설정
├── chunked_docs/                ← 입력 데이터 (JSONL 파일들)
│   ├── 페브릭환자용.jsonl
│   ├── 기넥신PSS가이드북.jsonl
│   └── ...
├── lightrag_before_chunk_test/  ← WORKING_DIR (자동 생성)
│   ├── graph_chunk_entity_relation.graphml   ← Knowledge Graph
│   ├── kv_store_full_docs.json
│   ├── kv_store_text_chunks.json
│   ├── kv_store_full_entities.json
│   ├── kv_store_full_relations.json
│   ├── query_cache.json         ← 쿼리 캐시
│   ├── batch_state.json         ← 배치 상태 저장 (자동 생성)
│   └── knowledge_graph.html     ← 시각화 결과
└── docs/
    └── guide.md
```

### `.env.json` 구조

```json
{
  "openai_api_key": "sk-proj-...",
  "langfuse_public_key": "pk-lf-...",
  "langfuse_secret_key": "sk-lf-...",
  "langfuse_host": "https://cloud.langfuse.com",
  "qdrant": {
    "host": "localhost",
    "port": "6333"
  }
}
```

> **Qdrant URL 우선순위**: 환경변수 `QDRANT_URL` → `.env.json`의 `qdrant.host/port` → `http://localhost:6333`

### 필수 패키지

```bash
pip install -r requirements.txt
```

Qdrant를 Docker로 로컬 실행하는 경우:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

## 3. CONFIG 설정

**파일 상단 CONFIG 블록만 수정**하면 전체 파이프라인이 맞춰 동작합니다.

```python
# [1] 데이터 경로
WORKING_DIR       = ".../lightrag_before_chunk_test"  # 그래프 저장 위치
MD_DIR            = ".../chunked_docs"                 # 입력 JSONL 파일 폴더
QDRANT_URL        = "http://localhost:6333"            # env 없을 때 기본값
QDRANT_COLLECTION = "lightrag_before_chunk_test"       # Qdrant 컬렉션명

# [2] LLM 모델
LLM_MODEL = "mini"   # "mini" = gpt-4o-mini | "4o" = gpt-4o

# [3] 임베딩 모델
EMB_MODEL = "text-embedding-3-large"
EMB_DIM   = 2048     # Qdrant 컬렉션 생성 시 이 차원으로 고정됨

# [4] 입력 형식 및 청크 설정
INPUT_FORMAT   = "jsonl"   # "jsonl" | "md"
JSONL_TEXT_KEY = "text"    # JSONL에서 본문을 읽는 필드명 (fallback: "chunk_text")
CHUNK_SIZE     = 1000      # MD 모드에서만 사용 (JSONL 모드에서는 무시됨)
CHUNK_OVERLAP  = 150       # MD 모드에서만 사용

# [5] 쿼리 캐시
CACHE_SIMILARITY_THRESHOLD = 0.85   # 캐시 히트 코사인 유사도 임계값

# [6] 힐링 파라미터
HEAL_PRUNE_MIN_DESC  = 10    # Prune: 설명 최소 길이 (자)
HEAL_EMBED_THRESHOLD = 0.75  # Embed: 연결 최소 코사인 유사도
HEAL_EMBED_TOP_K     = 2     # Embed: 노드당 최대 연결 수
HEAL_LLM_LIMIT       = 50    # LLM: 처리할 고립 노드 최대 수
HEAL_LLM_MIN_CONF    = 0.5   # LLM: 관계 신뢰도 최소값
HEAL_RELINK_LIMIT    = 30    # Re-link: 처리할 노드 최대 수

# [7] 배치 API 설정
BATCH_POLL_INTERVAL  = 60    # 폴링 간격 (초)
BATCH_COMPLETION_WIN = "24h" # OpenAI 배치 완료 창
```

---

## 4. 입력 데이터 형식 (JSONL)

현재 `INPUT_FORMAT = "jsonl"` 모드로 동작합니다. `chunked_docs/` 디렉터리의 `.jsonl` 파일을 읽습니다.

### JSONL 스키마

각 라인이 하나의 청크입니다. 현재 파이프라인이 사용하는 필드:

```json
{
  "text": "본문 텍스트 (핵심 필드)",
  "etc": ["<chart>차트 설명...</chart>", "이미지 설명..."],
  "tables": ["표 내용..."],
  "doc_name": "문서명",
  "chunk_id": "aa5cc884-838b-5171-b011-d8114b27ec9a_chunk_0",
  "parent_id": null,
  "schema_type": "product",
  "product_names": [],
  "tags": [],
  "questions": [],
  "refs": ["참고문헌1", "참고문헌2"],
  "created_at": "2026-04-28T15:46:10+09:00"
}
```

### 필드 활용 방식

| 필드 | 활용 | 비고 |
|------|------|------|
| `text` | 본문으로 사용 | 핵심 필드 |
| `etc` | 차트·이미지 설명을 본문 뒤에 추가 | 시각적 정보 보완 |
| `tables` | 표 내용을 본문 뒤에 추가 | |
| `doc_name`, `chunk_id` | LightRAG 내부 소스 추적 | 직접 주입 불필요 |
| `refs`, `questions`, `tags` | 미사용 | 엔티티 추출 노이즈 방지 |

### LightRAG 내부 청킹 비활성화

JSONL 모드에서는 데이터가 **이미 청킹된 상태**이므로, LightRAG가 다시 청크로 쪼개지 않도록 내부 청킹을 비활성화합니다.

```python
# JSONL 모드: chunk_token_size=10,000,000 으로 재청킹 방지
# MD 모드:    chunk_token_size=CHUNK_SIZE (1000 토큰) 사용
```

각 JSONL 라인의 `text`를 개별적으로 `rag.ainsert(text)` 에 넘기므로, LightRAG 입장에서는 "매우 짧은 단일 청크" 한 개를 받는 것과 동일합니다.

### MD 모드로 전환

```python
# CONFIG에서
INPUT_FORMAT = "md"
```

`chunked_docs/` 안의 `.md` 파일을 읽어 내부적으로 `CHUNK_SIZE`/`CHUNK_OVERLAP` 기준으로 청킹합니다.

---

## 5. 전체 파이프라인 구조

```
chunked_docs/*.jsonl
        │
        ▼
각 라인(청크)에서 entity extraction 프롬프트 생성
        │
        ▼
┌─────────────────────┐
│  OpenAI Batch API   │  ← 최대 24시간 (--no-batch 시 실시간)
│  (JSONL 업로드/폴링) │
└─────────────────────┘
        │
        ▼
BatchCachedLLM (캐시 재주입)
→ LightRAG ainsert() 호출
  ├─ entity extraction → 캐시 히트 (비용 없음)
  └─ gleaning/merge   → 실시간 API (비율 낮음)
        │
        ▼
Knowledge Graph (GraphML) + Qdrant 벡터 저장
        │
        ▼
힐링: Prune → Embed → LLM → Re-link
        │
        ▼
HTML 시각화 / 통계 출력
        │
        ▼
쿼리 (실시간, 또는 Retrieve-only)
```

### 임베딩이 사용되는 위치

| 위치 | 내용 | 비용 비중 |
|------|------|-----------|
| 삽입 | 청크·엔티티 벡터화 → Qdrant 저장 | 높음 |
| 힐링 B | 고립 노드 ↔ 연결 노드 코사인 유사도 계산 | 낮음 |
| 쿼리 캐시 | 쿼리 텍스트 벡터화 → 이전 쿼리 유사도 비교 | 매우 낮음 |

---

## 6. 실행 모드별 사용법

### 6-1. 전체 파이프라인 (기본)

배치 삽입 → 배치 힐링 → 시각화 → 통계 → 쿼리 순으로 전부 실행합니다.

```bash
python total_process.py
```

### 6-2. 배치 제출만 하고 나중에 재개

터미널을 닫아도 되는 워크플로우입니다.

```bash
# Step 1: 배치 제출만 하고 즉시 종료
python total_process.py --submit-only

# Step 2: (선택) 상태 확인
python total_process.py --batch-status

# Step 3: 완료 후 재개
python total_process.py --resume insert   # 삽입만 재개
python total_process.py --resume heal     # 힐링만 재개
python total_process.py --resume all      # 삽입 + 힐링 모두 재개
```

`batch_state.json` 예시:

```json
{
  "insert": {
    "batch_id": "batch_abc123",
    "submitted_at": "2026-04-27T15:30:00",
    "req_count": 320,
    "hash_map": { "cid_0001": "md5hash...", ... },
    "input_files": ["페브릭환자용.jsonl", "기넥신PSS.jsonl"]
  }
}
```

### 6-3. 삽입 건너뛰고 힐링만

그래프가 이미 구축된 경우 힐링만 실행합니다.

```bash
# 기본 힐링 (A+B+C, 실시간)
python total_process.py --heal

# 전략 D(Re-link)까지 포함
python total_process.py --heal-all

# 특정 전략만
python total_process.py --heal-prune
python total_process.py --heal-embed
python total_process.py --heal-llm
python total_process.py --heal-relink

# 힐링 C+D를 Batch API로 (기본은 실시간)
python total_process.py --heal --batch-heal

# 건식 실행 (그래프 변경 없이 미리보기)
python total_process.py --heal --dry-run

# 고립 노드 목록 상세 출력
python total_process.py --heal --isolated-detail
```

### 6-4. 쿼리

```bash
# 단건 쿼리 (기본 hybrid 모드)
python total_process.py -q 기넥신 누구한테 써?

# 모드 지정
python total_process.py -q 외과 추천 약 --mode local
python total_process.py -q 류마티스 관절염 --mode global

# 4가지 모드 비교
python total_process.py --mode-compare 기넥신 적응증

# 파일에서 다건 쿼리 (결과 → batch_result.md 저장)
python total_process.py --batch queries.txt

# 캐시 끄기
python total_process.py -q 기넥신 --no-cache

# 저장된 쿼리 캐시 목록 확인
python total_process.py --show-cache
```

**쿼리 모드:**

| 모드 | 설명 | 적합한 질문 |
|------|------|-------------|
| `naive` | 벡터 검색만 | 단순 유사도 검색 |
| `local` | 엔티티 주변 로컬 그래프 탐색 | 특정 개체 상세 정보 |
| `global` | 전체 그래프 커뮤니티 요약 활용 | 종합적 개요 |
| `hybrid` | local + global 결합 (기본값) | 일반적인 질문 |
| `mix` | 그래프 데이터 + 벡터 검색 청크 | 맥락이 필요한 질문 |

### 6-5. 실시간 모드 (배치 없이)

배치 API 없이 모든 LLM 호출을 즉시 실행합니다. 소규모 테스트에 적합합니다.

```bash
python total_process.py --no-batch
```

### 6-6. 통계 / 시각화

```bash
python total_process.py --stats
python total_process.py --visualize
python total_process.py --visualize --max-nodes 500
```

---

## 7. 힐링 전략 상세

힐링은 그래프에서 **고립 노드 (degree=0, 엣지가 없는 노드)** 를 해소하는 과정입니다. 엔티티 추출 과정에서 항상 일정 비율의 고립 노드가 생기므로, 삽입 후 힐링을 거치면 그래프 품질이 개선됩니다.

### 전략 A: Prune (삭제)

다음 조건의 고립 노드를 삭제합니다.
- 설명(`description`)이 비어 있는 노드
- 타입이 `UNKNOWN`이고 설명이 `--prune-min-desc`(기본 10)자 미만인 노드

```bash
python total_process.py --heal-prune --prune-min-desc 20
```

### 전략 B: Embed (유사도 연결)

임베딩 코사인 유사도로 고립 노드를 가장 유사한 기존 노드에 연결합니다. LLM 없이 빠르게 처리됩니다.

- `--embed-threshold` (기본 0.75): 연결 최소 유사도
- `--embed-top-k` (기본 2): 노드당 최대 연결 수

추가 엣지 속성: `relation_name = "semantic_similarity"`, `weight = 유사도값`

### 전략 C: LLM 관계 제안

고립 노드와 후보 노드 목록을 LLM에 보내 관계를 제안받습니다.

- `--llm-limit` (기본 50): 처리할 고립 노드 최대 수
- `--llm-min-confidence` (기본 0.5): 관계 신뢰도 최소값
- `--no-batch` 없이 실행하면 기본적으로 실시간; `--batch-heal` 추가 시 배치 API 사용

LLM 응답 형식:

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

### 전략 D: Re-link 소스 재추출

고립 노드의 원본 청크 텍스트에서 관계를 재추출합니다. `--heal-all` 또는 `--heal-relink`로 활성화.

- `--relink-limit` (기본 30): 처리할 노드 최대 수

---

## 8. Retrieve-only 사용법 (LLM 답변 없이 검색만)

LightRAG의 검색(Retrieve) 단계와 답변 생성(LLM) 단계는 분리되어 있습니다. `QueryParam`의 옵션을 통해 LLM 호출 없이 검색 결과만 가져올 수 있습니다.

### 옵션 1: `only_need_context=True` — 검색된 컨텍스트 텍스트만 반환

```python
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=tracked_llm,
    embedding_func=EmbeddingFunc(
        embedding_dim=EMB_DIM,
        max_token_size=EMB_MAX_TOKENS,
        func=tracked_embed,
        model_name=EMB_MODEL,
    ),
    vector_storage="QdrantVectorDBStorage",
    vector_db_storage_cls_kwargs={"collection_name": QDRANT_COLLECTION},
)
await rag.initialize_storages()

result = await rag.aquery(
    "기넥신 적응증은?",
    param=QueryParam(
        mode="hybrid",
        only_need_context=True,   # ← LLM 호출 없이 컨텍스트만 반환
    )
)

# result.content: 검색된 엔티티·관계·청크를 포맷한 텍스트
# result.raw_data: 구조화된 원본 데이터 (엔티티, 관계, 청크 목록)
print(result.content)
```

### 옵션 2: `only_need_prompt=True` — LLM에 넘길 완성된 프롬프트 반환

```python
result = await rag.aquery(
    "기넥신 적응증은?",
    param=QueryParam(
        mode="hybrid",
        only_need_prompt=True,   # ← 시스템 프롬프트 + 컨텍스트 + 쿼리 전체 반환
    )
)

# result.content: "--- System Prompt ---\n...\n--- User Query ---\n기넥신 적응증은?"
# 자체 LLM으로 직접 호출하거나 프롬프트 검사에 활용
print(result.content)
```

### 옵션 3: 자체 LLM으로 후처리

```python
# 컨텍스트 검색 후 자체 로직으로 처리하는 패턴
result = await rag.aquery(query, param=QueryParam(mode="hybrid", only_need_context=True))

context_text = result.content
raw_data      = result.raw_data   # {"entities": [...], "relations": [...], "chunks": [...]}

# 원하는 방식으로 LLM 호출
response = await my_llm_client.chat(
    system="당신은 의약품 전문가입니다.",
    user=f"다음 자료를 참고해 답하시오:\n\n{context_text}\n\n질문: {query}"
)
```

### raw_data 구조

`result.raw_data`에는 검색된 원본 데이터가 들어 있습니다:

```python
{
  "entities": [
    {"entity_name": "기넥신", "entity_type": "Artifact", "description": "...", ...},
    ...
  ],
  "relations": [
    {"src_id": "기넥신", "tgt_id": "뇌혈관질환", "keywords": "적응증", ...},
    ...
  ],
  "chunks": [
    {"content": "기넥신은 뇌혈관 질환에 사용...", "chunk_id": "...", ...},
    ...
  ]
}
```

### 독립 서비스로 분리하는 경우

삽입(인덱싱)은 이미 완료된 상태에서 검색 서비스만 별도로 띄울 수 있습니다.

```python
# FastAPI 예시
@app.post("/retrieve")
async def retrieve(query: str, mode: str = "hybrid"):
    result = await rag.aquery(
        query,
        param=QueryParam(mode=mode, only_need_context=True)
    )
    return {
        "context": result.content,
        "entities": result.raw_data.get("entities", []),
        "chunks":   result.raw_data.get("chunks", []),
    }
```

필요한 것:
- `WORKING_DIR`의 GraphML + kv_store JSON 파일들
- Qdrant 컬렉션 (삽입 시 채워진 상태)
- `rag.initialize_storages()` 호출 (파일 및 Qdrant 로드)

---

## 9. Langfuse 옵저버빌리티

모든 LLM 호출, 임베딩, 파일 삽입이 Langfuse에 자동으로 기록됩니다.

### 설정

`.env.json`에 Langfuse 키를 추가하면 자동으로 활성화됩니다:

```json
{
  "langfuse_public_key": "pk-lf-...",
  "langfuse_secret_key": "sk-lf-...",
  "langfuse_host": "https://cloud.langfuse.com"
}
```

키가 없으면 Langfuse 없이 정상 동작합니다 (no-op fallback).

### 추적 구조

```
insert_file (trace)
  ├── llm_batch_cache (generation)   ← 배치 캐시 히트
  ├── llm (generation)               ← 실시간 LLM 호출 (gleaning/merge)
  └── embed (span)                   ← 임베딩 호출

query (trace)
  ├── llm (generation)
  └── embed (span)

heal (trace)
  ├── heal_strategy_c (span)
  └── heal_strategy_d (span)
```

### 주의사항

- Langfuse 환경변수는 **다른 SDK import 전에** 주입되어야 합니다. 이미 코드 최상단에서 처리하고 있습니다.
- 프로세스 종료 전 `langfuse_context.flush()` 가 자동 호출됩니다 (`print_total_cost()` 내부).

---

## 10. 전체 CLI 옵션 레퍼런스

### 공통

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--llm {mini\|4o}` | `mini` | LLM 모델 선택 |
| `--chunk-size N` | `1000` | MD 모드 청크 토큰 크기 |
| `--chunk-overlap N` | `150` | MD 모드 청크 오버랩 |

### 배치 제어

| 옵션 | 설명 |
|------|------|
| `--no-batch` | 배치 API 비활성화 (실시간 모드) |
| `--submit-only` | 배치 제출 후 즉시 종료 |
| `--resume {insert\|heal\|all}` | 이전 배치 재개 |
| `--poll-interval N` | 폴링 간격 초 (기본 60) |
| `--batch-status` | 저장된 배치 상태 조회 |

### 삽입 제어

| 옵션 | 설명 |
|------|------|
| `--skip-insert` | 삽입 건너뜀 (그래프 이미 있을 때) |

### 힐링

| 옵션 | 설명 |
|------|------|
| `--heal` | 힐링 A+B+C 실행 |
| `--heal-all` | 힐링 A+B+C+D 실행 |
| `--heal-prune` | 전략 A만 |
| `--heal-embed` | 전략 B만 |
| `--heal-llm` | 전략 C만 |
| `--heal-relink` | 전략 D만 |
| `--batch-heal` | 힐링 C+D를 Batch API로 (`--heal`과 함께) |
| `--dry-run` | 그래프 변경 없이 미리보기 |
| `--isolated-detail` | 고립 노드 목록 상세 출력 |
| `--prune-min-desc N` | Prune 설명 최소 길이 (기본 10) |
| `--embed-threshold F` | Embed 코사인 유사도 임계값 (기본 0.75) |
| `--embed-top-k N` | Embed 노드당 최대 연결 수 (기본 2) |
| `--llm-limit N` | LLM 처리 노드 최대 수 (기본 50) |
| `--llm-min-confidence F` | LLM 신뢰도 최소값 (기본 0.5) |
| `--relink-limit N` | Re-link 처리 노드 최대 수 (기본 30) |

### 쿼리

| 옵션 | 설명 |
|------|------|
| `-q / --query WORD...` | 단건 쿼리 |
| `--mode {naive\|local\|global\|hybrid\|mix}` | 쿼리 모드 (기본 hybrid) |
| `--mode-compare WORD...` | 전체 모드 비교 실행 |
| `--batch FILE` | 파일에서 다건 쿼리, 결과를 MD로 저장 |
| `--no-cache` | 쿼리 캐시 비활성화 |
| `--cache-threshold F` | 캐시 히트 유사도 임계값 (기본 0.85) |
| `--show-cache` | 저장된 쿼리 캐시 목록 출력 |

### 통계 / 시각화

| 옵션 | 설명 |
|------|------|
| `--stats` | 그래프 통계 출력 |
| `--visualize` | HTML 시각화 생성 |
| `--max-nodes N` | 시각화 최대 노드 수 (기본 1000) |

---

## 11. 비용 구조 및 출력 예시

### 모델별 단가

| 모델 | 실시간 입력 | 실시간 출력 | 배치 입력 | 배치 출력 |
|------|-------------|-------------|-----------|-----------|
| gpt-4o-mini | $0.150/1M | $0.600/1M | $0.075/1M | $0.300/1M |
| gpt-4o | $2.500/1M | $10.000/1M | $1.250/1M | $5.000/1M |
| text-embedding-3-large | $0.130/1M | — | — | — |

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

### 삽입 파일별 요약

```
  파일명                                     | 실시간 |  배치 |   입력(실시간) |   입력(배치) |     비용($) |   시간
  페브릭환자용.jsonl[0]                      |      1 |     4 |         1,200 |        6,000 |    0.00220 |  18.3초
  페브릭환자용.jsonl[1]                      |      1 |     3 |           980 |        4,500 |    0.00168 |  14.1초
  ─────────────────────────────────────────────────────────────────────────────────────────────────────────
  합계                                       |      5 |    20 |         7,000 |       30,000 |    0.01231 | 103.4초

  배치 처리 비율: 80%  (실시간 $0.00420 + 배치 $0.00811)
```

> JSONL 모드에서는 파일명이 `파일명.jsonl[청크인덱스]` 형식으로 표시됩니다.

---

## 12. 내부 구조 설명

### BatchJobManager

OpenAI Batch API 생명주기 관리.

```
add_request(custom_id, messages)  → 요청 목록에 추가
submit(description)               → JSONL 업로드 + 배치 생성 → batch_id 반환
check_status(batch_id)            → 현재 상태 조회
poll_until_done(batch_id)         → 완료될 때까지 폴링 → {custom_id: 응답} 반환
save_state(type, batch_id)        → batch_state.json 저장 (재개용)
load_state(type)                  → batch_state.json 로드
clear_state(type)                 → 완료 후 상태 삭제
```

### BatchCachedLLM

배치 결과를 LightRAG의 `llm_model_func`에 주입하는 어댑터.

```
캐시 키 = MD5(user_prompt)
캐시 히트 → 배치 단가로 집계, 즉시 반환 (API 호출 없음)
캐시 미스 → 실시간 API 폴백 (gleaning/merge 등)
```

### _load_jsonl_chunks(fpath)

JSONL 파일을 읽어 텍스트 청크 목록 반환. `text` → `etc`(차트 설명) → `tables` 순서로 내용을 합쳐 하나의 청크 문자열을 만듭니다.

### _iter_insert_units()

`INPUT_FORMAT`에 따라 삽입 단위 `(label, text)` 목록을 반환. JSONL이면 `파일명.jsonl[i]` 레이블로 각 라인을, MD이면 파일 단위로 반환.

### QueryCache

쿼리를 임베딩 벡터로 저장해두고, 다음 쿼리 시 코사인 유사도가 `CACHE_SIMILARITY_THRESHOLD` 이상이면 LLM 없이 캐시 응답 반환.

---

## 13. 자주 묻는 질문

**Q. JSONL 파일에서 `text` 필드가 아닌 다른 필드명을 쓰고 싶습니다.**

CONFIG에서 변경합니다:
```python
JSONL_TEXT_KEY = "chunk_text"   # 또는 원하는 필드명
```
지정한 필드가 없으면 `chunk_text` → 빈 문자열 순으로 fallback합니다.

---

**Q. 배치가 24시간이 지나 expired 됐습니다.**

처음부터 재제출해야 합니다. `--resume`으로는 복구되지 않습니다.

```bash
python total_process.py
```

---

**Q. 삽입은 됐는데 힐링만 다시 하고 싶습니다.**

```bash
python total_process.py --heal-all
```

---

**Q. 배치 없이 즉시 결과를 보고 싶습니다.**

```bash
python total_process.py --no-batch
```

소규모 테스트(파일 1~2개)에 적합합니다.

---

**Q. 쿼리 캐시가 잘 안 걸립니다.**

`--cache-threshold`를 낮추면 유사 쿼리를 더 잘 캐치합니다 (기본 0.85).

```bash
python total_process.py -q "질문" --cache-threshold 0.75
```

---

**Q. 그래프 통계에서 고립 노드 비율이 너무 높습니다.**

힐링을 순서대로 실행하면 점진적으로 줄어듭니다:

```bash
python total_process.py --heal-all --isolated-detail
```

전략 순서: Prune(노이즈 제거) → Embed(유사도 연결) → LLM(관계 제안) → Re-link(소스 재추출)

---

**Q. gpt-4o로 품질을 높이고 싶습니다.**

```bash
python total_process.py --llm 4o
```

또는 CONFIG에서 `LLM_MODEL = "4o"`. 배치 단가: 입력 $1.250/1M, 출력 $5.000/1M.
