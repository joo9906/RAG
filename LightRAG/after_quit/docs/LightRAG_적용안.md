# LightRAG 도입 설계 보고서 — skch-aix-ce 프로젝트 기준

작성일: 2026-04-23

---

## 0. 핵심 요약: 왜 LightRAG인가

> **현재 시스템의 근본 한계**: 7개 에이전트가 각자의 컬렉션에서 독립적으로 검색하고, 결과를 Supervisor LLM이 프롬프트로 조합한다. 문서 간 관계는 LLM의 추론에만 의존한다.
>
> **LightRAG가 바꾸는 것**: 문서 전체에 걸쳐 엔티티와 관계를 사전 추출해 그래프로 저장한다. 질문 시 관계를 직접 탐색해 LLM 없이도 연결된 지식을 찾는다.

---

## 1. 현재 시스템의 구조적 한계

### 1-1. 에이전트별 컬렉션 사일로

현재 7개 에이전트는 접근 가능한 컬렉션이 코드 레벨에서 고정되어 있다.

| 에이전트              | 접근 가능 컬렉션                                      | 접근 불가                    |
| --------------------- | ----------------------------------------------------- | ---------------------------- |
| ProductInformation    | ce*paper, ce_disease_knowledge, ce_company_product*\* | **ce_call, ce_cp**           |
| CallAnalysis          | **ce_call 단독**                                      | 논문, 컴플라이언스, 고성과자 |
| ComplianceManagement  | **ce_cp_qdrant 단독**                                 | 콜기록, 논문, 고성과자       |
| HighPerformerAnalysis | ce_high_performer_knowledge                           | 콜기록, 논문, 컴플라이언스   |

**핵심 문제**: 콜기록과 논문은 서로 접근하지 못한다. 컴플라이언스와 고성과자 전략도 단절되어 있다.

### 1-2. 현재 크로스 도메인 질문 처리 흐름

```
MR 질문: "기넥신을 자주 처방하는 김 원장님께 어떤 이야기를 해야 할까?"

현재 흐름:
  Supervisor → CallAnalysisAgent    → ce_call 검색 → "김 원장 콜기록: 기넥신 3회"
  Supervisor → ProductInformationAgent → ce_paper 검색 → "기넥신 임상근거"
  Supervisor → ComplianceManagement → ce_cp 검색 → "기넥신 홍보 규정"
  Supervisor → HighPerformerAnalysis → ce_high_performer 검색 → "고성과자 전략"
  Supervisor LLM → 4개 결과를 프롬프트에 넣고 종합

문제:
  - 4회의 독립적 LLM 호출
  - 각 에이전트는 다른 에이전트의 검색 결과를 모른 채 검색
  - "김 원장"과 "기넥신"의 연결 관계가 없으므로 콜기록 검색은 키워드 의존
  - 컴플라이언스에서 나온 "기넥신 금기사항"이 고성과자 전략과 연결되지 않음
  - 종합은 Supervisor LLM 프롬프트 길이에 의존 → 컨텍스트 초과 위험
```

### 1-3. 현재 벡터 검색의 본질적 한계

```
현재 RAG 동작 방식:
  질문 → 임베딩 → Qdrant 유사도 검색 → "이 청크가 질문과 얼마나 비슷한가?"

이 방식으로 답하기 어려운 질문 유형:
  ① "기넥신의 주성분이 포함된 다른 약물은?"
     → 주성분-약물 관계가 벡터에 없음. 동시 등장한 청크를 찾는 것에 의존.

  ② "A 병원에서 기넥신 처방률이 높아진 이유가 논문과 관련 있나?"
     → 콜기록(ce_call)과 논문(ce_paper)은 별개 컬렉션. 연결 검색 불가.

  ③ "기넥신 홍보 시 이 규정을 위반하지 않으면서 고성과자처럼 접근하려면?"
     → 컴플라이언스(ce_cp)와 고성과자 전략(ce_high_performer) 동시 탐색 불가.

  ④ "지난 콜에서 언급된 의사의 관심 질환과 관련된 최신 임상 근거는?"
     → 콜기록 내 의사 정보 → 논문으로 이어지는 2홉 탐색 불가.
```

---

## 2. LightRAG 도입 시 달라지는 것

### 2-1. 지식 그래프로 사일로 해소

LightRAG는 모든 문서에서 엔티티와 관계를 추출해 **단일 그래프**에 저장한다.

```
논문 적재 시 추출:
  (기넥신) --[임상근거]--> (말초순환장애)
  (기넥신) --[경쟁약물]--> (타나민)
  (기넥신) --[주성분]--> (은행엽 추출물)

컴플라이언스 적재 시 추출:
  (기넥신) --[홍보금지]--> (리베이트 제공)
  (기넥신) --[허용범위]--> (학술 심포지엄 지원)

콜기록 적재 시 추출 (복호화 후):
  (김OO 원장) --[근무]--> (삼성서울병원)
  (김OO 원장) --[처방]--> (기넥신)
  (김OO 원장) --[관심질환]--> (치매 예방)

고성과자 암묵지 적재 시 추출:
  (고성과자 전략A) --[활용근거]--> (기넥신 임상근거)
  (고성과자 전략A) --[타겟고객]--> (신경과 전문의)
```

이후 "기넥신" 노드 하나에서 출발하면 **컬렉션 경계 없이** 모든 연결 정보에 도달한다.

### 2-2. 같은 질문에 대한 처리 방식 비교

**질문: "기넥신을 자주 처방하는 김 원장님께 어떤 이야기를 해야 할까?"**

```
현재 방식:
  - 4개 에이전트 순차 또는 병렬 호출
  - 에이전트별 독립 검색 (연결 없음)
  - Supervisor가 4개 결과를 프롬프트에 넣고 LLM 종합
  - 총 LLM 호출: 4~6회 + 답변 생성 1회

LightRAG 방식:
  - 기넥신 노드 + 김OO 원장 노드 동시 검색
  - 그래프 탐색: 기넥신 → 임상근거, 경쟁약, 홍보규정, 고성과자전략
  - 그래프 탐색: 김원장 → 처방이력, 관심질환, 근무병원
  - 두 노드의 연결 경로에서 최적 컨텍스트 자동 수집
  - 총 LLM 호출: 1회 (답변 생성만)
```

**질문: "A 병원 기넥신 콜 시 컴플라이언스 위반 없이 접근하는 고성과자 방식은?"**

```
현재 방식:
  - ComplianceManagement: ce_cp에서 기넥신 규정 검색
  - HighPerformerAnalysis: ce_high_performer에서 전략 검색
  - 두 결과 간 연관성은 Supervisor LLM이 프롬프트에서 추론
  → "컴플라이언스 규정 X가 고성과자 전략 Y와 어떻게 연결되는가"는 LLM이 추측

LightRAG 방식:
  - 기넥신 노드 → ALLOWED_APPROACH 관계 탐색
  - 동일 그래프에 컴플라이언스 출처와 고성과자 출처가 모두 연결됨
  - "이 접근법은 컴플라이언스 허용 범위 내이며 고성과자 OOO이 사용한 방법"을
    관계 기반으로 직접 근거와 함께 제시
```

### 2-3. 쿼리 모드별 강점

LightRAG는 4가지 검색 모드를 제공한다. 현재 시스템은 naive 모드에만 해당한다.

| 모드       | 동작 방식                 | 현재 시스템 | 강점이 드러나는 질문 예시                         |
| ---------- | ------------------------- | :---------: | ------------------------------------------------- |
| **naive**  | 순수 벡터 유사도          |   ✓ 동일    | "기넥신 용법·용량은?"                             |
| **local**  | 엔티티 중심 로컬 탐색     |   ✗ 불가    | "기넥신과 직접 연결된 모든 근거는?"               |
| **global** | 그래프 전체 패턴 분석     |   ✗ 불가    | "우리 제품군 전체에서 가장 많이 연결된 적응증은?" |
| **hybrid** | 벡터 + 로컬 + 글로벌 통합 |   ✗ 불가    | "기넥신 경쟁 상황을 포괄적으로 분석해줘"          |

### 2-4. 지식 축적 효과

현재 시스템은 문서를 추가해도 **청크 수만 늘어날 뿐** 관계는 생기지 않는다.

```
현재: 논문 100편 → 청크 10,000개 → 유사도 검색 범위만 넓어짐
LightRAG: 논문 100편 → 엔티티 50,000개, 관계 40,000개
          → 그래프가 조밀해질수록 멀티홉 탐색 품질 향상
          → "처음 보는 조합의 질문"도 기존 지식 연결로 답변 가능
```

### 2-5. 에이전트 구조 개선 가능성

현재 에이전트 경계(컬렉션 접근 제한)를 유지하면서 **LightRAG를 보조 검색 레이어**로 추가할 수 있다.

```
기존 흐름 (변경 없음):
  각 에이전트 → 자기 컬렉션 검색 (빠른 단일 도메인 답변)

LightRAG 추가 흐름:
  크로스 도메인 질문 감지 시 → LightRAG hybrid 검색 → 에이전트 검색 보완

  예: "기넥신 관련 종합 전략" 질문
      → LightRAG가 논문+컴플라이언스+고성과자 연결 컨텍스트 제공
      → 개별 에이전트 호출 없이 1회 답변 가능
```

---

## 3. LightRAG가 해결하지 못하는 것 (한계 명시)

| 항목                                  |  현재 시스템   |         LightRAG          | 판단                        |
| ------------------------------------- | :------------: | :-----------------------: | --------------------------- |
| 실시간 MariaDB 쿼리 (매출, 고객 현황) |       ✓        |             ✗             | RDB 에이전트 유지 필수      |
| 단순 사실 검색 ("기넥신 용량은?")     |       ✓        |             ✓             | 차이 없음                   |
| FAQ 캐싱 (반복 질문 즉답)             |       ✓        |             ✗             | 기존 캐시 유지 필수         |
| 암호화 콜기록 처리                    |       ✓        |     별도 전처리 필요      | 복호화 파이프라인 추가 필요 |
| 최신 데이터 반영 속도                 | 즉시 (적재 시) | 적재 + 엔티티 추출 (지연) | LightRAG 적재 비용 존재     |
| 크로스 도메인 관계 탐색               |       ✗        |           **✓**           | LightRAG 핵심 강점          |
| 멀티홉 연결 질문                      |       ✗        |           **✓**           | LightRAG 핵심 강점          |
| 복수 문서 종합 (관계 기반)            |     부분적     |           **✓**           | LightRAG 핵심 강점          |

---

## 4. 도입 효과가 가장 큰 질문 유형 (현재 프로젝트 기준)

아래 질문들은 현재 시스템에서 에이전트 여러 개를 거치거나, 부정확하게 답변되는 유형이다.
LightRAG 도입 시 단일 쿼리로 처리 가능해진다.

```
[크로스 도메인 종합]
"기넥신 처방 시 컴플라이언스 주의사항과 고성과자 접근법을 같이 알려줘"
→ 현재: ComplianceAgent + HighPerformerAgent 2회 호출 → 수동 종합
→ LightRAG: 기넥신 노드에서 두 관계 동시 탐색 → 1회 답변

[멀티홉 관계 탐색]
"지난달 콜에서 관심 보인 의사들이 많이 쓰는 경쟁 약물의 논문 근거는?"
→ 현재: 콜기록 검색 → 경쟁약 파악 → 논문 검색 (3단계, 수동)
→ LightRAG: 의사 → 처방약 → 경쟁약 → 논문 (그래프 자동 탐색)

[암묵적 관계 질문]
"기넥신 주성분이 들어간 다른 제품은?"
→ 현재: 키워드 일치 의존 (주성분명이 청크에 동시 등장해야 함)
→ LightRAG: (기넥신) --[주성분]--> (은행엽 추출물) --[포함됨]--> (타제품) 직접 탐색

[전략 + 근거 연결]
"이 고성과자 전략의 임상적 근거 논문이 있어?"
→ 현재: ce_high_performer와 ce_paper는 연결 없음
→ LightRAG: 고성과자 전략 노드 → 인용 근거 관계 → 논문 노드 직접 연결
```

---

## 5. 현재 프로젝트 스택 (고정, 변경 불가)

### 인프라

| 구성 요소    | 현황                  | 역할                                         |
| ------------ | --------------------- | -------------------------------------------- |
| **Qdrant**   | 외부 서버 운영 중     | 벡터 임베딩 + 청크 텍스트 payload 저장       |
| **MariaDB**  | 외부 서버 운영 중     | 비즈니스 RDB (콜기록 메타, 고객, 제품, 매출) |
| **Redis**    | 외부 서버 운영 중     | 히스토리 요약 캐시 (TTL 86400s)              |
| **Redshift** | AWS, MariaDB fallback | 분석용 쿼리                                  |

### LLM 및 임베딩 (Azure OpenAI)

| 용도       | 모델                              | 비고            |
| ---------- | --------------------------------- | --------------- |
| 기본 LLM   | `gpt-4.1`                         | 답변 생성, 분석 |
| 경량 LLM   | `gpt-4.1-mini`                    | 요약, 분류 등   |
| 초경량 LLM | `gpt-4.1-nano`                    | 간단 처리       |
| 임베딩     | `text-embedding-3-large`          | 차원: **2048d** |
| 리전       | korea_central → sweden (fallback) | -               |

### 서버

```
uvicorn src.server.ce.main:app --host 0.0.0.0 --port 8000 --reload
workers: 1 (단일 프로세스)
```

---

## 6. 현재 Qdrant 컬렉션 전체 목록

전체 컬렉션명은 `qdrant_vector_store_{name}` 형태로 통일되어 있다.

| 컬렉션 논리명  | 물리 컬렉션명                                     | 저장 내용           | 암호화          |
| -------------- | ------------------------------------------------- | ------------------- | --------------- |
| ce_call        | `qdrant_vector_store_ce_call_encrypt`             | 콜기록 청크         | **Fernet+HKDF** |
| ce_email       | `qdrant_vector_store_ce_email_qdrant`             | 이메일 요약 청크    | 없음            |
| ce_paper       | `qdrant_vector_store_ce_paper_qdrant`             | 논문 청크           | 없음            |
| ce_cp          | `qdrant_vector_store_ce_cp_qdrant`                | 컴플라이언스 청크   | 없음            |
| ce_qa          | `qdrant_vector_store_ce_{lvl2}`                   | Q&A (lvl2별 분리)   | 없음            |
| ce_questionmap | `qdrant_vector_store_ce_questionmap`              | 질문 분류 노드      | 없음            |
| ce_faq_cache   | `qdrant_vector_store_ce_faq_cache`                | FAQ 응답 캐시       | 없음            |
| blue_cp        | `qdrant_vector_store_blue_cp`                     | 컴플라이언스 (blue) | 없음            |
| high_performer | `qdrant_vector_store_ce_high_performer_knowledge` | 고성과자 암묵지     | 없음            |

### 컬렉션별 주요 payload 필드

**ce_call_encrypt (암호화 대상)**

```
text (암호화), implication_info (암호화 list), act_text (암호화 list)
activity_id, employee_id, company_id, contact_id
act_date, company_name, contact_name, product_list
sales_type, call_type, activity_code, warning
```

**ce_paper_qdrant**

```
text, product_name, classification, paper_type
created_datetime, created_timestamp, filename
```

**ce_cp_qdrant**

```
text, document_name, page_number, file_path
created_datetime, created_timestamp, image_s3_keys
```

**ce_email_qdrant**

```
text, title, summary, product_name, disease_name
lvl1, lvl2, lvl3, validity
```

**ce_qa (lvl2별 분리 컬렉션)**

```
text, id, file_name, product_name, disease_name
lvl1, lvl2, lvl3, page_number, image_path
created_datetime, created_timestamp
```

---

## 7. 청킹 전략

### 7-1. LightRAG 기본 청킹

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,     # 문자 수 기준
    chunk_overlap=200,
)
# 예외: images/tables 메타데이터 포함 문서는 청킹 제외 (원본 보존)
```

배치 저장 단위:

- 일반 문서: `batch_size=100`
- 콜기록: `batch_size=50` (암호화 처리 부하)

### 7-2. 커스텀 청킹

문서를 현재 논의중인 방법으로 청킹 후 적재. 만약 문서의 길이가 짧은 경우(1200 미만)은 괜찮지만 길어질 경우 추가 방법이 필요(그대로 넣을지 말지)

---

## 8. 질문 그래프 현황 (기존 기능) - 이게 구현은 되어있는데 딱히 의미 있는 사용 X. 답변시에 필요한 필터링만 기재

```
nodes: 165개
edges: 197개
```

LightRAG의 지식 그래프와 완전히 별개다.

- **질문 그래프**: 사람이 수동 설계한 질문 분류 트리 (GraphML 파일, NetworkX 인메모리)
- **LightRAG 그래프**: LLM이 문서에서 자동 추출하는 엔티티-관계 지식 그래프

---

## 9. LightRAG 도입 설계

### 9-1. 저장소 역할 분리

```
Qdrant (기존 + LightRAG 전용 컬렉션 추가)
  ├─ [기존 RAG 파이프라인 — 재적재 후 동일 구조 유지]
  │    ├─ qdrant_vector_store_ce_call_encrypt
  │    ├─ qdrant_vector_store_ce_email_qdrant
  │    ├─ qdrant_vector_store_ce_paper_qdrant
  │    ├─ qdrant_vector_store_ce_cp_qdrant
  │    ├─ qdrant_vector_store_ce_{lvl2} (Q&A)
  │    ├─ qdrant_vector_store_ce_questionmap
  │    ├─ qdrant_vector_store_ce_faq_cache
  │    └─ qdrant_vector_store_blue_cp
  │
  └─ [LightRAG 전용 — 신규 추가]
       ├─ lightrag_entities_vdb     : 엔티티 임베딩 (local 모드 검색)
       ├─ lightrag_relationships_vdb: 관계 임베딩 (global 모드 검색)
       └─ lightrag_chunks_vdb       : 청크 임베딩 + 청크 텍스트 payload

Neo4j 또는 NetworkX (그래프 저장소 — 결정 필요)
  └─ LightRAG 전용
       ├─ Entity 노드: 약물, 병원, 의사, 적응증, 성분, 규정 등
       └─ Relationship 엣지: 처방, 경쟁, 함유, 금지, 권장 등

파일 기반 JSON (LightRAG working_dir)
  ├─ full_docs/{id}           : 문서 전체 원문 (PDF 1건 전체, 수백KB~수MB)
  └─ llm_response_cache/{hash}: LLM 응답 캐시

MariaDB  → 변경 없음 (LightRAG와 무관)
Redis    → 기존 캐시 용도만 유지
```

### 9-2. 청크 텍스트 저장 위치 결정

LightRAG KV Store가 담당하는 항목과 대체 가능 여부:

| KV 항목              | 내용           |    Qdrant payload 대체     | 결정                |
| -------------------- | -------------- | :------------------------: | ------------------- |
| `text_chunks`        | 청크 텍스트    |   **가능** (기존도 동일)   | Qdrant payload 포함 |
| `full_docs`          | 문서 전체 원문 | 불가 (벡터와 무관, 대용량) | 파일 JSON           |
| `llm_response_cache` | LLM 응답 캐시  |      불가 (벡터 없음)      | 파일 JSON           |

→ `lightrag_chunks_vdb` payload에 청크 텍스트를 포함시킨다.
→ Redis는 LightRAG 용도로 투입하지 않는다.

### 9-3. LightRAG 전용 LLM/임베딩 설정

| 항목            | 기존 RAG                       | LightRAG 권장                  | 이유                                             |
| --------------- | ------------------------------ | ------------------------------ | ------------------------------------------------ |
| 엔티티 추출 LLM | -                              | `gpt-4.1-mini`                 | 비용 절감 (청크당 2~4회 호출)                    |
| 답변 생성 LLM   | `gpt-4.1`                      | `gpt-4.1`                      | 품질 유지                                        |
| 임베딩          | `text-embedding-3-large` 2048d | `text-embedding-3-small` 1536d | LightRAG 전용 컬렉션, 비용 약 5배 절감(논의필요) |

LightRAG 전용 Qdrant 컬렉션은 기존 컬렉션과 모델이 달라도 무관하다.

### 9-4. 초기화 코드 구조

```python
from lightrag import LightRAG, QueryParam

rag = LightRAG(
    working_dir="./lightrag_storage",  # full_docs + llm_cache 저장 위치

    # LLM: 엔티티 추출은 mini로 비용 절감
    llm_model_func=azure_openai_mini_func,   # gpt-4.1-mini
    embedding_func=azure_openai_embedding,   # text-embedding-3-small

    # 저장소
    vector_storage="QdrantVectorDBStorage",  # 기존 Qdrant 재사용
    kv_storage="JsonKVStorage",              # 파일 기반 (full_docs + cache)
    graph_storage="NetworkXStorage",         # 또는 "Neo4JStorage"

    # Qdrant 연결 (기존 서버 그대로)
    addon_params={
        "qdrant_url": config["qdrant"]["host"],
        "qdrant_port": config["qdrant"]["port"],
        "qdrant_collection_prefix": "lightrag",  # 기존 컬렉션과 분리
    },
)
```

### 9-5. LightRAG 적용 대상 문서

| 컬렉션                      | LightRAG 적합성 | 이유                                             |
| --------------------------- | :-------------: | ------------------------------------------------ |
| **ce_paper** (논문)         |    **핵심**     | 약물-적응증-근거 관계 추출에 최적                |
| **ce_cp** (컴플라이언스)    |    **핵심**     | 규정-행위-금지 관계망 구성                       |
| **high_performer** (암묵지) |    **권장**     | 전략-고객-제품 관계 추출                         |
| ce_email (이메일)           |      선택       | 고객-제품 관계, 대용량 주의                      |
| ce_qa (Q&A)                 |      선택       | 이미 구조화, 효과 제한적                         |
| **ce_call (콜기록)**        |   **조건부**    | 암호화 데이터 → 복호화 후 별도 처리 필요, 대용량 |

---

## 10. 그래프 저장소 비교: NetworkX vs Neo4j

### 10-0. 그래프 저장소 도입 여부

적재하는 문서의 개수(특히 청킹 이후 얼마나 많은 문서가 나올 지 모르기에)를 고려했을 때 그래프 저장소를 도입하는 것이 맞는지에 대한 논의가 필요하다.

문서와 관계(노드, 엣지)가 작을 경우 서버 메모리만 여유가 있다면 NetworkX를 사용하는 것이 좋지만, 문서의 양이 많아질 경우 Neo4j를 도입하는 것이 좋다. Neo4j Community의 경우 무료 + DB 서버 할당만 받으면 됨(Qdrant 적재와 난이도는 비슷). 자세한 비교는 10-6에 작성

### 10-1. 단건 연산 성능 (실측, 567 노드)

| 연산              |           NetworkX |  PostgreSQL 참고\* | 비고                      |
| ----------------- | -----------------: | -----------------: | ------------------------- |
| 엔티티 조회       |           0.000 ms |           0.766 ms | 인메모리 해시 vs TCP 왕복 |
| 1홉 탐색          |           0.001 ms |           0.800 ms | adjacency list vs JOIN    |
| 2홉 탐색          |           0.005 ms |           1.252 ms | BFS vs 재귀 CTE           |
| 전체 파이프라인   |           0.002 ms |           2.341 ms | -                         |
| **LLM 답변 생성** | **4,000~8,000 ms** | **4,000~8,000 ms** | **← 실제 병목**           |

\* Neo4j는 네이티브 그래프 엔진으로 PostgreSQL+AGE보다 빠름. 실측 시 0.3~1ms 예상.

> 그래프 저장소 선택이 사용자 체감 속도에 미치는 영향: **사실상 없음**
> 전체 응답 시간의 80% 이상이 LLM 호출이다.

### 10-2. 동시 사용자 부하 (실측)

| 항목           |     NetworkX | 디스크 기반 DB |
| -------------- | -----------: | -------------: |
| 처리량 (20명)  | 20,881 req/s |     81.4 req/s |
| 처리량 (200명) | 41,236 req/s |     80.1 req/s |
| 평균 지연      |     0.003 ms |         4.8 ms |
| P99 지연       |     0.020 ms |         8.7 ms |

> 현재 프로젝트: `workers=1` 단일 프로세스.
> 수십 명 이하 동시 사용자라면 두 방식 모두 LLM 병목 앞에서 차이 없음.

### 10-3. 대규모 스케일 (실측, 30,000 노드)

| 항목             |     NetworkX | 디스크 기반 DB |
| ---------------- | -----------: | -------------: |
| 서버 재시작 로딩 | **1,702 ms** |           0 ms |
| 런타임 RAM       |  **+204 MB** |   DB 서버 관리 |
| 쿼리 평균 지연   |     0.003 ms |         4.4 ms |

### 10-4. 현재 프로젝트 예상 노드 수

실측 기준: LightRAG 청크당 약 11개 엔티티 추출 (52청크 → 567노드)
현재 프로젝트 청킹: 1000자 / 200자 overlap

| 적용 대상               | 예상 문서 수 | 청크 수 추정 | 예상 노드 수 | 누적         |
| ----------------------- | ------------ | ------------ | ------------ | ------------ |
| 논문 (ce_paper)         | 50~300건     | 500~3,000    | 5,500~33,000 | 5,500~33,000 |
| 컴플라이언스 (ce_cp)    | 10~50건      | 100~500      | 1,100~5,500  | 6,600~38,500 |
| 암묵지 (high_performer) | 50~200건     | 300~1,200    | 3,300~13,200 | 9,900~51,700 |
| 이메일 (선택)           | 수백~수천건  | 수천         | 수만         | —            |
| 콜기록 (조건부)         | 수천~수만건  | 수만         | 수십만       | —            |

**초기 도입 현실적 범위 (논문+컴플라이언스+암묵지): 약 10,000~52,000 노드**

### 10-5. NetworkX 규모별 비용

| 노드 수                   | GraphML 로딩 | RAM 점유   | 판단            |
| ------------------------- | ------------ | ---------- | --------------- |
| ~567 (테스트)             | 25 ms        | ~7 MB      | NetworkX 적합   |
| 10,000~30,000 (초기 도입) | 500~1,700 ms | ~70~200 MB | **주의 구간**   |
| 30,000 초과               | 1,700 ms+    | 200 MB+    | Neo4j 전환 권장 |
| 100,000+ (이메일 포함)    | ~10초+       | ~500 MB+   | Neo4j 필수      |
| 수십만+ (콜기록 포함)     | 수분         | GB 단위    | Neo4j 필수      |

### 10-6. 운영 특성 비교

| 항목                 | NetworkX                   | Neo4j Community            |
| -------------------- | -------------------------- | -------------------------- |
| 추가 인프라          | 없음                       | Docker 컨테이너 1개        |
| 재시작 복구          | GraphML 파일 재로딩 (지연) | 즉시 (디스크 기반)         |
| 런타임 메모리        | 그래프 전체 상주           | 쿼리 시에만 사용           |
| 멀티 프로세스        | N배 메모리 복사            | DB 공유 (1배)              |
| 동시 쓰기            | Lock 직접 구현 필요        | 트랜잭션 기본 제공         |
| 데이터 영속성        | GraphML 파일 (손상 위험)   | 트랜잭션 로그 보장         |
| LightRAG 지원        | 공식 기본 내장             | 커뮤니티 기여 (안정적)     |
| 무료 범위            | 완전 무료                  | Community 영구 무료        |
| 데이터 크기 제한     | 없음                       | 없음                       |
| 클러스터링           | 해당 없음                  | Enterprise만 (현재 불필요) |
| 현재 프로젝트 적합성 | workers=1 → 문제 없음      | 단일 인스턴스 → 완전 부합  |

---

## 11. 의사결정 기준

### 선택 매트릭스

| 결정 기준       | 가중치 | NetworkX | Neo4j |
| --------------- | ------ | :------: | :---: |
| 초기 도입 속도  | 높음   |  ★★★★★   | ★★★☆☆ |
| 인프라 단순성   | 높음   |  ★★★★★   | ★★★☆☆ |
| LightRAG 안정성 | 높음   |  ★★★★★   | ★★★★☆ |
| 대용량 확장성   | 중간   |  ★★☆☆☆   | ★★★★★ |
| 메모리 효율     | 중간   |  ★★☆☆☆   | ★★★★★ |
| 재시작 안정성   | 중간   |  ★★★☆☆   | ★★★★★ |
| 향후 워커 확장  | 중간   |  ★★☆☆☆   | ★★★★★ |

### Neo4j 전환 트리거 (하나라도 해당 시 즉시 전환)

```
노드 수 > 30,000개          → 재시작 로딩 1.7초+ 발생
이메일/콜기록 LightRAG 적재  → 수십만 노드 예상
uvicorn workers > 1        → 인메모리 그래프 N배 복사
동시 사용자 > 50명          → 커넥션 풀 기반 DB 필요
무중단 배포 요구사항 발생    → 파일 기반 불안정
```

---

## 12. 권장 도입 단계

### Phase 1: NetworkX로 시작 (논문 + 컴플라이언스 + 암묵지)

```
예상 노드: 10,000~52,000개
예상 RAM: 70~350 MB (허용 범위)
재시작 로딩: 500ms~1,700ms

Qdrant   → 기존 컬렉션 재적재 + lightrag_{entities,relationships,chunks}_vdb 추가
NetworkX → GraphML 파일 영속 (working_dir 내 자동 저장)
파일 JSON → full_docs + llm_response_cache
```

### Phase 2: Neo4j 전환 (트리거 충족 시)

```
compose.yml에 neo4j 서비스 추가
LightRAG graph_storage="Neo4JStorage" 로 변경
GraphML → Neo4j 마이그레이션 스크립트 실행 (1회)
```

---

## 13. 속도 개선 우선순위

LightRAG 쿼리 응답 시간 구조:

```
쿼리 1회 ≈ 5~10초
  ├─ 쿼리 임베딩        : ~0.5초
  ├─ Qdrant 벡터 검색   : ~0.1초
  ├─ 그래프 탐색        : ~0~2ms  (LLM 대비 무의미)
  ├─ 검색 보조 LLM      : ~1~2초
  └─ 답변 생성 LLM      : ~4~8초  ← 전체의 80%
```

| 우선순위 | 방법                                    | 예상 효과               | 난이도 |
| -------- | --------------------------------------- | ----------------------- | ------ |
| 1        | **시맨틱 캐싱** — 유사 질문 응답 재사용 | 반복 쿼리 수초 → 수백ms | 중     |
| 2        | **top_k 축소** — 기본 60 → 20           | 컨텍스트 토큰 50%↓      | 하     |
| 3        | **max_token 상한** — 섹션별 제한        | LLM 입력 토큰 절감      | 하     |
| 4        | **프롬프트 압축(LLMLingua)**            | 추론 속도 ~18% 향상     | 상     |
| 5        | **그래프 워크 압축** — 저연결 노드 제거 | 컨텍스트 ~60% 절감      | 상     |

### 즉시 적용 가능한 QueryParam 튜닝

```python
QueryParam(
    mode="hybrid",
    top_k=20,                           # 기본 60 → 비용/속도 최적화
    max_token_for_text_unit=1000,       # 현재 청크 크기(1000자)와 정렬
    max_token_for_global_context=1500,
    max_token_for_local_context=1500,
)
```

---

## 14. 콜기록 LightRAG 적용 시 추가 고려사항

현재 `ce_call_encrypt` 컬렉션은 Fernet+HKDF 암호화가 적용되어 있다.
LightRAG 엔티티 추출 LLM에 암호화된 텍스트를 그대로 넣으면 의미 있는 추출이 불가능하다.

```
필요한 처리 순서:
1. MariaDB/Redshift에서 원본 콜기록 조회
2. 복호화 (decrypt_text_with_salt 적용)
3. LightRAG에 복호화된 텍스트 삽입
4. LightRAG 내부에서 엔티티 추출 → Neo4j/NetworkX 저장
5. LightRAG 벡터 (non-encrypted) → lightrag_chunks_vdb 저장

주의: 콜기록은 대용량 (수천~수만 건) → 적재 비용 높음
     초기 도입 범위에서 제외하고 별도 검토 권장
```
