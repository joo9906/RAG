# 금융 멀티 에이전트 RAG 챗봇

LangGraph 기반 고도화 RAG 파이프라인 — Redis 시맨틱 캐시, 멀티쿼리 확장, RRF 리랭킹, Self-correction, 할루시네이션 검증을 포함한 멀티 에이전트 금융 챗봇

---

## 전체 파이프라인 흐름

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  현재 구현 (✅)  /  개선 예정 (🔧)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

사용자 쿼리 입력
        │
        ▼
┌───────────────────────────────────┐
│  ✅ Redis 시맨틱 캐시              │
│                                   │
│  쿼리 임베딩 → 코사인 유사도 계산  │
│  유사도 ≥ 0.8 → 즉시 캐시 반환   │
│  TTL: 24h                         │
└───────────────┬───────────────────┘
                │ 캐시 미스
                ▼
┌───────────────────────────────────┐
│  ✅ 동적 라우터                    │
│                                   │
│  GPT-4o가 질문 도메인 분류         │
│  "rag" (금융 지식)                 │
│  "stock" (실시간 주가)             │
│  🔧 "hybrid" (복합 질문) — 미구현  │
└──────────┬───────────────┬─────────┘
           │               │
         "rag"           "stock"
           │               │
           │               ▼
           │    ┌──────────────────────┐
           │    │  ✅ 주식 에이전트     │
           │    │                      │
           │    │  MCP Client          │
           │    │  └─ yfinance         │
           │    │     ├ get_stock_price │
           │    │     ├ get_stock_info  │
           │    │     ├ get_stock_hist  │
           │    │     └ compare_stocks  │
           │    └──────────┬───────────┘
           │               │
           │          최종 답변
           │
           ▼
┌───────────────────────────────────┐
│  ✅ 멀티쿼리 확장                  │
│                                   │
│  원본 쿼리 + GPT-4o 생성 변형 ×3  │
│  동의어 / 상위 개념 / 관련 용어    │
│  총 4개 쿼리로 검색 다양성 확보    │
└───────────────┬───────────────────┘
                │  4개 쿼리
                ▼
┌───────────────────────────────────┐
│  ✅ 병렬 하이브리드 검색            │
│                                   │
│  쿼리별 asyncio.gather 병렬 실행   │
│                                   │
│  ┌─────────────┐ ┌─────────────┐  │
│  │  Weaviate   │ │Elasticsearch│  │
│  │ Dense Vector│ │    BM25     │  │
│  │코사인 유사도 │ │ 키워드 빈도  │  │
│  └─────────────┘ └─────────────┘  │
│                                   │
│  4쿼리 × 2스토어 = 최대 8개 결과  │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  ✅ RRF 리랭킹                     │
│                                   │
│  Reciprocal Rank Fusion           │
│  score = Σ 1 / (rank + 60)        │
│  여러 리스트 통합 → top-5 선별     │
│                                   │
│  🔧 Cross-Encoder 2단계 리랭킹    │
│     RRF top-20 → Cross-Encoder   │
│     → 최종 top-5  (미구현)        │
└───────────────┬───────────────────┘
                │  top-5 문서
                ▼
┌───────────────────────────────────┐
│  ✅ 답변 생성                      │
│                                   │
│  RRF 문서 기반 GPT-4o 생성         │
│  문서에 없는 내용 추측 금지 지시    │
│  출처 명시 요구                    │
└───────────────┬───────────────────┘
                │
                ▼
┌───────────────────────────────────┐
│  ✅ 할루시네이션 검증               │
│                                   │
│  GPT-4o가 답변 vs 소스 문서 비교   │
│  verdict: "grounded"              │
│         / "hallucinated"          │
└──────────┬────────────────┬────────┘
           │ hallucinated   │ grounded
           │ & 시도 < 2회   │
           ▼                ▼
┌────────────────┐  ┌───────────────────┐
│ ✅ Self-       │  │ ✅ 캐시 저장 + 출력│
│    correction  │  │                   │
│                │  │ Redis에 저장       │
│ 할루시네이션   │  │ (다음 동일 질문    │
│ 원인 분석 →   │  │  즉시 반환)        │
│ 개선 쿼리 생성 │  └───────────────────┘
└────────┬───────┘
         │ 최대 2회 루프
         └──────────────────► 멀티쿼리 확장으로 복귀


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  🔧 개선 예정 — Hybrid 라우팅 (복합 질문 처리)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"ETF가 뭔지 설명하고 삼성전자 주가도 알려줘" 같은 복합 질문

현재: 라우터가 "rag" 또는 "stock" 하나만 선택 → 나머지 무시

목표:
        "hybrid" 라우팅
               │
     ┌─────────┴──────────┐
     │ LangGraph Send API  │ (병렬 fan-out)
     └──────┬──────────────┘
            │
     ┌──────┴──────┐
     ▼             ▼
 RAG 파이프라인  주식 에이전트
     │             │
     └──────┬──────┘
            ▼
       Merge Node
       두 답변 합성
            │
            ▼
       최종 답변
```

---

## 기술 스택

| 레이어 | 기술 |
|--------|------|
| LLM | GPT-4o (라우팅 · 답변 · 검증) |
| 임베딩 | OpenAI text-embedding-3-small |
| 오케스트레이션 | LangGraph StateGraph |
| 시맨틱 검색 | Weaviate 1.28 (Dense Vector) |
| 키워드 검색 | Elasticsearch 8.11 (BM25) |
| 캐시 | Redis 7 (임베딩 기반 시맨틱 캐시) |
| 실시간 주가 | MCP + yfinance |
| 트레이싱 | LangSmith |

---

## 빠른 시작

### 1. 환경 설정

```bash
# .env 파일에 아래 키 입력
OPENAI_API_KEY=sk-...
LANGSMITH_TRACING=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=my-project
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. Docker 서비스 실행

```bash
docker compose up -d
# Weaviate:       http://localhost:8080
# Elasticsearch:  http://localhost:9200
# Redis:          localhost:6379
```

### 4. 문서 인덱싱

```bash
python ingest.py
# --reset 플래그로 기존 데이터 초기화 가능
```

### 5. 챗봇 실행

```bash
python main.py
```

---

## 예시 대화

```
🧑 질문: PER이란 무엇인가요?
→ RAG 에이전트 | 멀티쿼리 4개 | 검색 문서 5개 | 검증: ✅grounded

🧑 질문: PER이란 무엇인가요?  (동일 질문 재입력)
→ ⚡ 캐시 히트 (유사도: 1.000) — 즉시 반환

🧑 질문: 애플 현재 주가 알려줘
→ 주식 에이전트 | MCP → yfinance 실시간 조회

🧑 질문: ETF 투자 전략 알려줘 (할루시네이션 감지 시)
→ RAG 에이전트 | Self-correction 1회 → 재검색 → 검증: ✅grounded
```

---

## 파일 구조

```
05_Langgraph/
├── main.py                     # 챗봇 CLI 진입점
├── ingest.py                   # 문서 → 임베딩 → 인덱싱
├── docker-compose.yml          # Weaviate + ES + Redis
├── requirements.txt
│
├── cache/
│   └── redis_cache.py          # 시맨틱 캐시 (임베딩 코사인 유사도)
│
├── graph/
│   ├── state.py                # AdvancedRAGState TypedDict
│   ├── nodes.py                # 전체 노드 함수 구현
│   └── builder.py              # StateGraph 조립 + 조건 엣지
│
├── agents/
│   ├── supervisor.py           # 구버전 단순 라우터 (legacy)
│   ├── rag_agent.py            # 구버전 RAG 에이전트 (legacy)
│   └── stock_agent.py          # MCP 주식 에이전트
│
├── vector_store/
│   ├── weaviate_store.py       # Dense Vector 검색
│   └── elasticsearch_store.py  # BM25 키워드 검색
│
├── mcp_server/
│   └── stock_mcp_server.py     # FastMCP + yfinance 4개 도구
│
├── docs/
│   ├── 01_financial_basics.md      # 금융 기초 개념
│   └── 02_investment_strategies.md # 투자 전략
│
└── optimization_docs/          # 고도화 참고 자료
    ├── AKB.md
    ├── OpenKB.md
    ├── OpenKB_Official.md
    └── monimo.md               # 모니모급 고도화 전략
```

---

## 환경 변수

| 변수 | 설명 | 필수 |
|------|------|------|
| `OPENAI_API_KEY` | GPT-4o + 임베딩 | ✅ |
| `WEAVIATE_HOST` | Weaviate 호스트 (기본: localhost) | |
| `ELASTICSEARCH_URL` | ES URL (기본: http://localhost:9200) | |
| `REDIS_URL` | Redis URL (기본: redis://localhost:6379) | |
| `LANGSMITH_TRACING` | LangSmith 추적 활성화 | |
| `LANGCHAIN_API_KEY` | LangSmith API 키 | |
| `LANGSMITH_PROJECT` | LangSmith 프로젝트명 | |
