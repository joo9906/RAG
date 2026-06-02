# 금융 멀티 에이전트 챗봇

LangGraph 기반 멀티 에이전트 시스템 — RAG 에이전트와 주식 에이전트를 Supervisor가 라우팅합니다.

## 아키텍처

```
사용자 입력
     │
     ▼
┌─────────────────────────────────────────┐
│            Supervisor (Claude)           │
│   "rag" vs "stock" 라우팅 결정           │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
┌──────────────┐  ┌────────────────────┐
│  RAG 에이전트 │  │  주식 에이전트      │
│              │  │                    │
│  Query       │  │  MCP Client        │
│    ├─Weaviate│  │    └─ yfinance     │
│    └─ES BM25 │  │  (get_stock_price  │
│              │  │   get_stock_info   │
│  결과 비교   │  │   get_stock_history│
│  Claude 답변 │  │   compare_stocks)  │
└──────────────┘  └────────────────────┘
```

### 검색 방식 비교 (RAG 에이전트)

| 항목 | Weaviate | Elasticsearch |
|------|----------|---------------|
| 검색 방식 | Dense Vector (코사인 유사도) | BM25 키워드 빈도 |
| 임베딩 | OpenAI text-embedding-3-small | 불필요 |
| 장점 | 의미론적 유사성 포착 | 정확한 키워드 매칭 |
| 단점 | 임베딩 비용 발생 | 동의어·문맥 미인식 |

---

## 빠른 시작

### 1. 환경 설정

```bash
cp .env.example .env
# .env 파일에 API 키 입력
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. Docker 서비스 실행

```bash
docker compose up -d
# Weaviate: http://localhost:8080
# Elasticsearch: http://localhost:9200
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
→ RAG 에이전트 (Weaviate + ES 검색 결과 비교 + Claude 답변)

🧑 질문: 애플 현재 주가 알려줘
→ 주식 에이전트 (MCP → yfinance → 실시간 조회)

🧑 질문: 삼성전자와 SK하이닉스 주가 비교해줘
→ 주식 에이전트 (compare_stocks 도구 사용)

🧑 질문: ETF 투자 전략 설명해줘
→ RAG 에이전트 (두 검색 방식의 결과 차이 표시)
```

---

## 파일 구조

```
financial_chatbot/
├── docker-compose.yml          # Weaviate + Elasticsearch
├── requirements.txt
├── .env.example
├── docs/
│   ├── 01_financial_basics.md      # 금융 기초 개념
│   └── 02_investment_strategies.md # 투자 전략
├── vector_store/
│   ├── weaviate_store.py       # Weaviate CRUD + 벡터 검색
│   └── elasticsearch_store.py  # ES BM25 + kNN 검색
├── mcp_server/
│   └── stock_mcp_server.py     # FastMCP 기반 yfinance 서버
├── agents/
│   ├── rag_agent.py            # 이중 검색 + 비교 + Claude 답변
│   ├── stock_agent.py          # MCP 클라이언트 + ReAct 에이전트
│   └── supervisor.py           # LangGraph StateGraph 라우터
├── ingest.py                   # 문서 → 임베딩 → 인덱싱
└── main.py                     # 챗봇 CLI
```

---

## 환경 변수

| 변수 | 설명 |
|------|------|
| `ANTHROPIC_API_KEY` | Claude API 키 (필수) |
| `OPENAI_API_KEY` | OpenAI 임베딩 키 (필수) |
| `WEAVIATE_URL` | Weaviate URL (기본: http://localhost:8080) |
| `ELASTICSEARCH_URL` | ES URL (기본: http://localhost:9200) |
| `LANGCHAIN_API_KEY` | LangSmith 트레이싱 (선택) |
