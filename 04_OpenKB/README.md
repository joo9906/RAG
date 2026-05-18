# OpenKB 실습 가이드

> 블로그 포스팅: [OpenKB란?](https://joo9906.tistory.com/115)

---

## 이 폴더가 무엇인가요?

LightRAG 학습 문서를 **OpenKB**로 인덱싱하고 질의하는 실습 예제입니다.

`base_study/` 폴더에 정리해둔 LightRAG 학습 문서(01~06번 마크다운)를 OpenKB에 올려서 지식베이스를 구성하였습니다.
이 지식베이스에 자연어로 질문을 던지면, 문서 내용을 바탕으로 답변을 받을 수 있습니다.

---

## 폴더 구조

```
04_OpenKB/
├── .env                    # OpenAI API 키
├── .openkb/
│   ├── config.yaml         # OpenKB 설정 (모델, 언어 등)
│   └── hashes.json         # 인덱싱된 문서 해시 추적
├── docs/             # 지식베이스에 올린 원본 학습 문서
│   ├── README.md
│   ├── 01_기초개념.md
│   ├── 02_설치와_빠른시작.md
│   ├── 03_Core_API_활용.md
│   ├── 04_Server_WebUI_API.md
│   ├── 05_저장소와_운영.md
│   └── ....
├── raw/                    # docs에 저장된 것과 동일한 문서 (OpenKB 소스 경로용)
├── wiki/                   # OpenKB가 자동으로 생성하는 지식 구조
│   ├── index.md            # 전체 문서 카탈로그
│   ├── log.md              # 인덱싱/질의 이력
│   ├── sources/            # 원본 문서 내용
│   ├── summaries/          # 문서별 요약 페이지
│   └── concepts/           # 크로스 문서 개념 합성 페이지
├── requirements.txt        # 의존성 목록
└── test_openkb.py          # 간단한 질의 테스트 스크립트
```

---

## 시작 전에 준비할 것들

### 1. 패키지 설치

```bash
pip install openkb
```

전체 의존성을 한 번에 설치하려면 아래 명령을 사용하십시오.

```bash
pip install -r requirements.txt
```

> 핵심 패키지: `openkb==0.1.3`, `pageindex==0.3.0.dev1`

### 2. OpenAI API 키 설정

`.env` 파일에 키를 입력하십시오.

```env
OPENAI_API_KEY=sk-...
```

환경변수로 직접 설정하는 방법도 있습니다.

```bash
export OPENAI_API_KEY=sk-...
```

---

## OpenKB 설정 파일

`.openkb/config.yaml`에서 모델과 동작 방식을 조정할 수 있습니다.

```yaml
language: en # 응답 언어 (en / ko 등)
model: gpt-4o-mini # 사용할 OpenAI 모델
pageindex_threshold: 20 # 페이지 인덱스 생성 기준 (문서 길이)
```

---

## 사용 방법

### Python 코드로 질의하기

```python
from openkb import KnowledgeBase

# .openkb 디렉터리 경로로 지식베이스 초기화
kb = KnowledgeBase("./.openkb")

# 질문하고 답변 받기
answer = kb.aquery("라이트라그 구현하려면 뭐부터 해야해?")
print(answer.output)

# 검색된 문서 청크만 따로 확인하기
retrieve = kb.retrieve("라이트라그 구현하려면 뭐부터 해야해?")
print(retrieve.retrieved_docs)
```

### 주요 메서드

| 메서드                  | 설명                                                                        |
| ----------------------- | --------------------------------------------------------------------------- |
| `KnowledgeBase(path)`   | 지정 경로의 `.openkb` 설정으로 지식베이스를 초기화합니다                    |
| `kb.aquery(question)`   | 질문에 대한 LLM 답변을 생성합니다.`.output`으로 텍스트를 확인할 수 있습니다 |
| `kb.retrieve(question)` | 관련 문서 청크만 검색합니다.`.retrieved_docs`로 목록을 확인할 수 있습니다   |

---

## 새 문서를 추가하는 방법

1. `base_study/` (또는 `raw/`) 폴더에 마크다운 파일을 추가합니다.
2. 아래 CLI 명령으로 인덱싱을 실행합니다.

```bash
openkb ingest ./base_study --config ./.openkb/config.yaml
```

인덱싱이 완료되면 `.openkb/hashes.json`에 해시가 추가되고, `wiki/` 폴더에 요약 및 개념 페이지가 자동으로 생성됩니다.

---

## wiki 폴더 구조

OpenKB가 문서를 인덱싱하면 `wiki/` 폴더 아래에 다음과 같은 구조가 자동으로 만들어집니다.

| 경로              | 내용                                       |
| ----------------- | ------------------------------------------ |
| `wiki/index.md`   | 전체 문서·개념·탐색 목록 카탈로그          |
| `wiki/log.md`     | 인덱싱·질의 이력 (타임스탬프 포함)         |
| `wiki/sources/`   | 원본 문서 내용 (직접 수정하지 마십시오)    |
| `wiki/summaries/` | 문서별 핵심 요약                           |
| `wiki/concepts/`  | 여러 문서에 걸친 주제를 합성한 개념 페이지 |

---

## 테스트 실행

```bash
python test_openkb.py
```

`"라이트라그 구현하려면 뭐부터 해야해?"`라는 질문으로 답변과 검색 문서를 바로 확인할 수 있습니다.

---

## 현재 인덱싱된 문서 목록

`base_study/` 폴더의 LightRAG 학습 문서 7개가 인덱싱되어 있습니다.

| 파일                      | 내용                                       |
| ------------------------- | ------------------------------------------ |
| `01_기초개념.md`          | RAG 개념, LightRAG 구조, 질의 모드         |
| `02_설치와_빠른시작.md`   | 설치 방법, 첫 실행                         |
| `03_Core_API_활용.md`     | Python에서 LightRAG Core 직접 사용         |
| `04_Server_WebUI_API.md`  | REST API, WebUI, Ollama 호환 인터페이스    |
| `05_저장소와_운영.md`     | 저장소 백엔드, 워크스페이스, 운영 설정     |
| `06_고급기능과_실전팁.md` | Reranker, 멀티모달, 평가, 캐시, 트러블슈팅 |
| `README.md`               | LightRAG 학습 문서 전체 개요               |
