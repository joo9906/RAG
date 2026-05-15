# 04. Server, WebUI, API

LightRAG Server는 WebUI와 REST API를 제공한다. 문서 업로드, 인덱싱, 지식 그래프 시각화, 질의, Ollama 호환 chat interface를 지원한다.

공식 문서는 프로젝트 통합 시 Core 직접 사용보다 Server의 REST API 사용을 권장한다.

## 서버 시작

```bash
lightrag-server
```

기본값:

| 항목 | 기본값 |
|---|---|
| host | `0.0.0.0` |
| port | `9621` |
| working dir | `./rag_storage` |
| input dir | `./inputs` |
| log level | `INFO` |

문서 확인:

- Swagger UI: `http://localhost:9621/docs`
- ReDoc: `http://localhost:9621/redoc`

## 자주 쓰는 실행 옵션

```bash
lightrag-server --host 0.0.0.0 --port 9621 --working-dir ./rag_storage --input-dir ./inputs
```

| 옵션 | 설명 |
|---|---|
| `--host` | 서버 listen 주소 |
| `--port` | 서버 포트 |
| `--timeout` | LLM 요청 timeout |
| `--log-level` | 로그 레벨 |
| `--working-dir` | RAG 저장 디렉터리 |
| `--input-dir` | 입력 문서 디렉터리 |
| `--workspace` | 데이터 격리용 workspace 이름 |

## WebUI에서 할 수 있는 일

- 문서 업로드
- 문서 인덱싱 상태 확인
- 지식 그래프 탐색과 시각화
- RAG 질의 실행
- query mode 선택
- 검색된 context와 참조 확인

## REST API 기본 흐름

1. 서버 실행
2. 문서 업로드 또는 텍스트 삽입
3. 인덱싱 상태 확인
4. 질의 실행
5. 필요하면 context/reference 포함해 디버깅

비동기 문서 인덱싱을 지원하는 endpoint는 track id를 반환한다.

| endpoint | 역할 |
|---|---|
| `/documents/upload` | 파일 업로드 |
| `/documents/text` | 단일 텍스트 삽입 |
| `/documents/texts` | 여러 텍스트 삽입 |
| `/track_status/{track_id}` | 처리 상태 확인 |
| `/query` | 일반 질의 |
| `/query/stream` | streaming 질의 |

## 질의 예시

```json
{
  "query": "What is LightRAG?",
  "mode": "mix",
  "include_references": true,
  "include_chunk_content": true
}
```

`include_chunk_content=true`는 `include_references=true`일 때 의미가 있다. chunk content는 문자열 배열로 반환될 수 있다.

## Ollama 호환 인터페이스

LightRAG Server는 Ollama 호환 API를 제공해 Open WebUI 같은 프론트엔드에서 `lightrag:latest` 모델처럼 사용할 수 있다.

채팅 메시지 앞에 prefix를 붙여 query mode를 고를 수 있다.

```text
/local 질문
/global 질문
/hybrid 질문
/naive 질문
/mix 질문
/context 질문
/mixcontext 질문
/bypass 질문
```

예:

```text
/mix What are the main relationships in this document?
```

prefix가 없으면 기본적으로 `hybrid` mode가 사용된다.

## Chat에서 user prompt 붙이기

검색 질문과 출력 지시를 분리하려면 prefix 뒤 대괄호를 쓴다.

```text
/mix[Use a table and cite sources] Explain the main entities.
```

대괄호 안 내용은 retrieval에는 직접 참여하지 않고, 검색 후 LLM이 답변을 구성하는 방식에 영향을 준다.

## 인증 설정

기본 서버는 인증 없이 접근 가능하다. 외부에 노출하려면 API key와 계정 인증을 함께 설정하는 것이 좋다.

API key:

```env
LIGHTRAG_API_KEY=your-secure-api-key
WHITELIST_PATHS=/health,/api/*
```

요청 헤더:

```bash
curl -X POST "http://localhost:9621/documents/scan" -H "X-API-Key: your-secure-api-key" -d ""
```

JWT 계정 인증:

```env
AUTH_ACCOUNTS='admin:{bcrypt}$2b$12$replace-with-generated-hash'
TOKEN_SECRET='your-secret'
TOKEN_EXPIRE_HOURS=4
```

비밀번호 hash 생성:

```bash
lightrag-hash-password --username admin
```

API key만 설정하고 WebUI 계정 인증을 설정하지 않으면 Guest 경로로 접근 가능할 수 있으므로, 보호가 필요하면 둘 다 설정한다.

## 운영용 프로세스

Linux 운영 환경에서는 Gunicorn + Uvicorn 모드를 쓸 수 있다. Windows에서는 지원되지 않는다.

```bash
lightrag-gunicorn --workers 4
```

문서 추출 도구가 CPU를 많이 쓰는 경우 multiprocess가 query blocking을 줄이는 데 도움이 된다.

## Nginx reverse proxy 주의사항

파일 업로드 endpoint는 기본 Nginx 제한인 1MB에 걸릴 수 있다. `/documents/upload`에는 `client_max_body_size`를 크게 잡는다.

```nginx
location /documents/upload {
    client_max_body_size 100M;
    proxy_pass http://localhost:9621;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
}
```

Streaming endpoint는 gzip 압축을 끄는 것이 좋다.

```nginx
location ~ ^/(query/stream|api/chat|api/generate) {
    gzip off;
    proxy_pass http://localhost:9621;
    proxy_read_timeout 300s;
}
```
