# LightRAG 정리

이 폴더는 `HKUDS/LightRAG` 공식 저장소와 문서를 기준으로 LightRAG를 기초 개념부터 실제 활용까지 학습할 수 있도록 정리한 문서 모음이다.

> 확인일: 2026-05-13  
> 기준 자료: `HKUDS/LightRAG` main 브랜치 README, API Server 문서, Core 프로그래밍 문서, Advanced Features 문서

## 문서 구성

| 파일 | 내용 |
|---|---|
| [01_기초개념.md](./01_기초개념.md) | RAG와 LightRAG의 차이, 핵심 구조, 인덱싱/질의 흐름 |
| [02_설치와_빠른시작.md](./02_설치와_빠른시작.md) | PyPI, 소스, Docker 설치 및 첫 실행 |
| [03_Core_API_활용.md](./03_Core_API_활용.md) | Python 코드에서 LightRAG Core를 직접 쓰는 방법 |
| [04_Server_WebUI_API.md](./04_Server_WebUI_API.md) | LightRAG Server, WebUI, REST API, Ollama 호환 인터페이스 |
| [05_저장소와_운영.md](./05_저장소와_운영.md) | KV/Vector/Graph/DocStatus 저장소, 워크스페이스, 운영 설정 |
| [06_고급기능과_실전팁.md](./06_고급기능과_실전팁.md) | Reranker, 멀티모달, 평가, 관측성, 캐시, 트러블슈팅 |

## 한 줄 요약

LightRAG는 문서를 단순히 벡터 검색만 하는 RAG가 아니라, 문서에서 엔티티와 관계를 추출해 지식 그래프를 만들고, 벡터 검색과 그래프 검색을 함께 사용해 답변 품질을 높이는 RAG 프레임워크다.

## 언제 쓰면 좋은가

- 문서 안의 개념, 인물, 조직, 사건, 관계를 함께 추론해야 할 때
- "A와 B의 관계", "전체 흐름", "주요 주제", "문서 간 연결" 같은 질문이 중요할 때
- 단순 chunk 검색보다 지식 그래프 기반 탐색이 필요한 도메인
- WebUI/API 서버로 지식베이스를 운영하고 싶을 때
- Neo4j, PostgreSQL, Milvus, Qdrant, Redis, MongoDB, OpenSearch 같은 외부 저장소와 붙여 확장하고 싶을 때

## 먼저 알아둘 핵심

- LightRAG는 LLM, embedding model, storage backend가 모두 필요하다.
- 문서 인덱싱 단계에서 LLM이 엔티티와 관계를 추출하므로 일반 RAG보다 LLM 요구사항이 높다.
- embedding 모델은 인덱싱 후 바꾸면 기존 벡터 데이터와 차원이 맞지 않을 수 있어 재인덱싱이 필요하다.
- 공식 문서는 프로젝트 통합 시 Core 직접 사용보다 LightRAG Server의 REST API 사용을 권장한다.
- Reranker를 쓰면 retrieval 품질이 좋아지며, 공식 문서는 reranker 사용 시 `mix` 모드를 권장한다.

## 공식 출처

- 공식 저장소: https://github.com/HKUDS/LightRAG
- 논문: https://arxiv.org/abs/2410.05779
- API Server 문서: https://github.com/HKUDS/LightRAG/blob/main/docs/LightRAG-API-Server.md
- Core 프로그래밍 문서: https://github.com/HKUDS/LightRAG/blob/main/docs/ProgramingWithCore.md
- 고급 기능 문서: https://github.com/HKUDS/LightRAG/blob/main/docs/AdvancedFeatures.md
