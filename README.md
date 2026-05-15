# RAG (Retrieval-Augmented Generation) 스터디 및 프로젝트

RAG와 관련된 다양한 기술과 파이프라인 고도화를 연구하고 테스트한 내용을 블로그와 함께 정리하여 업로드하는 레포지토리입니다.

## 📂 디렉토리 구조 및 주요 내용

### `01_Weaviate/`
Weaviate 벡터 데이터베이스의 기본적인 세팅과 연동 테스트를 다룹니다.
- `docker-compose.yml`: Weaviate 서버 구동을 위한 Docker 환경
- `weaviate.py`: Weaviate Python 클라이언트 연동 및 테스트 스크립트
- `first_data.json`: 테스트를 위한 초기 샘플 데이터

### `02_history_improve/`
RAG 환경에서 사용자의 대화 맥락(History)을 유지하고, 리랭커(Reranker)를 도입해 검색 품질을 개선하는 파이프라인 고도화를 다룹니다.
- `total_test.py`, `history_test.py`: 대화 기록 반영 및 RAG 전체 흐름 테스트 코드
- `history_analysis.md`, `history, rerank 관련 기능 전체 테스트 결과.md`: 테스트 결과 상세 분석 리포트
- `improve_methods/`: Rerank 모델 활용 및 병렬 처리(Parallel processing) 테스트 코드(`rerank_test.py`, `parallel_test.py` 등)와 개선 리포트가 포함된 하위 모듈

### `03_LightRAG/`
벡터 검색(Vector Search)과 지식 그래프(Knowledge Graph)를 결합한 차세대 RAG 아키텍처인 **LightRAG**의 환경 구축 및 부하 테스트를 다룹니다.
- `total_process.py`, `load_test.py`: 문서 데이터 인제스천 및 쿼리 성능(부하) 테스트 
- `docker-compose.yml`, `Dockerfile`, `LightRAG_config.json`: LightRAG 구성을 위한 도커 환경 및 세팅 파일
- `docs/`, `chunked_docs/`: 벡터 및 그래프 변환에 사용된 원본 문서 및 분할된(Chunk) 문서 디렉토리

### `04_OpenKB/`
문서들을 바탕으로 서로 연결된 위키(Wiki) 형태의 지식 베이스를 자동으로 구성해주는 **OpenKB** 도구 테스트를 다룹니다. (옵시디언 등의 마크다운 뷰어와 연동)
- `test.py`: OpenKB 파이프라인 실행 및 쿼리 테스트 스크립트
- `wiki/`: OpenKB가 생성한 마크다운 기반의 위키 저장소 (요약문, 주제별 개념 추출 문서 등 구조화된 파일 포함)
- `base_study/`, `raw/`: 모델에 입력할 원본 마크다운 학습 문서 저장 공간
- `.openkb/`: DB 정보(해시, 설정 등)가 저장된 로컬 데이터베이스 공간
