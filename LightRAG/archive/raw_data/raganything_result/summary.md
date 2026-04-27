# RAGAnything 처리 요약

생성 시각: 2026-04-23 16:45:51
사용 LLM: GPT-4o-mini (OpenAI fallback)
임베딩 모델: OpenAI text-embedding-3-small (1536차원)
처리 파일 수: 1개

## 파일별 처리 현황

| 파일 | 블록 수 | 문자 수 | 토큰(추정) | 파싱(초) | 삽입(초) | 합계(초) | 오류 |
|------|--------|--------|-----------|---------|---------|---------|------|
| testpdf.pdf |    57 |  65,766 |   43,143 |   10.01 |  638.16 |  649.87 | - |
| **합계** |    57 |  65,766 |   43,143 |   10.01 |  638.16 |  649.87 | |

## LLM / 임베딩 누적 통계

| 항목 | 값 |
|------|----|
| LLM 총 호출 | 83회 |
| LLM 총 입력 토큰 | 400,670tok |
| LLM 총 출력 토큰 | 46,332tok |
| EMB 총 호출 | 742회 |
| EMB 총 토큰 | 77,462tok |

## 파싱 / 청킹 / 임베딩 방식

### 파싱
- 도구: RAGAnything `parse_document` (docling auto)
- docling 실패 시 pypdf 페이지 단위 직접 추출로 fallback
- 블록 타입: `text`, `table`, `image`, `equation` 등

### 청킹
- 도구: RAGAnything `insert_content_list`
- LightRAG 내부 chunking 적용 (토큰 크기 기준)
- 텍스트 블록: splitter -> 엔티티 추출 -> 그래프 삽입
- 테이블/수식: LLM 설명 생성 후 텍스트로 변환하여 삽입

### 임베딩
- 모델: OpenAI text-embedding-3-small (1536차원)
- 저장소: Qdrant (http://localhost:6333)
- 각 청크 -> 벡터 -> Qdrant 컬렉션에 저장

## 파일별 블록 타입 상세

### testpdf.pdf
파싱 방식: pypdf 직접 추출 (페이지 단위 fallback)
- `text`: 57개
