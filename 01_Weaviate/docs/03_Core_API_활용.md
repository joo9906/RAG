# 03. Core API 활용

LightRAG Core는 Python 애플리케이션 안에 LightRAG를 직접 임베드할 때 쓴다. 공식 문서는 일반 프로젝트 통합에는 Server REST API 사용을 권장하고, Core는 임베디드 앱이나 연구/평가 용도에 더 적합하다고 설명한다.

## 최소 예제

```python
import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import setup_logger

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
os.makedirs(WORKING_DIR, exist_ok=True)

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()
    return rag

async def main():
    rag = None
    try:
        rag = await initialize_rag()
        await rag.ainsert("Your text")
        answer = await rag.aquery(
            "What is this text about?",
            param=QueryParam(mode="hybrid"),
        )
        print(answer)
    finally:
        if rag:
            await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
```

핵심은 `LightRAG(...)` 생성 후 반드시 `await rag.initialize_storages()`를 호출해야 한다는 점이다.

## 주요 초기화 파라미터

| 파라미터 | 설명 |
|---|---|
| `working_dir` | 로컬 캐시와 기본 저장 파일 위치 |
| `workspace` | 여러 인스턴스의 데이터 격리 이름 |
| `kv_storage` | 문서, chunk, cache 저장소 |
| `vector_storage` | embedding vector 저장소 |
| `graph_storage` | entity-relation graph 저장소 |
| `doc_status_storage` | 문서 처리 상태 저장소 |
| `chunk_token_size` | chunk 최대 token 크기. 기본 1200 |
| `chunk_overlap_token_size` | chunk overlap token 크기. 기본 100 |
| `embedding_func` | embedding 함수 |
| `llm_model_func` | LLM 호출 함수 |
| `llm_model_max_async` | LLM 동시 호출 수 |
| `enable_llm_cache` | LLM 응답 cache 사용 여부 |
| `enable_llm_cache_for_entity_extract` | entity extraction cache 사용 여부 |
| `addon_params` | 추출 언어, entity type 등 추가 설정 |

## QueryParam

`QueryParam`은 검색과 답변 생성을 제어한다.

```python
QueryParam(
    mode="mix",
    response_type="Bullet Points",
    top_k=60,
    chunk_top_k=20,
    max_total_tokens=30000,
    only_need_context=False,
    stream=False,
    user_prompt="답변은 한국어로, 근거를 항목별로 정리해줘.",
    enable_rerank=True,
)
```

자주 쓰는 옵션:

| 옵션 | 설명 |
|---|---|
| `mode` | `local`, `global`, `hybrid`, `naive`, `mix`, `bypass` |
| `only_need_context` | 답변 생성 없이 검색 context만 반환 |
| `only_need_prompt` | 최종 prompt만 반환 |
| `response_type` | 답변 형식 힌트 |
| `stream` | streaming 응답 |
| `top_k` | entity/relation 검색 개수 |
| `chunk_top_k` | chunk 검색 및 rerank 후 유지 개수 |
| `conversation_history` | 대화 히스토리. 검색에는 사용되지 않고 LLM context로만 사용 |
| `user_prompt` | 검색 후 답변 생성 방식을 지시 |
| `enable_rerank` | reranker 사용 여부 |

## user_prompt와 query를 분리하기

검색 의도와 출력 형식 지시를 같은 문장에 섞으면 retrieval 품질이 떨어질 수 있다. 검색할 질문은 query에 두고, 출력 형식은 `user_prompt`에 둔다.

```python
param = QueryParam(
    mode="mix",
    user_prompt="답변은 표로 정리하고, 마지막에 요약 3줄을 추가해줘.",
)

answer = await rag.aquery(
    "Scrooge와 주변 인물들의 관계는 무엇인가?",
    param=param,
)
```

## 문서 삽입

단일 문서:

```python
await rag.ainsert("문서 내용")
```

여러 문서:

```python
await rag.ainsert(["문서 1", "문서 2"])
```

ID 지정:

```python
await rag.ainsert(
    ["문서 1", "문서 2"],
    ids=["doc-001", "doc-002"],
)
```

출처 경로 지정:

```python
await rag.ainsert(
    ["문서 내용"],
    file_paths=["docs/source.md"],
)
```

출처 추적과 citation이 중요하면 `file_paths`를 넣는 습관이 좋다.

## 엔티티와 관계 직접 관리

LightRAG는 추출된 지식 그래프를 직접 편집할 수 있다.

```python
rag.create_entity("Google", {
    "description": "Google is a technology company.",
    "entity_type": "company",
})

rag.create_relation("Google", "Gmail", {
    "description": "Google develops Gmail.",
    "keywords": "develops operates email",
    "weight": 2.0,
})
```

수정:

```python
rag.edit_entity("Gmail", {
    "entity_name": "Google Mail",
    "description": "Google Mail is an email service.",
})
```

병합:

```python
rag.merge_entities(
    source_entities=["AI", "Artificial Intelligence", "Machine Intelligence"],
    target_entity="Artificial Intelligence",
)
```

삭제:

```python
await rag.adelete_by_entity("Google")
await rag.adelete_by_relation("Google", "Gmail")
await rag.adelete_by_doc_id("doc-001")
```

삭제는 되돌릴 수 없으므로 운영 환경에서는 백업 후 실행하는 것이 좋다.

## Custom KG 삽입

이미 구조화된 엔티티/관계가 있다면 custom KG로 넣을 수 있다.

```python
custom_kg = {
    "chunks": [
        {
            "content": "Alice and Bob are collaborating on quantum computing.",
            "source_id": "doc-1",
            "file_path": "source.txt",
        }
    ],
    "entities": [
        {
            "entity_name": "Alice",
            "entity_type": "person",
            "description": "Alice is a researcher.",
            "source_id": "doc-1",
            "file_path": "source.txt",
        }
    ],
    "relationships": [
        {
            "src_id": "Alice",
            "tgt_id": "Bob",
            "description": "Alice collaborates with Bob.",
            "keywords": "collaboration research",
            "weight": 1.0,
            "source_id": "doc-1",
            "file_path": "source.txt",
        }
    ],
}

rag.insert_custom_kg(custom_kg)
```

## 자주 나는 오류

| 오류 | 원인 | 해결 |
|---|---|---|
| `AttributeError: __aenter__` | storage 초기화 누락 | `await rag.initialize_storages()` 호출 |
| `KeyError: 'history_messages'` | pipeline status 초기화 누락 | `await rag.initialize_storages()` 호출 |
| embedding dimension 오류 | embedding 모델/차원 변경 | 기존 vector data 삭제 후 재인덱싱 |
| 답변 품질 저하 | query에 출력 지시까지 섞음 | `query`와 `user_prompt` 분리 |
