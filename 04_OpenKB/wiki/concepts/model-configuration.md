---
sources: [summaries/README.md, summaries/06_고급기능과_실전팁.md, summaries/05_저장소와_운영.md, summaries/04_Server_WebUI_API.md, summaries/03_Core_API_활용.md, summaries/02_설치와_빠른시작.md]
brief: Configure models and storage together to avoid breakage, reindexing, and poor retrieval.
---

# Model Configuration

Model configuration in LightRAG is the process of aligning generation, retrieval, embedding, reranking, storage, and runtime settings before first use. The installation guide [[summaries/02_설치와_빠른시작]] shows that a working setup depends on more than installing the package: you must configure the LLM, embedding model, embedding dimension, optional reranker, and storage backend so the system can index and answer correctly. The Core API guide [[summaries/03_Core_API_활용]] adds an operational detail: when using the embedded Python API, you must also initialize storages explicitly before inserting or querying. The storage guide [[summaries/05_저장소와_운영]] expands this into a production concern by showing that LightRAG’s persistence is split across KV, vector, graph, and doc-status storage, each of which may use a different backend and workspace isolation strategy. The advanced features guide [[summaries/06_고급기능과_실전팁]] adds that reranking, observability, multimodal ingestion, evaluation, and cache behavior also affect how a configuration performs in practice.

## What must be configured

LightRAG requires at least the following settings:

- **LLM provider and model** — controls answer generation
- **Embedding provider and model** — controls vectorization for retrieval
- **Embedding dimension** — must match the embedding model output
- **Reranker** — optional, but useful for improving retrieval quality
- **Storage backend** — determines where data and indexes are kept
- **Workspace** — isolates data when multiple instances share infrastructure
- **Core runtime storage initialization** — required when using the embedded API

These settings are typically stored in `.env` for server-based deployments, but in Core usage they are passed directly to `LightRAG(...)` and activated with `await rag.initialize_storages()` before any ingestion or query calls.

## Why configuration matters

LightRAG combines multiple AI services and several persistence layers, so mismatched settings can break indexing or retrieval. The most important compatibility rule is that if you change the embedding model, embedding dimension, or asymmetric embedding behavior, existing vector data may no longer be valid. In that case, the workspace or vector store must be cleared and the documents indexed again. This makes model configuration tightly connected to [[concepts/embeddings]], [[concepts/vector-store-compatibility]], [[concepts/retrieval]], and [[concepts/storage backend selection|storage backend selection]].

Storage choice is also part of compatibility. LightRAG splits persistence into four categories:

- **KV Storage** for caches, chunks, and document metadata
- **Vector Storage** for embeddings
- **Graph Storage** for entity-relation structure
- **Doc Status Storage** for indexing state

Because each category may be backed by a different system, a configuration that works in local testing may need to change for production scale, concurrency, or operational reliability.

In Core usage, configuration mistakes also surface as runtime initialization errors. If storages are not initialized, operations such as `ainsert()` or `aquery()` can fail with errors like `AttributeError: __aenter__` or `KeyError: 'history_messages'`. So configuration is not just about choosing models; it also includes lifecycle setup.

## Retrieval quality depends on more than models

The advanced feature set shows that retrieval quality is not determined only by the embedding model. LightRAG can optionally apply a **reranker** to reorder retrieved chunks using a more precise relevance model. The official guidance recommends using reranker with `mix` mode, because the combination can improve query quality and search precision.

Supported reranker providers include:

- Cohere / vLLM compatible rerank APIs
- Jina AI
- Aliyun

Typical configuration can be set globally:

```env
RERANK_BY_DEFAULT=True
```

Or disabled per query:

```json
{
  "query": "What is the relation between A and B?",
  "mode": "mix",
  "enable_rerank": false
}
```

This makes reranking part of model configuration in practice, because the chosen retriever, chunking strategy, and reranker all shape the final answer quality. It is closely related to [[concepts/retrieval-quality]] and [[concepts/rag-evaluation]].

## Practical configuration patterns

The source documents give representative setups that combine model and storage choices in different ways.

### OpenAI LLM + Ollama embedding

This pattern uses one provider for generation and another for embeddings:

```env
LLM_BINDING=openai
LLM_MODEL=gpt-4o
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_BINDING_API_KEY=your_api_key

EMBEDDING_BINDING=ollama
EMBEDDING_BINDING_HOST=http://localhost:11434
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
```

This is useful when a hosted LLM is preferred, but local embedding generation is available.

### Ollama LLM + Ollama embedding

This pattern keeps both generation and embedding local:

```env
LLM_BINDING=ollama
LLM_MODEL=mistral-nemo:latest
LLM_BINDING_HOST=http://localhost:11434
OLLAMA_LLM_NUM_CTX=16384

EMBEDDING_BINDING=ollama
EMBEDDING_BINDING_HOST=http://localhost:11434
EMBEDDING_MODEL=bge-m3:latest
EMBEDDING_DIM=1024
```

This setup emphasizes local control and can reduce dependence on external APIs.

## Core API configuration pattern

When using LightRAG as an embedded library, the configuration is expressed in Python instead of only environment variables. A minimal runtime setup looks like this:

```python
rag = LightRAG(
    working_dir=WORKING_DIR,
    embedding_func=openai_embed,
    llm_model_func=gpt_4o_mini_complete,
)
await rag.initialize_storages()
```

The `working_dir` determines the local storage location, while other constructor arguments connect the runtime to the selected embedding and LLM functions. This is the Core equivalent of choosing model bindings in server-oriented configuration.

### Core initialization-related settings

Beyond model selection, Core exposes parameters that affect how the system behaves at runtime:

- `working_dir`: local cache and default storage location
- `workspace`: namespace for isolating multiple instances
- `kv_storage`, `vector_storage`, `graph_storage`, `doc_status_storage`: backend components for documents, vectors, graph data, and processing state
- `chunk_token_size`, `chunk_overlap_token_size`: chunking controls
- `llm_model_max_async`: concurrency limit for LLM calls
- `enable_llm_cache`, `enable_llm_cache_for_entity_extract`: caching controls
- `addon_params`: extra extraction settings such as language or entity type

These parameters show that configuration in LightRAG spans both model choice and storage/runtime orchestration.

## Storage selection as part of configuration

The storage guide makes clear that storage backend selection is not a minor implementation detail; it is one of the primary configuration decisions.

LightRAG uses four storage categories, each with a default local implementation:

- **KV** → `JsonKVStorage`
- **Vector** → `NanoVectorDBStorage`
- **Graph** → `NetworkXStorage`
- **Doc Status** → `JsonDocStatusStorage`

For production or larger deployments, these can be replaced with external systems such as PostgreSQL, Redis, Milvus, Neo4j, Qdrant, OpenSearch, MongoDB, Memgraph, or Faiss.

Common production patterns include:

- **Neo4j** for graph storage when graph performance matters
- **PostgreSQL** for consolidating KV, vector, doc status, and graph storage in one database
- **Milvus** or **Qdrant** for large-scale vector storage
- **Faiss** for local vector indexing
- **OpenSearch** as a unified backend for all four storage categories

A crucial operational caution is that changing storage implementations after documents have been ingested is not fully supported. This means model configuration, storage selection, and workspace planning should be decided together before indexing begins.

## Workspace and multitenancy

A **workspace** is a logical namespace used to separate data across multiple LightRAG instances. It is especially important when the same backend infrastructure supports multiple teams or environments.

Isolation differs by backend:

- local stores use workspace subdirectories
- Redis, Milvus, Mongo, and PostgreSQL-backed graph/storage often use collection/table prefixes
- Qdrant uses payload-based multitenancy
- PostgreSQL-based stores use a `workspace` field
- Neo4j and Memgraph use labels
- OpenSearch uses index-name prefixes

This makes workspace configuration part of the overall compatibility story, not just a deployment convenience. For example, if you reconfigure storage without preserving the same workspace assumptions, you may appear to “lose” data even though it still exists under another namespace.

## Model selection guidance

The document provides several practical recommendations:

- Prefer **32B+ LLMs** when possible
- Use a **context length of at least 32K**, ideally 64K
- Avoid **reasoning models** during indexing
- Use stronger models during querying if you want better answer quality
- Choose proven embedding models such as `BAAI/bge-m3` or `text-embedding-3-large`
- Consider rerankers like `BAAI/bge-reranker-v2-m3` or Jina models

These guidelines are useful for balancing cost, speed, and quality in [[concepts/model-selection]].

## Multimodal and operational extensions

Advanced usage can also expand configuration beyond standard text retrieval.

### Multimodal integration

LightRAG can integrate with `RAG-Anything` to process PDF, Office documents, images, tables, and equations. This means configuration may need to account for multimodal processors, not just text embedding and generation models.

### Token usage tracking

The `TokenTracker` utility helps measure LLM consumption and cost during ingestion and querying. Because cache hits may avoid provider calls, token usage can remain unchanged even when the system serves a result.

### Knowledge graph export

The knowledge graph can be exported in multiple formats for analysis, backup, or inspection. This is helpful when verifying whether model and storage choices are producing sensible graph structure.

### Cache management

Cache can be cleared globally, and query-related cache can be cleaned selectively. Cache policy is an important part of tuning, because it affects both responsiveness and usage metrics.

### Observability and evaluation

Langfuse integration provides traces for OpenAI-compatible calls, while RAGAS scripts help evaluate retrieval and context quality. These tools are valuable when comparing configurations, especially after changing storage, reranking, or embedding choices.

## Query-time configuration

LightRAG separates retrieval from answer formatting through `QueryParam`. This matters because a poorly chosen query configuration can reduce retrieval quality even when the underlying models are good.

Common options include:

- `mode`: `local`, `global`, `hybrid`, `naive`, `mix`, or `bypass`
- `top_k` and `chunk_top_k`: how much graph and chunk evidence to retrieve
- `enable_rerank`: whether to rerank retrieved chunks
- `response_type`: output style hint
- `user_prompt`: post-retrieval instruction for formatting or language control
- `conversation_history`: only used as LLM context, not for retrieval
- `only_need_context` / `only_need_prompt`: return intermediate artifacts instead of a final answer

A key operational principle is to keep the actual question in `query` and use `user_prompt` only for response-shaping instructions. This avoids mixing retrieval intent with generation style, which can hurt recall and answer quality.

## Environment setup and operational workflow

The setup workflow separates configuration into concerns that should usually be tuned independently:

- `make env-base` for model-related settings
- `make env-storage` for storage selection
- `make env-server` for server parameters
- `make env-security-check` for deployment validation

This separation is useful because model compatibility and storage compatibility often fail in different ways. Model errors usually show up as retrieval or generation quality problems, while storage issues can appear as initialization errors, namespace collisions, or unsupported migrations.

## Key idea

Model configuration in LightRAG is about making all AI and data-layer components agree with one another. Good configuration makes the system reliable, while mismatches in embedding settings, model capabilities, reranker setup, storage backends, workspace names, or Core storage initialization can force reindexing or cause runtime failures. In practice, the best configuration is the one that aligns generation quality, retrieval quality, reranking behavior, storage compatibility, workspace isolation, observability, and the chosen deployment style.

See also: [[summaries/04_Server_WebUI_API]], [[summaries/05_저장소와_운영]], [[summaries/06_고급기능과_실전팁]]

See also: [[summaries/README]]