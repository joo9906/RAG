---
sources: [summaries/README.md, summaries/06_고급기능과_실전팁.md, summaries/05_저장소와_운영.md, summaries/04_Server_WebUI_API.md]
brief: Production LightRAG combines security, storage, concurrency, and tuning choices.
---

# Authentication, Deployment, and Storage Operations

This concept covers the practical steps needed to expose LightRAG safely and operate it reliably in production. It combines access control, process management, reverse proxy configuration, storage backend selection, workspace isolation, concurrency tuning, reranking, observability, and evaluation so the server can be used beyond local development.

Related source: [[summaries/04_Server_WebUI_API]], [[summaries/05_저장소와_운영]], [[summaries/06_고급기능과_실전팁]]

## Core idea

LightRAG is easy to start locally, but production use requires a coordinated set of decisions:

1. **How the server is secured and served**  
   Authentication, worker process management, reverse proxy settings, and observability determine whether the system can be safely exposed to untrusted networks.

2. **How persistence and retrieval are organized**  
   LightRAG separates storage into multiple layers, and the backend choice affects scalability, concurrency, isolation, exportability, and operational complexity.

3. **How retrieval quality and cost are controlled**  
   Reranking, cache policy, token tracking, and evaluation tools affect answer quality and operating cost.

This makes the concept closely related to [[concepts/api_security]], [[concepts/authentication]], [[concepts/reverse_proxying]], [[concepts/streaming_apis]], [[concepts/production_deployment]], [[concepts/multiprocess_execution]], [[concepts/storage backend selection]], [[concepts/vector storage]], [[concepts/workspace isolation]], [[concepts/retrieval-quality]], [[concepts/llm-usage-tracking]], [[concepts/observability]], and [[concepts/rag-evaluation]].

## Authentication strategy

When the server is exposed externally, the document recommends using both of the following:

- **API key authentication** for machine-to-machine access
- **JWT-based account authentication** for WebUI or user sessions

### API key protection

An API key can be configured through environment variables:

```env
LIGHTRAG_API_KEY=your-secure-api-key
WHITELIST_PATHS=/health,/api/*
```

Requests must then include the key in the `X-API-Key` header.

This is useful for restricting access to API endpoints while still allowing selected public or health-check routes.

### JWT account authentication

For user-based access control, the server supports account authentication with JWT tokens:

```env
AUTH_ACCOUNTS='admin:{bcrypt}$2b$12$replace-with-generated-hash'
TOKEN_SECRET='your-secret'
TOKEN_EXPIRE_HOURS=4
```

A password hash can be generated with:

```bash
lightrag-hash-password --username admin
```

### Important security note

The document warns that enabling only an API key may still allow Guest-path access if WebUI account authentication is not also configured. For proper protection, both mechanisms should be used together.

## Production serving

For Linux environments, LightRAG supports a Gunicorn + Uvicorn deployment mode:

```bash
lightrag-gunicorn --workers 4
```

This is intended for operational use and is not supported on Windows.

### Why this matters

- multiple workers can improve throughput
- CPU-heavy document extraction is less likely to block queries
- the server becomes more suitable for production workloads

This operational model connects to [[concepts/production_deployment]] and [[concepts/multiprocess_execution]].

## Storage architecture

LightRAG separates persistence into four storage categories:

- **KV Storage**: stores LLM cache, text chunks, and document metadata
- **Vector Storage**: stores chunk/entity/relation embeddings
- **Graph Storage**: stores the entity-relation graph
- **Doc Status Storage**: tracks document processing state

This separation allows the system to use lightweight local storage for experiments and external databases for production.

### Default storage setup

The default implementation is local and file-based:

- `JsonKVStorage`
- `NanoVectorDBStorage`
- `NetworkXStorage`
- `JsonDocStatusStorage`

These defaults are suitable for quick experiments, but production use should consider scale, persistence, and concurrency requirements.

## Choosing production storage backends

LightRAG supports multiple implementations for each storage type, allowing the deployment to be tuned to the workload.

### Common backend options

**KV Storage**

- `JsonKVStorage`
- `PGKVStorage`
- `RedisKVStorage`
- `MongoKVStorage`
- `OpenSearchKVStorage`

**Vector Storage**

- `NanoVectorDBStorage`
- `PGVectorStorage`
- `MilvusVectorDBStorage`
- `ChromaVectorDBStorage`
- `FaissVectorDBStorage`
- `MongoVectorDBStorage`
- `QdrantVectorDBStorage`
- `OpenSearchVectorDBStorage`

**Graph Storage**

- `NetworkXStorage`
- `Neo4JStorage`
- `PGGraphStorage`
- `AGEStorage`
- `MemgraphStorage`
- `OpenSearchGraphStorage`

**Doc Status Storage**

- `JsonDocStatusStorage`
- `PGDocStatusStorage`
- `MongoDocStatusStorage`
- `OpenSearchDocStatusStorage`

### Practical production choices

The document highlights a few operationally useful options:

- **Neo4j** for graph storage when graph performance matters
- **PostgreSQL** as a consolidated option for KV, vector, doc status, and graph storage, with Apache AGE possibly needed for graph support
- **Milvus** and **Qdrant** for larger-scale vector storage
- **Faiss** for simple local vector indexing
- **OpenSearch** as a unified backend that can support all four storage categories

These choices connect directly to [[concepts/storage backend selection]] and [[concepts/vector storage]].

### Backend selection warnings

A key operational warning is that changing storage implementations after documents have already been ingested is not fully supported. The backend should therefore be chosen before indexing begins.

## Workspace isolation

A **workspace** is a logical namespace used to separate data across multiple LightRAG instances.

Example server startup:

```bash
lightrag-server --port 9621 --workspace team-a
lightrag-server --port 9622 --workspace team-b
```

### Isolation behavior by backend

| Storage backend | Isolation method |
|---|---|
| Json / NetworkX / NanoVectorDB / Faiss | workspace subdirectory |
| Redis / Milvus / Mongo / PGGraph | collection or table prefix |
| Qdrant | payload filtering multitenancy |
| PGKV / PGVector / PGDocStatus | `workspace` field in table |
| Neo4j / Memgraph | label |
| OpenSearch | index name prefix |

External storage systems may also provide their own workspace environment variables, which can override the common `WORKSPACE` setting:

```env
REDIS_WORKSPACE=team-a
MILVUS_WORKSPACE=team-a
QDRANT_WORKSPACE=team-a
MONGODB_WORKSPACE=team-a
POSTGRES_WORKSPACE=team-a
NEO4J_WORKSPACE=team-a
MEMGRAPH_WORKSPACE=team-a
OPENSEARCH_WORKSPACE=team-a
```

Workspace separation is important for multi-tenant deployments and for preventing data collisions across teams or environments. This is a useful [[concepts/workspace isolation|workspace isolation]] pattern.

## Reverse proxy configuration

When placing LightRAG behind Nginx or a similar proxy, endpoint-specific tuning is required.

### Upload endpoint limits

File upload requests may exceed Nginx's default 1 MB body size limit. The `/documents/upload` endpoint should therefore be configured with a larger body size:

```nginx
location /documents/upload {
    client_max_body_size 100M;
    proxy_pass http://localhost:9621;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
}
```

### Streaming endpoint settings

Streaming endpoints should generally disable gzip compression to avoid response buffering issues:

```nginx
location ~ ^/(query/stream|api/chat|api/generate) {
    gzip off;
    proxy_pass http://localhost:9621;
    proxy_read_timeout 300s;
}
```

These settings support stable file upload and streaming behavior and are part of [[concepts/reverse_proxying]].

## Concurrency and throughput

Document indexing performance is described as being limited mainly by LLM throughput rather than storage alone.

Important knobs include:

```env
WORKERS=2
MAX_PARALLEL_INSERT=2
MAX_ASYNC=4
```

| Setting | Meaning |
|---|---|
| `WORKERS` | server worker count |
| `MAX_PARALLEL_INSERT` | number of files processed concurrently |
| `MAX_ASYNC` | total concurrent LLM requests |

The document recommends starting `MAX_PARALLEL_INSERT` conservatively, around 2–10, because too much parallelism can increase entity merge conflicts.

This connects storage behavior, indexing throughput, and operational safety, and relates to [[concepts/indexing performance]] and [[concepts/concurrency tuning]].

## Reranker and retrieval quality

LightRAG can improve retrieval quality by reranking chunks after initial search. The reranker uses a more precise relevance model to reorder retrieved results.

### Recommended usage

The official document recommends using reranker with `mix` mode for better query quality.

### Supported providers

- Cohere / vLLM-compatible rerank APIs
- Jina AI
- Aliyun

### Example configuration

```env
RERANK_BINDING=cohere
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_BINDING_HOST=http://localhost:8000/rerank
RERANK_BINDING_API_KEY=your_key
RERANK_BY_DEFAULT=True
```

Reranking can also be toggled per request:

```json
{
  "query": "What is the relation between A and B?",
  "mode": "mix",
  "enable_rerank": false
}
```

Operationally, reranker testing belongs in production planning because it can materially improve search precision, especially in [[concepts/retrieval-quality]] workflows.

## Multimodal document handling

LightRAG can integrate with `RAG-Anything` to process PDFs, Office files, images, tables, and equations as multimodal input.

### Installation

```bash
pip install raganything
```

### Main capabilities

- PDF, DOC/DOCX, PPT/PPTX, XLS/XLSX, and image processing
- specialized processors for images, tables, and equations
- multimodal knowledge graph support
- joint search over text and multimodal content

In practice, an existing LightRAG instance is connected to `RAGAnything` rather than rebuilt from scratch.

This extends the deployment model into [[concepts/multimodal-rag]] and [[concepts/knowledge-graph]].

## Token usage tracking

To track LLM cost, LightRAG supports `TokenTracker`.

```python
from lightrag.utils import TokenTracker

token_tracker = TokenTracker()

rag = LightRAG(
    working_dir="./rag_storage",
    llm_model_func=llm_model_func,
    llm_model_kwargs={"token_tracker": token_tracker},
    embedding_func=embedding_func,
)

await rag.initialize_storages()

token_tracker.reset()
await rag.ainsert(["document one", "document two"])
answer = await rag.aquery("your question", param=QueryParam(mode="mix"))
print(token_tracker.get_usage())
```

Cache hits do not necessarily trigger provider calls, so token usage may not increase when responses are served from cache. This is part of [[concepts/llm-usage-tracking]] and cost control.

## Knowledge graph export

The knowledge graph can be exported for analysis, backup, or external tooling.

```python
rag.export_data("knowledge_graph.csv")
rag.export_data("knowledge_graph.xlsx", file_format="excel")
rag.export_data("knowledge_graph.md", file_format="md")
rag.export_data("knowledge_graph.txt", file_format="txt")
```

Vector data can also be included:

```python
rag.export_data("complete_data.csv", include_vector_data=True)
```

This makes the exported state easier to inspect and is useful for [[concepts/knowledge-graph]] operations.

## Cache management

Full LLM cache deletion can be done asynchronously or synchronously:

```python
await rag.aclear_cache()
```

or:

```python
rag.clear_cache()
```

Query-specific cache cleanup can be handled with the official tool `lightrag.tools.clean_llm_query_cache`.

Cache policy affects latency, cost, and reproducibility, and belongs to [[concepts/caching]].

## Langfuse observability

To trace OpenAI-compatible LLM calls, install the observability extra:

```bash
pip install lightrag-hku[observability]
```

`.env`:

```env
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_ENABLE_TRACE=true
```

The document notes that Langfuse integration is centered on OpenAI-compatible API calls. Providers such as Ollama, Azure, or AWS Bedrock may not be supported in the same way yet.

This is a production debugging and audit aid related to [[concepts/observability]] and [[concepts/llm-integrations]].

## RAGAS evaluation

LightRAG provides RAGAS-based evaluation scripts. Returning references and chunk content makes it easier to check retrieval quality and compute metrics like context precision.

### Evaluation-oriented query example

```json
{
  "query": "What is LightRAG?",
  "mode": "mix",
  "include_references": true,
  "include_chunk_content": true
}
```

This supports systematic assessment of [[concepts/rag-evaluation]] and retrieval behavior.

## Practical quality tips

1. **Separate the query from output instructions**

   Keep the retrieval question focused and put formatting or style instructions into `user_prompt`.

   Bad example:

   ```text
   표로 정리해서 Scrooge의 관계를 설명해줘.
   ```

   Better example:

   ```python
   await rag.aquery(
       "Scrooge의 주요 인물 관계는 무엇인가?",
       param=QueryParam(
           mode="mix",
           user_prompt="표로 정리하고, 마지막에 핵심 요약을 추가해줘.",
       ),
   )
   ```

2. **Start with a small corpus**

   Check indexing quality before scaling. If entities are split too finely or duplicated, tune entity type, language, prompts, or model quality.

3. **Choose embeddings carefully from the start**

   Embedding dimension may be reflected in the storage schema, so changing models later can require reindexing.

4. **Test reranker in production-like conditions**

   Especially with `mix` mode, reranking can improve retrieval precision.

5. **Split large documents before ingestion**

   Smaller units reduce failure blast radius and make incremental processing easier.

6. **Include source file paths**

   `file_paths` improve citation quality and traceability.

7. **Design storage and workspace policies early**

   Storage migration is not trivial after deployment, so team, customer, and project workspace rules should be planned ahead of time.

These recommendations support robust [[concepts/rag-configuration]], [[concepts/document-ingestion]], and [[concepts/troubleshooting]].

## Troubleshooting overview

The document ends with a compact troubleshooting table covering common issues:

| Symptom | What to check |
|---|---|
| Initialization error | Whether `await rag.initialize_storages()` was called |
| Poor query results | Mode, reranker, top_k, chunk_top_k, and prompt separation |
| Embedding error | Whether the embedding model or dimension changed |
| Slow indexing | LLM performance, `MAX_ASYNC`, `MAX_PARALLEL_INSERT`, and cache |
| Document upload failure | Proxy upload size, `MAX_UPLOAD_SIZE`, and timeout |
| Open WebUI integration problem | Ollama-compatible endpoint, model name `lightrag:latest`, and prefix settings |
| External exposure security issue | API key and JWT auth both configured, plus whitelist checks |

## Operational pattern

A secure and scalable deployment usually follows this pattern:

1. Run LightRAG Server locally or on a protected host
2. Choose storage backends before ingesting documents
3. Configure workspace boundaries to isolate projects or tenants
4. Set API key and/or JWT authentication
5. Deploy with Gunicorn + Uvicorn on Linux when appropriate
6. Place the server behind a reverse proxy
7. Tune upload and streaming settings for large requests and long-lived responses
8. Set concurrency limits to balance throughput and conflict risk
9. Enable reranker where retrieval precision matters
10. Add token tracking, cache policy, and observability for production monitoring
11. Use graph export and RAGAS evaluation for debugging and quality control

## Operational checklist

Before production use, the document recommends confirming the following:

- embedding model and dimension are fixed before indexing starts
- storage backend choices are finalized before ingestion
- API key and JWT auth are both configured when the server is exposed externally
- upload limits, streaming proxy settings, and timeouts are correct
- workspace names do not collide across environments
- indexing cache usage is decided
- LLM cost tracking and log retention are in place
- reranker settings are tested with the target mode, especially `mix`
- observability integration is working for the chosen provider
- backup exists before mass deletion or re-indexing

## Summary

Authentication, deployment, storage operations, and retrieval tuning are tightly connected in LightRAG. A production-ready setup is not just about starting the server; it requires secure access control, reliable process management, careful backend selection, workspace isolation, concurrency tuning, reranking, observability, and evaluation. The goal is to combine these pieces into a deployment that is safe, scalable, and operationally predictable.

See also: [[summaries/README]]