---
sources: [summaries/README.md, summaries/06_고급기능과_실전팁.md, summaries/05_저장소와_운영.md, summaries/04_Server_WebUI_API.md, summaries/03_Core_API_활용.md, summaries/02_설치와_빠른시작.md]
brief: LightRAG setup spans install choice, runtime config, and storage selection.
---

# Installation and Setup

[[summaries/README]]
[[summaries/02_설치와_빠른시작]]
[[summaries/03_Core_API_활용]]
[[summaries/04_Server_WebUI_API]]
[[summaries/05_저장소와_운영]]
[[summaries/06_고급기능과_실전팁]]

## Overview

LightRAG setup is best understood as a **deployment readiness process**: choosing an installation path, deciding between the Server and Core API, and preparing the runtime configuration needed for the first successful launch. Setup is not only about package installation; it also includes model provider selection, embedding compatibility, storage backend configuration, workspace isolation, API exposure, and validation that the chosen environment matches persisted data.

The README overview reinforces this framing by presenting LightRAG as a learning and deployment stack that spans basic concepts, quick start, Core usage, server/WebUI/API operation, storage and operations, and advanced features. In other words, setup is the entry point into the whole LightRAG lifecycle, not a one-time install step.

This concept spans both local development and production deployment, so it connects closely with [[concepts/async-lifecycle-management]], [[concepts/storage-initialization]], [[concepts/model-configuration]], [[concepts/embeddings]], [[concepts/storage-backends]], [[concepts/storage backend selection|storage backend selection]], [[concepts/workspace isolation]], [[concepts/lightrag-core]], and [[concepts/lightrag-server]].

## Main installation paths

LightRAG supports three practical setup routes:

- **PyPI installation** — the quickest way to try the server or core library
- **Source installation** — best for development, customization, and working with the latest code
- **Docker Compose** — convenient for running the server, WebUI, and storage together

The documentation recommends using `uv` where possible, though `pip` remains supported.

## Server vs. Core usage

A key setup decision is whether you want:

- **LightRAG Server**: includes REST API, WebUI, and Ollama-compatible chat support, installed with the `api` extra
- **LightRAG Core**: the Python library for direct use of the `LightRAG` class inside an application

The README adds an important integration guideline: for most application projects, the official documentation recommends using the **LightRAG Server REST API** rather than integrating directly against the Core library. This makes setup a strategic decision about operational surface, not just a code import choice.

### Server-oriented setup requirements

The Server is the recommended integration surface for most applications. It exposes document upload, indexing, querying, knowledge graph visualization, and chat-style interaction through both WebUI and REST API endpoints.

Typical server setup includes:

- choosing host, port, and working directories
- exposing or protecting API access
- configuring query behavior and model providers
- selecting and initializing storage backends
- defining a workspace for data isolation
- optionally deploying behind a reverse proxy
- validating WebUI and API access through `/docs` or `/redoc`

The default startup command is:

```bash
lightrag-server
```

Default values include:

| Item | Default |
|---|---|
| host | `0.0.0.0` |
| port | `9621` |
| working dir | `./rag_storage` |
| input dir | `./inputs` |
| log level | `INFO` |

Useful documentation endpoints are:

- Swagger UI: `http://localhost:9621/docs`
- ReDoc: `http://localhost:9621/redoc`

Common options include:

```bash
lightrag-server --host 0.0.0.0 --port 9621 --working-dir ./rag_storage --input-dir ./inputs
```

| Option | Description |
|---|---|
| `--host` | Server listen address |
| `--port` | Server port |
| `--timeout` | LLM request timeout |
| `--log-level` | Log level |
| `--working-dir` | RAG storage directory |
| `--input-dir` | Input document directory |
| `--workspace` | Workspace name for data isolation |

### Core-specific setup requirements

When using LightRAG Core, setup is not complete after constructing the `LightRAG(...)` object. The application must also call:

- `await rag.initialize_storages()` before any insert or query operation
- `await rag.finalize_storages()` when shutting down

A minimal Core workflow typically includes:

1. create a working directory
2. initialize the `LightRAG` instance with embedding and LLM functions
3. choose storage implementations
4. initialize storages
5. insert documents with `ainsert()`
6. query with `aquery()`
7. finalize storages

The README makes clear that LightRAG Core requires the same foundational components as the Server: LLM, embedding model, and storage backend. Core setup is therefore especially relevant for [[concepts/async-lifecycle-management]] and [[concepts/storage-initialization]].

## Source and Docker setup patterns

### Source installation

Source installation is the most flexible option and is typically used when you want to develop or customize LightRAG. The workflow includes cloning the repository, installing development dependencies, and optionally building the WebUI.

For source installs, the project provides a wizard-style setup flow to simplify `.env` configuration:

- `make env-base` — configure LLM, embedding, and reranker
- `make env-storage` — configure databases and vector stores
- `make env-server` — configure server runtime settings
- `make env-security-check` — validate security settings before deployment

### Docker Compose

Docker Compose is the most integrated setup path. It is designed to bring up the application alongside its supporting services. Before starting containers, the `.env` file must be configured correctly.

## Required configuration before first run

Installation is only the first part of setup. LightRAG also requires a working runtime configuration. The most important items are:

- **LLM provider and model**
- **Embedding provider and model**
- **Embedding dimension**
- **Optional reranker**
- **Storage backend**
- **Workspace name**

The README emphasizes that LightRAG needs all three major runtime pillars: an LLM, an embedding model, and a storage backend. It also notes that indexing uses the LLM to extract entities and relationships, so the model requirement is higher than in ordinary RAG systems. These settings are central to [[concepts/model-configuration]], [[concepts/embeddings]], and [[concepts/storage-backends]]. Storage selection also affects long-term operability, because persisted data is tied to the backend choice and workspace namespace.

### Storage setup and backend selection

LightRAG separates persistence into four storage categories:

- **KV Storage**: LLM cache, text chunks, and document metadata
- **Vector Storage**: chunk/entity/relation embedding vectors
- **Graph Storage**: entity-relation graph
- **Doc Status Storage**: document processing state

The default implementation is local and file-based:

- `JsonKVStorage`
- `NanoVectorDBStorage`
- `NetworkXStorage`
- `JsonDocStatusStorage`

These are suitable for fast experiments, but production should consider scale, concurrency, and persistence requirements.

Supported backends include:

- **KV**: `JsonKVStorage`, `PGKVStorage`, `RedisKVStorage`, `MongoKVStorage`, `OpenSearchKVStorage`
- **Vector**: `NanoVectorDBStorage`, `PGVectorStorage`, `MilvusVectorDBStorage`, `ChromaVectorDBStorage`, `FaissVectorDBStorage`, `MongoVectorDBStorage`, `QdrantVectorDBStorage`, `OpenSearchVectorDBStorage`
- **Graph**: `NetworkXStorage`, `Neo4JStorage`, `PGGraphStorage`, `AGEStorage`, `MemgraphStorage`, `OpenSearchGraphStorage`
- **Doc Status**: `JsonDocStatusStorage`, `PGDocStatusStorage`, `MongoDocStatusStorage`, `OpenSearchDocStatusStorage`

In practice, setup means deciding early whether you want local simplicity or production-grade external storage. The document warns that changing storage implementations after documents have already been ingested is not fully supported, so backend choice should be made before indexing begins.

### Example provider combinations

The documentation shows that LightRAG can mix providers, such as:

- OpenAI for LLM generation with Ollama for embeddings
- Ollama for both LLM and embeddings

In Core usage, these model functions are passed directly into `LightRAG(...)` as `llm_model_func` and `embedding_func`, so runtime configuration can happen inside Python rather than only through `.env` files.

## Query and retrieval configuration in Core and Server

Setup also includes choosing query behavior. In Core this is commonly done through `QueryParam`, while the Server and Ollama-compatible chat interface expose query modes through request payloads or message prefixes.

Common options include:

- `mode` — `local`, `global`, `hybrid`, `naive`, `mix`, or `bypass`
- `response_type` — answer format hint
- `top_k`, `chunk_top_k` — retrieval limits
- `only_need_context` — return context without answer generation
- `only_need_prompt` — return the final prompt only
- `stream` — streaming output
- `user_prompt` — instructions for answer style
- `enable_rerank` — enable reranking

The server query API can also request citations and chunk content:

```json
{
  "query": "What is LightRAG?",
  "mode": "mix",
  "include_references": true,
  "include_chunk_content": true
}
```

A useful setup practice is to separate the search question from output instructions: keep the question in `query` and move formatting instructions into `user_prompt` or the bracketed chat prompt. This improves retrieval quality and avoids conflating intent with presentation.

This is related to [[concepts/query_mode_selection]], [[concepts/retrieval_modes]], and [[concepts/citations_references]].

## Starting the system

After configuration, the system can be started in one of two main ways:

- **Server mode** with `lightrag-server`
- **Core mode** by running an async application that initializes storages and executes insert/query operations

For Server mode, the UI and API are available once the process is running, including the WebUI, Swagger, and ReDoc.

For Core-only usage, the setup is validated by running an async demo script after setting the required model and storage parameters. A complete Core setup must include storage initialization before inserting or querying data.

## Document insertion and source tracking

LightRAG Core supports inserting single or multiple documents with `ainsert()`. It also allows explicit document IDs and source paths:

- `ids=[...]` for stable document identifiers
- `file_paths=[...]` for provenance and citation tracking

Supplying `file_paths` is recommended when citation quality and source traceability matter. This connects setup choices to later retrieval and reporting workflows.

The Server exposes similar ingestion functionality through REST endpoints such as:

- `/documents/upload` — upload files
- `/documents/text` — insert a single text item
- `/documents/texts` — insert multiple text items
- `/track_status/{track_id}` — check async processing status

Asynchronous indexing endpoints return a track ID, which is important for monitoring ingestion progress in UI or API clients.

## WebUI and operator workflow

The WebUI is part of the setup surface, not just a convenience layer. It supports:

- document upload
- indexing status monitoring
- knowledge graph exploration and visualization
- RAG query execution
- query mode selection
- inspection of retrieved context and references

This makes it useful for validating that configuration, retrieval, and indexing behave as expected before moving into production.

## Ollama-compatible chat setup

LightRAG Server provides an Ollama-compatible API, allowing frontends such as Open WebUI to use LightRAG as if it were a model like `lightrag:latest`.

Query mode can be selected through message prefixes:

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

If no prefix is provided, the default mode is `hybrid`.

The chat interface also supports separating search intent from response instructions:

```text
/mix[Use a table and cite sources] Explain the main entities.
```

The bracketed text does not directly participate in retrieval. Instead, it guides how the model composes the final answer after search.

This is relevant to [[concepts/ollama_compatibility]] and [[concepts/query_mode_selection]].

## Authentication and exposure control

By default, the server is open and does not require authentication. When exposing it externally, the documentation recommends configuring both API key access and account-based JWT authentication for stronger protection.

### API key setup

```env
LIGHTRAG_API_KEY=your-secure-api-key
WHITELIST_PATHS=/health,/api/*
```

Requests can include the API key in the `X-API-Key` header.

Example:

```bash
curl -X POST "http://localhost:9621/documents/scan" -H "X-API-Key: your-secure-api-key" -d ""
```

### JWT account authentication

```env
AUTH_ACCOUNTS='admin:{bcrypt}$2b$12$replace-with-generated-hash'
TOKEN_SECRET='your-secret'
TOKEN_EXPIRE_HOURS=4
```

A password hash can be generated with:

```bash
lightrag-hash-password --username admin
```

The document warns that if only API key protection is enabled without WebUI account authentication, guest access may still be possible. For real protection, both mechanisms should be configured.

This section is relevant to [[concepts/api_security]] and [[concepts/authentication]].

## Production operation

For Linux deployments, LightRAG can run with Gunicorn + Uvicorn workers:

```bash
lightrag-gunicorn --workers 4
```

This is not supported on Windows. The document notes that multiprocess execution can help reduce query blocking when document extraction is CPU-intensive.

## Workspace and storage isolation

A **workspace** is a logical name used to separate the data of multiple LightRAG instances.

Example server usage:

```bash
lightrag-server --port 9621 --workspace team-a
lightrag-server --port 9622 --workspace team-b
```

Isolation varies by storage backend:

| Storage | Isolation method |
|---|---|
| Json/NetworkX/NanoVectorDB/Faiss | workspace subdirectory |
| Redis/Milvus/Mongo/PGGraph | collection/table prefix |
| Qdrant | payload filtering-based multitenancy |
| PGKV/PGVector/PGDocStatus | `workspace` field in table |
| Neo4j/Memgraph | label |
| OpenSearch | index name prefix |

External stores may also provide their own workspace-specific environment variables, which can take precedence over a common `WORKSPACE` setting:

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

Workspace planning is a key part of setup because it affects data separation, multi-instance operation, and safe migration paths.

## Recommended backend choices for setup

The document highlights several practical storage choices:

- **Neo4j** for production graph storage, especially when graph performance matters
- **PostgreSQL** as an all-in-one option for KV, vector, doc status, and graph storage, with Apache AGE possibly required for graph support
- **Milvus** and **Qdrant** for larger-scale vector storage
- **Faiss** for simple local vector indexing
- **OpenSearch** as a unified backend capable of handling all four storage categories

The README's broader list of supported storage systems reinforces that LightRAG is designed to integrate with common databases and vector stores rather than locking users into a single backend. This connects to broader [[concepts/storage backend selection|storage backend selection]] and helps determine whether setup should optimize for simplicity, scalability, or operational consolidation.

## Reverse proxy considerations

Reverse proxy configuration requires special care for upload and streaming endpoints.

### Upload size limits

Uploads may hit Nginx's default 1 MB request-size limit, so `/documents/upload` should be configured with a larger `client_max_body_size`.

```nginx
location /documents/upload {
    client_max_body_size 100M;
    proxy_pass http://localhost:9621;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;
}
```

### Streaming endpoints

Streaming endpoints should generally disable gzip compression:

```nginx
location ~ ^/(query/stream|api/chat|api/generate) {
    gzip off;
    proxy_pass http://localhost:9621;
    proxy_read_timeout 300s;
}
```

These deployment notes connect to [[concepts/reverse_proxying]] and [[concepts/streaming_apis]].

## Compatibility warning

A crucial operational point is that changing the embedding model, embedding dimension, or asymmetric embedding settings can break compatibility with existing vector data. If that happens, the workspace or vector store may need to be cleared and documents re-indexed.

This is an important aspect of [[concepts/vector-store-compatibility]].

The same caution applies to Core usage: if the embedding function changes after data has already been stored, the existing vector data may no longer be valid and should be rebuilt.

## Practical meaning of this concept

In LightRAG, installation and setup mean more than installation commands. A successful setup requires:

1. choosing an installation path
2. deciding between Server and Core usage
3. configuring models, storage, and runtime options
4. selecting a workspace and isolation strategy
5. initializing runtime state correctly
6. validating retrieval, API access, and embedding compatibility
7. starting the server or running the Core workflow

The README clarifies that this setup also serves as the bridge into LightRAG's broader architecture: a system that combines entity extraction, graph construction, vector retrieval, and API-based deployment. Setup is therefore the practical foundation for using LightRAG as a [[concepts/graph-rag]] system in development or production.

See also: [[summaries/README]] and [[summaries/06_고급기능과_실전팁]]