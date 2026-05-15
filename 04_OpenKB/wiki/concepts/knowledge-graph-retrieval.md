---
sources: [summaries/README.md, summaries/06_고급기능과_실전팁.md, summaries/04_Server_WebUI_API.md, summaries/03_Core_API_활용.md, summaries/02_설치와_빠른시작.md, summaries/01_기초개념.md]
brief: Retrieval that blends vector similarity with graph relations and entity context.
---

# Knowledge Graph-Enhanced Retrieval

## Overview

Knowledge graph-enhanced retrieval combines **vector-based search** with **graph-based structure** to improve how systems find and use relevant information. Instead of relying only on text chunks and embedding similarity, it also extracts **entities** and **relationships** from documents, then uses those links to support deeper reasoning, broader context, and more traceable evidence.

This concept is closely reflected in [[summaries/README]], which frames [[concepts/lightrag]] as extending traditional [[concepts/rag]] with a knowledge graph layer. In LightRAG, the retrieval model is available through both the embedded Core API and the higher-level Server/WebUI/API layer. The official guidance for many integrations is to use the [[concepts/server-rest-api]] approach, while Core remains valuable for experiments, embedded workflows, and direct graph control.

## Why it matters

Standard retrieval systems are good at finding semantically similar text, but they often miss:

- relationships between entities across different chunks
- global structure in a document collection
- indirect connections that are not obvious from one passage alone
- context around a specific person, organization, or concept
- retrieval behavior that is easier to inspect and debug in production

Knowledge graph-enhanced retrieval addresses these gaps by using both:

- **vector similarity** for broad semantic matching
- **graph traversal** for structured relationship-aware lookup
- **entity context** for focused answers about people, places, systems, or ideas

This becomes especially useful when queries are entity-centered, when evidence is distributed across many documents, or when answer quality depends on understanding how concepts relate to each other rather than just matching wording.

## Core components

A typical knowledge graph-enhanced retrieval system includes:

- **Text chunks** — original source passages used as evidence
- **Entities** — extracted nodes such as people, organizations, concepts, places, or events
- **Relationships** — links between entities with descriptions or weights
- **Vector store** — embedding search over chunks, entities, and relationships
- **Graph store** — traversal over entity and relation structure
- **Storage/cache layer** — for documents, processing state, and repeated LLM outputs

In LightRAG, these pieces are configured through the core runtime and storage backends described in [[summaries/README]] and the companion docs. Common configuration concerns include:

- `working_dir` for local storage and cache location
- `workspace` for isolating multiple instances
- `kv_storage`, `vector_storage`, `graph_storage`, and `doc_status_storage`
- `embedding_func` and `llm_model_func`
- cache controls like `enable_llm_cache`
- `addon_params` for extraction settings such as language or entity type

The README also stresses that LightRAG needs **three things at minimum**: an LLM, an embedding model, and a storage backend. This makes the system more capable than plain vector retrieval, but also more demanding during setup and indexing.

## How it works

A knowledge graph-enhanced retrieval pipeline usually follows these steps:

1. Split documents into chunks.
2. Extract entities and relationships from each chunk.
3. Merge duplicate or overlapping entities and relations.
4. Embed chunks, entities, and relationships.
5. Store them in both vector and graph backends.
6. Retrieve using a combination of semantic similarity and graph context.
7. Use the retrieved evidence to generate the final answer.

In LightRAG Core, this lifecycle is implemented with asynchronous operations. A typical flow is to create a `LightRAG(...)` instance, call `await rag.initialize_storages()`, insert documents with `ainsert()`, query with `aquery()`, and finally close with `await rag.finalize_storages()`. Skipping storage initialization is a common source of runtime errors such as `AttributeError: __aenter__` or `KeyError: 'history_messages'`.

In LightRAG Server, the same lifecycle is exposed through service endpoints: upload or insert text, monitor indexing progress, and query the indexed corpus through REST or the Ollama-compatible chat interface.

## Query control in practice

LightRAG exposes retrieval behavior through `QueryParam` in Core and through request payloads or chat prefixes in Server mode. Common controls include:

- `mode`: `local`, `global`, `hybrid`, `naive`, `mix`, or `bypass`
- `top_k`: number of entity/relation results to consider
- `chunk_top_k`: number of chunks kept after retrieval and reranking
- `response_type`: output format hint
- `stream`: streaming response support
- `only_need_context` and `only_need_prompt`: return context or prompt only
- `enable_rerank`: reranker usage
- `conversation_history`: LLM context only, not used for retrieval
- `user_prompt`: instructions for how the answer should be presented

A key practice is to separate the actual question from presentation instructions. Put the retrieval question in `query`, and keep formatting or style guidance in `user_prompt`. Mixing them can reduce retrieval quality because the system may treat output instructions as part of the search intent.

The Server API supports the same idea through query payloads such as:

```json
{
  "query": "What is LightRAG?",
  "mode": "mix",
  "include_references": true,
  "include_chunk_content": true
}
```

The README further highlights that reranking can significantly improve retrieval quality, and that the official documentation recommends `mix` mode when a reranker is used. That makes the hybrid path an important default for many real-world retrieval tasks.

## Retrieval styles

Different queries benefit from different retrieval strategies:

- **Local retrieval** — focuses on the neighborhood around one entity
- **Global retrieval** — explores relationships and overall corpus structure
- **Hybrid retrieval** — combines local and global signals
- **Mixed retrieval** — integrates graph retrieval with vector search, often with reranking
- **Naive retrieval** — relies more directly on chunk matching
- **Bypass** — skips the usual retrieval path when appropriate

These modes help the system adapt to fact lookup, context exploration, and corpus-level summarization. In practice, the right mode depends on whether the user is asking about a specific entity, a broad theme, or a relationship chain across multiple documents.

## Document ingestion and provenance

Knowledge graph-enhanced retrieval is only as useful as its evidence trail. LightRAG supports several insertion patterns in Core and Server:

- inserting a single document with `await rag.ainsert("text")`
- inserting multiple documents with `await rag.ainsert(["doc1", "doc2"])`
- assigning stable IDs with `ids=[...]`
- attaching source paths with `file_paths=[...]`
- uploading files through the Server endpoint `/documents/upload`
- inserting plain text through `/documents/text` or `/documents/texts`

Using `file_paths` or upload metadata is especially valuable when citation and traceability matter, because it preserves document provenance for later inspection and evidence-backed answers. When queries request references, LightRAG can return chunk contents and source references alongside the answer for debugging and validation.

## Server, WebUI, and operational workflow

LightRAG Server provides a practical production-friendly wrapper around the retrieval architecture. It exposes:

- a **WebUI** for document upload, indexing status, knowledge graph visualization, query execution, and reference inspection
- a **REST API** for automation and integration
- an **Ollama-compatible chat interface** for frontend interoperability
- an API documentation surface through Swagger UI and ReDoc

The typical server startup is:

```bash
lightrag-server
```

Default server settings include:

| Item | Default |
|---|---|
| host | `0.0.0.0` |
| port | `9621` |
| working dir | `./rag_storage` |
| input dir | `./inputs` |
| log level | `INFO` |

The API docs are available at:

- Swagger UI: `http://localhost:9621/docs`
- ReDoc: `http://localhost:9621/redoc`

Common startup options include `--host`, `--port`, `--timeout`, `--log-level`, `--working-dir`, `--input-dir`, and `--workspace`. The normal operational flow is:

1. start the server
2. upload documents or insert text
3. check indexing status
4. run queries
5. inspect returned context and references when debugging

This server-first workflow is why the official documentation recommends the REST API for integration rather than direct Core access in most production cases. It is the main bridge between retrieval architecture and deployable application design, and it complements [[concepts/server-rest-api]].

## Ollama-compatible chat interface

LightRAG Server offers an Ollama-compatible API, so frontends such as Open WebUI can use it like a model endpoint, for example `lightrag:latest`.

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

This prefix system enables lightweight control over retrieval behavior from chat interfaces, which is useful when users want a conversational surface but still need access to structured retrieval modes. It connects naturally to [[concepts/ollama_compatibility]] and [[concepts/query_mode_selection]].

### Separating retrieval intent from prompt instructions

The chat interface also supports square-bracket instructions after the mode prefix:

```text
/mix[Use a table and cite sources] Explain the main entities.
```

The bracketed text does not directly participate in retrieval. Instead, it guides how the model composes the final answer after search. This separation helps keep the retrieval query focused while still allowing output-format instructions.

## Direct graph management

One important aspect of the Core API is that it allows direct editing of the extracted graph. This makes the retrieval architecture more than an automatic pipeline: it becomes a structure that can be curated.

Supported operations include:

- `create_entity()` to add entities
- `create_relation()` to add relations
- `edit_entity()` to update entity metadata
- `merge_entities()` to combine aliases or duplicates
- deletion methods for entities, relations, and document IDs

These features are useful when the automatic extraction produces duplicate names, when entity normalization is needed, or when the knowledge graph must be corrected manually. Because deletions are irreversible, production workflows should back up data before removing entities or relations.

## Custom KG ingestion

If structured knowledge already exists, it can be inserted directly as a custom knowledge graph with `insert_custom_kg()`.

A custom KG can include:

- `chunks`
- `entities`
- `relationships`

Each item can include source metadata such as `source_id` and `file_path`, allowing imported data to participate in the same retrieval and citation flow as automatically extracted content. This is especially helpful for curated datasets, annotated corpora, and research prototypes.

## Authentication and deployment security

When exposed beyond a local environment, the server should be secured. By default it is accessible without authentication, so external deployments should configure protection explicitly.

### API key setup

```env
LIGHTRAG_API_KEY=your-secure-api-key
WHITELIST_PATHS=/health,/api/*
```

Requests can include the API key in the `X-API-Key` header.

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

The documentation warns that if only API key protection is enabled without WebUI account authentication, guest access may still be possible. For stronger protection, both mechanisms should be configured.

## Production operation

For Linux deployments, LightRAG can run with Gunicorn + Uvicorn workers:

```bash
lightrag-gunicorn --workers 4
```

This is not supported on Windows. The document notes that multiprocess execution can help reduce query blocking when document extraction is CPU-intensive.

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

These deployment details are important because the retrieval architecture often needs to support large uploads, long-running queries, and streaming output without proxy-level interference.

## Strengths

Knowledge graph-enhanced retrieval is useful when you need to:

- connect information scattered across multiple documents
- understand entity-centric context
- reason over relationship chains
- summarize themes or patterns across a corpus
- provide evidence-backed answers with traceable source chunks
- expose retrieval through a practical server, WebUI, and API layer

It is also well suited to embedded applications where developers want fine-grained control over indexing, storage, query modes, and graph maintenance.

## Tradeoffs

The approach is more capable than plain [[concepts/vector-search]]-only retrieval, but it also introduces challenges:

- higher indexing cost due to entity and relation extraction
- dependency on LLM quality for graph construction
- need for careful embedding and storage design
- possible duplicate or noisy entities that require cleanup
- additional operational complexity when using Core directly

For teams that want a simpler deployment model, the [[concepts/server-rest-api]] path may be a better fit than embedding Core directly.

## Relation to other concepts

This concept sits between [[concepts/rag]] and [[concepts/knowledge-graph]]:

- from [[concepts/rag]], it inherits retrieval-augmented generation and chunk-based evidence use
- from [[concepts/knowledge-graph]], it inherits structured entity-relation modeling
- in systems like [[concepts/lightrag]], both are combined into a single retrieval architecture
- in LightRAG Core, the architecture is exposed directly through async APIs, query parameters, and graph management functions
- in LightRAG Server, the same model is exposed through WebUI, REST API, and Ollama-compatible chat
- in the project overview summarized in [[summaries/README]], the recommended practical path is to combine vector search with graph-backed reasoning rather than rely on chunk search alone

## Summary

Knowledge graph-enhanced retrieval improves retrieval quality by combining semantic search with structured graph reasoning and entity context. It is especially effective for multi-document understanding, entity-centered questions, and corpus-wide synthesis. LightRAG implements this concept in both Core and Server form: Core provides direct control over storage, ingestion, query modes, and graph editing, while Server provides the recommended integration surface with WebUI, REST API, chat compatibility, authentication, and deployment tooling.

See also: [[summaries/02_설치와_빠른시작]], [[summaries/03_Core_API_활용]], [[summaries/04_Server_WebUI_API]]

See also: [[summaries/06_고급기능과_실전팁]]
