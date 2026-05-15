---
sources: [summaries/README.md, summaries/06_고급기능과_실전팁.md, summaries/05_저장소와_운영.md, summaries/04_Server_WebUI_API.md, summaries/03_Core_API_활용.md, summaries/02_설치와_빠른시작.md, summaries/01_기초개념.md]
brief: LightRAG blends vector retrieval with entity graphs for richer RAG.
---

# LightRAG

[[summaries/01_기초개념]] and [[summaries/README]] describe LightRAG as a graph-enhanced extension of standard [[concepts/rag]] that combines vector retrieval with entity and relationship extraction for richer retrieval and reasoning.

## Overview

LightRAG is a retrieval-augmented generation system that goes beyond chunk-only retrieval. It extracts **entities** and **relationships** from text, builds a knowledge graph, and combines **vector search** with **graph traversal** to retrieve context.

The README frames this as a practical way to answer questions that depend on:

- connections between concepts across documents
- relationships among people, organizations, events, and topics
- local context around a specific entity
- broad corpus-level structure and themes
- evidence-backed answers with traceable source chunks

LightRAG is designed for both deployed systems and embedded use. In practice, it can run as a standalone server stack or be embedded directly inside a Python application through the [[concepts/core-api]] workflow described in [[summaries/03_Core_API_활용]]. The README also notes that, for integration, the official documentation generally recommends the server REST API over direct Core usage.

## How it differs from standard RAG

Traditional [[concepts/rag]] usually follows this pipeline:

1. Split documents into chunks
2. Embed each chunk
3. Embed the query
4. Retrieve similar chunks by vector distance
5. Generate the answer with the retrieved context

This is simple and effective, but it can miss deeper relationships in the source material.

LightRAG adds another layer:

- it keeps **text chunks** as grounding evidence
- it extracts **entities** such as people, organizations, concepts, and events
- it extracts **relationships** between those entities
- it stores both graph structure and embeddings for retrieval

The result is a hybrid system that can retrieve both semantically similar text and graph-connected knowledge. This is the core idea behind [[concepts/graph-rag]] and [[concepts/knowledge-graph-retrieval]].

## Main components

LightRAG typically uses the following storage and retrieval components:

- **Text chunks** — original text segments used as evidence
- **Entities** — extracted nodes in the knowledge graph
- **Relationships** — links between entities, often with descriptions and weights
- **Vector DB** — embeddings for chunks, entities, and relationships
- **Graph DB** — traversal over entity-relationship structure
- **KV Store** — storage for documents, chunks, and cached LLM outputs
- **Doc Status Store** — tracks ingestion and processing state

The README emphasizes that LightRAG is not just a retrieval layer, but a system that depends on multiple backends. In addition to vector and graph storage, it commonly relies on LLMs, embedding models, and operational storage choices such as Neo4j, PostgreSQL, Milvus, Qdrant, Redis, MongoDB, or OpenSearch.

These pieces allow the system to combine the strengths of [[concepts/vector-search]] and [[concepts/knowledge-graph]].

## Indexing workflow

LightRAG indexing is more involved than standard RAG:

1. Split the document into chunks
2. Use an LLM to extract entities and relationships from each chunk
3. Merge duplicate or overlapping entities and relations
4. Summarize and normalize the extracted graph content
5. Embed chunks, entities, and relationships
6. Store them in vector and graph backends
7. Save document state and cache metadata

Because this process depends on LLM calls, indexing can be slow and costly. The README highlights that LightRAG requires an LLM, an embedding model, and a storage backend, and that the extraction step makes LLM quality especially important. An **entity extraction cache** can reduce repeated costs during testing and reindexing.

A practical warning from the README is that changing the embedding model after indexing may invalidate vector compatibility, which can require full reindexing.

## Query modes

LightRAG supports multiple retrieval modes:

- **`naive`** — basic vector search for simple fact lookup
- **`local`** — retrieves information near a specific entity
- **`global`** — focuses on broad relationships and corpus structure
- **`hybrid`** — combines local and global retrieval
- **`mix`** — integrates graph and vector retrieval; recommended with a reranker
- **`bypass`** — skips retrieval and queries the LLM directly

Among these, `mix` is highlighted as a practical default when reranking is used. The README also reinforces that rerankers improve retrieval quality and are often best paired with hybrid-style retrieval.

## Core API and embedded usage

[[summaries/03_Core_API_활용]] adds an important deployment style: the LightRAG Core API.

Core is intended for Python applications that want to embed LightRAG directly rather than run a separate server. The recommended pattern is:

1. Create `LightRAG(...)`
2. Call `await rag.initialize_storages()` before any use
3. Insert documents with `ainsert()`
4. Query with `aquery()` and a `QueryParam`
5. Finish with `await rag.finalize_storages()`

This means the embedded workflow is fully asynchronous and storage lifecycle management is required. Skipping initialization can cause errors such as `AttributeError: __aenter__` or missing pipeline status fields.

The README adds an important deployment recommendation: for project integration, the official docs prefer using the LightRAG Server REST API rather than relying on Core directly. That makes the distinction between [[concepts/core-api]] and [[concepts/lightrag-server]] especially important.

Core also exposes direct control over graph content:

- create and edit entities
- create and delete relations
- merge duplicate entities
- delete by entity, relation, or document ID
- insert custom knowledge graphs with structured chunks, entities, and relationships

This makes LightRAG not only a retrieval system but also a programmable knowledge-graph workspace.

## Query control with QueryParam

Core usage makes `QueryParam` especially important for shaping retrieval and generation.

Common options include:

- `mode`: retrieval mode such as `local`, `global`, `hybrid`, `naive`, `mix`, or `bypass`
- `response_type`: output format hint
- `top_k`, `chunk_top_k`: graph and chunk retrieval limits
- `max_total_tokens`: token budget
- `only_need_context`, `only_need_prompt`: return context or prompt only
- `stream`: streaming output
- `conversation_history`: LLM context only, not used for retrieval
- `user_prompt`: post-retrieval instructions for answer style
- `enable_rerank`: enable reranking

A key design rule is to separate search intent from output formatting. Put the actual question in `query` and keep formatting instructions in `user_prompt` so retrieval quality is not degraded.

This aligns with [[concepts/query-parameter-design]] and helps preserve retrieval precision when prompts become complex.

## Document insertion and provenance

Core supports inserting one or many documents asynchronously:

- `await rag.ainsert("text")`
- `await rag.ainsert(["doc1", "doc2"])`
- `ids=[...]` to assign document IDs
- `file_paths=[...]` to preserve source paths for traceability and citation

Using `file_paths` is recommended when provenance and citation matter. The README’s overall emphasis on traceability makes this especially relevant for knowledge bases that need source accountability.

## Direct entity and relation management

LightRAG allows direct manipulation of the extracted knowledge graph:

- `create_entity()` to add entities
- `create_relation()` to add relations
- `edit_entity()` to update entity metadata
- `merge_entities()` to consolidate aliases or duplicates
- deletion methods for entities, relations, or document IDs

This is useful for graph curation, fixing extraction noise, and enforcing controlled vocabularies.

## Custom KG insertion

If knowledge is already structured, it can be inserted as a custom knowledge graph via `insert_custom_kg()`.

A custom KG can include:

- `chunks`
- `entities`
- `relationships`

Each item may include source metadata such as `source_id` and `file_path`, making it suitable for imported datasets or curated graph content.

## Installation and deployment paths

LightRAG can be started in several ways depending on the use case:

- **PyPI installation** — fastest way to try the server or core library
- **Source installation** — best for development, customization, and latest code
- **Docker Compose** — best for running server, WebUI, and storage together

The README’s document map shows the intended progression from basic concepts through installation, Core usage, server/API operation, storage/operations, and advanced features. For server and WebUI/API usage, install with the `api` extra. For direct Python usage of the `LightRAG` class, the core package is enough.

This makes LightRAG flexible across both local experimentation and deployed environments. The server can be launched with `lightrag-server`, and the default port is `9621`.

## First-run configuration

LightRAG requires several settings before first use:

- LLM provider and model
- embedding provider and model
- embedding dimension
- optional reranker
- storage backend selection

The document examples show both OpenAI and Ollama-based configurations. A key requirement is that the embedding model and its dimension must match the stored vector data.

This is closely related to [[concepts/model-configuration]], [[concepts/embeddings]], and [[concepts/storage-backends]].

## Setup workflow

For source installations, LightRAG provides a setup wizard-style workflow to generate environment settings:

- `make env-base` — LLM, embedding, reranker settings
- `make env-storage` — PostgreSQL, Neo4j, Redis, Milvus, Qdrant, and other storage options
- `make env-server` — ports, authentication, API keys, SSL
- `make env-security-check` — deployment security validation

This separates model setup, storage setup, server setup, and security checks into distinct steps.

## Quick start workflow

A fast validation path is to install the core package or server package, set the required API key, download sample data, and run an example script.

This quick-start flow is useful for confirming that LLM access, embedding configuration, and storage dependencies are working correctly before larger-scale ingestion.

## Model selection guidance

The official guidance emphasizes choosing models carefully:

- LLMs of **32B or larger** are recommended
- context length should be at least **32KB**, ideally **64KB**
- avoid reasoning models during indexing
- stronger models can improve answer quality at query time
- recommended embedding models include `BAAI/bge-m3` and `text-embedding-3-large`
- reranker candidates include `BAAI/bge-reranker-v2-m3` and Jina models

These recommendations reflect the importance of balancing indexing cost, retrieval quality, and context capacity. They connect closely to [[concepts/model-selection]], [[concepts/reranker]], and [[concepts/context-window]].

## Strengths

LightRAG is a good fit for:

- questions requiring cross-document relationship discovery
- entity-centered exploration
- summarizing a large collection’s structure or themes
- RAG systems that need evidence and explainability
- internal knowledge bases with graph visualization needs
- projects that may need both a library-style core and a deployable server

## Limitations

LightRAG also has tradeoffs:

- indexing is heavier than plain vector RAG
- smaller LLMs may produce weaker entity/relation extraction
- embedding model choice and dimensionality should be planned early
- changing storage backends after ingestion may be difficult
- extracted entities may need manual cleanup or merging
- changing embedding model, dimension, or asymmetric embedding settings can invalidate existing vector data and require reindexing
- embedded Core usage requires explicit storage initialization and cleanup

That last point makes compatibility between embeddings and stored vectors a major operational concern for [[concepts/vector-store-compatibility]].

## Related concepts

- [[concepts/rag]]
- [[concepts/vector-search]]
- [[concepts/knowledge-graph]]
- [[concepts/entity-extraction]]
- [[concepts/reranker]]
- [[concepts/model-selection]]
- [[concepts/embeddings]]
- [[concepts/storage-backends]]
- [[concepts/core-api]]
- [[concepts/query-parameter-design]]
- [[concepts/lightrag-server]]
- [[concepts/graph-rag]]
- [[concepts/knowledge-graph-retrieval]]

## Source

- [[summaries/01_기초개념]]
- [[summaries/02_설치와_빠른시작]]
- [[summaries/03_Core_API_활용]]
- [[summaries/README]]

See also: [[summaries/04_Server_WebUI_API]]

See also: [[summaries/05_저장소와_운영]]

See also: [[summaries/06_고급기능과_실전팁]]