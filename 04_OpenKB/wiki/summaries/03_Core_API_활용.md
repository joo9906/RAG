---
doc_type: short
full_text: sources/03_Core_API_활용.md
---

# Summary: 03. Core API 활용

LightRAG Core is the embedded Python API for integrating LightRAG directly into an application. The document emphasizes that the official recommendation for typical projects is the [[concepts/server-rest-api]] approach, while Core is better suited to embedded apps, experiments, and evaluation workflows.

## Core initialization pattern

A minimal usage flow is:

1. Create a `LightRAG(...)` instance with a working directory and model functions.
2. Call `await rag.initialize_storages()` before any insert or query operations.
3. Use async methods such as `ainsert()` and `aquery()`.
4. Always finalize with `await rag.finalize_storages()`.

The document highlights that storage initialization is mandatory; skipping it leads to common runtime errors.

## Main setup parameters

The Core constructor exposes several configuration options:

- `working_dir`: local cache and default storage location
- `workspace`: namespace for isolating multiple instances
- `kv_storage`, `vector_storage`, `graph_storage`, `doc_status_storage`: storage backends for documents, vectors, graph data, and processing state
- `chunk_token_size`, `chunk_overlap_token_size`: chunking controls
- `embedding_func`, `llm_model_func`: embedding and LLM call functions
- `llm_model_max_async`: concurrency limit for LLM calls
- `enable_llm_cache`, `enable_llm_cache_for_entity_extract`: caching controls
- `addon_params`: extra extraction settings such as language or entity type

These parameters show that Core is highly configurable for local or research-oriented deployments.

## QueryParam and retrieval control

`QueryParam` controls retrieval and answer generation behavior. Important options include:

- `mode`: retrieval mode such as `local`, `global`, `hybrid`, `naive`, `mix`, or `bypass`
- `response_type`: output format hint
- `top_k`, `chunk_top_k`: graph and chunk retrieval limits
- `max_total_tokens`: token budget
- `only_need_context`, `only_need_prompt`: return context or prompt only
- `stream`: streaming output
- `conversation_history`: LLM context only, not used for retrieval
- `user_prompt`: post-retrieval instructions for answer style
- `enable_rerank`: enable reranking

The document stresses that query intent and output formatting should be separated: keep the actual question in `query` and put formatting instructions in `user_prompt` to avoid harming retrieval quality.

## Document insertion

Core supports inserting one or many documents asynchronously:

- `await rag.ainsert("text")`
- `await rag.ainsert(["doc1", "doc2"])`
- `ids=[...]` to assign document IDs
- `file_paths=[...]` to preserve source paths for traceability and citation

Using `file_paths` is recommended when provenance and citation matter.

## Direct entity and relation management

LightRAG Core allows direct manipulation of the extracted knowledge graph:

- `create_entity()` to add entities
- `create_relation()` to add relations
- `edit_entity()` to update entity metadata
- `merge_entities()` to consolidate aliases or duplicates
- deletion methods for entities, relations, or document IDs

This makes Core useful not only for retrieval but also for manual graph curation and cleanup. Deletions are irreversible, so backups are recommended before operating in production.

## Custom KG insertion

If knowledge is already structured, it can be inserted as a custom knowledge graph via `insert_custom_kg()`. The input can include:

- `chunks`
- `entities`
- `relationships`

Each item may include source metadata such as `source_id` and `file_path`, making it suitable for imported datasets or curated graph content.

## Common errors and fixes

The document lists recurring issues:

- `AttributeError: __aenter__` → storage initialization was skipped
- `KeyError: 'history_messages'` → pipeline status was not initialized
- embedding dimension mismatch → embedding model or vector dimension changed; delete old vectors and reindex
- poor answer quality → query and output instructions were mixed together; separate `query` and `user_prompt`

## Overall takeaway

This page is a practical Core API reference focused on embedded usage, async lifecycle management, retrieval tuning, document ingestion, graph editing, and troubleshooting. It is especially relevant for users who need direct programmatic control over LightRAG instead of a standalone service architecture. Related topics include [[concepts/embedding]], [[concepts/retrieval]], and [[concepts/knowledge-graph]].

## Related Concepts
- [[concepts/query-parameter-control]]
- [[concepts/knowledge-graph-management]]
- [[concepts/installation-and-setup]]
- [[concepts/knowledge-graph-retrieval]]
- [[concepts/model-configuration]]
- [[concepts/vector-store-compatibility]]
- [[concepts/lightrag]]
- [[concepts/rag]]
