---
sources: [summaries/README.md, summaries/06_고급기능과_실전팁.md, summaries/05_저장소와_운영.md, summaries/04_Server_WebUI_API.md, summaries/03_Core_API_활용.md, summaries/02_설치와_빠른시작.md]
brief: Stored vectors break when embedding settings change, so reindexing is required.
---

# Stored Vectors Stay Usable Only When Embedding Settings Remain Unchanged

Stored vectors are only reliable when the current embedding configuration matches the one used to create them. If the embedding model, vector dimension, asymmetric embedding behavior, embedding function, or related pipeline settings change, previously indexed data may no longer be compatible with the workspace and can become unusable without reindexing.

This idea is emphasized across the LightRAG learning bundle. [[summaries/02_설치와_빠른시작]] warns that changing embedding-related settings may require deleting existing vector data and indexing the documents again. [[summaries/03_Core_API_활용]] notes that embedding dimension changes can trigger vector-store errors and that old vector data should be removed before rebuilding the index. [[summaries/06_고급기능과_실전팁]] adds that embeddings should be chosen carefully from the start because their dimension may be reflected in the storage schema, making later changes expensive. The project overview in [[summaries/README]] reinforces the same point at a higher level: embedding models are part of the core LightRAG stack, and changing them after indexing can require reindexing because stored vectors no longer match the new pipeline.

## Why compatibility matters

A vector store contains numerical representations of text, but those vectors are only meaningful within the exact embedding pipeline that produced them. If the pipeline changes, the stored vectors may:

- have the wrong dimensionality
- live in a different semantic space
- be incompatible with retrieval logic
- produce poor, misleading, or invalid search results
- fail when the application expects a different vector layout

In practice, the vector store is tied not only to the source documents but also to the specific embedding configuration used during indexing.

## Settings that can break compatibility

The source documents identify several changes that can invalidate existing stored vectors:

- **Embedding model changes**
- **Embedding dimension changes**
- **Asymmetric embedding setting changes**
- **Embedding function changes in the Core API**

[[summaries/03_Core_API_활용]] shows that the embedding function is a core initialization parameter in the Python Core API, which means a different `embedding_func` can alter the entire indexing pipeline and make old vectors unusable. [[summaries/06_고급기능과_실전팁]] also emphasizes that embedding choices should be made early because changing them later can require a full rebuild of the index. [[summaries/README]] broadens this into a general architectural warning: LightRAG requires an LLM, embedding model, and storage backend together, and embedding model changes after indexing may force reindexing.

## Core API implications

In [[summaries/03_Core_API_활용]], LightRAG Core is configured with `embedding_func` during construction, and the system requires `await rag.initialize_storages()` before insert or query operations. This makes storage initialization and embedding configuration part of the same operational contract.

A few practical consequences follow:

- the embedding function should be chosen before indexing begins
- changing `embedding_func` may require a full rebuild of the vector store
- existing workspace data may need to be deleted if the new embedding output is incompatible
- embedding-related errors often indicate stale or mismatched stored vectors rather than a problem with the query itself

## Operational consequence

When compatibility is broken, the recommended response is to:

1. remove the existing workspace or vector data
2. re-run ingestion and indexing
3. rebuild the vector store using the new configuration

This ensures retrieval results reflect the current embedding setup rather than a mixed or corrupted index.

## Embedding choices as part of workspace design

[[summaries/06_고급기능과_실전팁]] frames embedding selection as an early architectural decision rather than a minor tuning parameter. The document notes that embedding dimensions can be reflected in the storage schema, so changing models later may introduce migration or rebuild costs.

This aligns with the broader LightRAG guidance in [[summaries/README]], which highlights that LightRAG is built from tightly coupled pieces: the LLM, embedding model, and storage backend all need to fit together. That means embedding configuration should be planned alongside storage and deployment decisions, not treated as a late-stage tweak.

That makes embedding configuration part of the broader workspace design, along with storage and deployment planning:

- choose a stable embedding model early
- treat vector schema as coupled to the embedding output
- plan for reindexing if model changes become necessary
- separate workspaces when experiments need different embedding setups

## Related operational tips

The same document also recommends several practices that help prevent embedding-related breakage from becoming operationally expensive:

- start with a small corpus to verify indexing quality before scaling up
- split large documents into smaller units so failures are easier to recover from
- include source file paths for better traceability and citation quality
- design storage and workspace policies before production rollout

These practices reduce the chance that a bad embedding decision or index rebuild will affect a large corpus at once.

## Related setup dependencies

Vector store compatibility is tightly connected to other configuration areas:

- [[concepts/embeddings]] — the embedding model and output shape determine vector format
- [[concepts/model-configuration]] — model selection affects the entire indexing pipeline
- [[concepts/storage-backends]] — the vector store backend must preserve indexed data correctly
- [[concepts/model-selection]] — choosing stable, validated models reduces compatibility risk
- [[summaries/03_Core_API_활용]] — Core API initialization and `embedding_func` configuration directly affect index validity
- [[summaries/06_고급기능과_실전팁]] — embedding choice, storage schema, and operational planning should be aligned early
- [[summaries/README]] — LightRAG’s official overview emphasizes that embedding models are a required part of the system and that post-index changes can require reindexing

## Practical guidance

To avoid compatibility problems:

- decide on the embedding model early
- keep the embedding dimension fixed once data is indexed
- avoid changing asymmetric embedding behavior after ingestion
- treat `embedding_func` as part of the workspace contract
- document the configuration used for each workspace
- reindex whenever embedding settings must change
- if an embedding change is unavoidable, delete or isolate old vector data before rebuilding
- verify indexing quality on a small corpus before committing to a large workspace

## Summary

Stored vectors stay usable only while the embedding settings used to create them remain unchanged. In LightRAG, changing the embedding model, embedding function, dimension, or asymmetric settings can invalidate existing vector data, so reindexing is often necessary after such changes. Operationally, embedding choice should be treated as a core part of workspace design, not as a late-stage tweak.

See also: [[summaries/04_Server_WebUI_API]]

See also: [[summaries/05_저장소와_운영]]