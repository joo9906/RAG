---
sources: [summaries/README.md, summaries/06_고급기능과_실전팁.md, summaries/05_저장소와_운영.md]
brief: Choosing LightRAG storage backends to fit workload, scale, and operations.
---

# Storage Backend Selection

[[summaries/05_저장소와_운영]] explains that LightRAG splits persistence into four storage categories: [[concepts/vector storage|vector storage]], key-value storage, graph storage, and document status tracking. **Storage backend selection** is the design-time process of choosing the right implementation for each category based on workload, scale, concurrency, operational simplicity, and production requirements.

The updated README reinforces an important framing: LightRAG is not just a vector-search app, but a graph-enhanced RAG system that depends on an [[concepts/entity-extraction]]-driven indexing pipeline, a [[concepts/knowledge-graph]], and a retrieval stack that may include reranking and hybrid search. That means storage choices affect not only persistence, but also indexing cost, retrieval quality, workspace isolation, and long-term maintainability.

## What must be selected

LightRAG lets you choose storage independently for:

- **KV Storage**: caches, chunks, document metadata
- **Vector Storage**: embeddings for chunks, entities, and relations
- **Graph Storage**: the entity-relation graph
- **Doc Status Storage**: ingestion and processing state

This modularity means you can keep one part lightweight and another part production-grade. In practice, many deployments use one backend family for most categories and specialized engines only where the workload demands it.

The README adds an important constraint: LightRAG requires an LLM, an embedding model, and a storage backend together. Because indexing extracts entities and relationships with the LLM, backend decisions should be made together with [[concepts/llm-integrations]] and embedding strategy, not after deployment.

## Selection principles

### 1. Start with the workload

The best backend depends on what the system will do most:

- **small local experiments**: file-based defaults are usually enough
- **large-scale ingestion**: external databases become more important
- **graph-heavy workloads**: use a dedicated graph database
- **vector-heavy workloads**: use a dedicated vector database
- **multi-purpose deployments**: favor a unified backend if it simplifies operations
- **multimodal document pipelines**: consider backend and retrieval choices that work well with [[concepts/multimodal-rag]] and rich document content

The README’s use cases make this especially clear: LightRAG is a strong fit when questions involve relationships between concepts, people, organizations, events, or document-level structure. Those workloads often benefit from storage that supports both vector retrieval and graph traversal.

### 2. Decide early

A major warning from the source material is that changing storage implementations after documents have already been indexed is not fully supported. In practice, backend choice is part of initial architecture, not a late optimization.

This matters even more once you consider embedding dimensions, graph structure, and operational metadata. Reindexing later may be costly. The same caution applies to changing embedding models after indexing: vector data may no longer match and the corpus may need to be rebuilt.

### 3. Balance simplicity vs. scalability

The built-in defaults are simple and fast to start with:

- `JsonKVStorage`
- `NanoVectorDBStorage`
- `NetworkXStorage`
- `JsonDocStatusStorage`

These are great for prototyping, but they may not be the best fit for multi-user or high-volume production.

The README’s overall message aligns with this: LightRAG can be used locally for learning and experimentation, but production usage often benefits from external services and a server-oriented deployment model.

### 4. Plan for retrieval quality, not just storage

Storage choice indirectly affects query quality. For example, the advanced tips document recommends testing a reranker in operational settings, especially with `mix` mode. If a backend makes retrieval noisy or slow, even a strong reranker may not fully compensate.

This is why storage planning should be considered together with [[concepts/rag-configuration]], [[concepts/retrieval-quality]], and [[concepts/llm-integrations]]. It also connects to the README’s observation that LightRAG combines vector retrieval with graph-based reasoning, so backend quality influences the whole retrieval path.

### 5. Include traceability and cost visibility

Operational design should also support source paths, exports, and token accounting. Features like [[concepts/llm-usage-tracking]] and [[concepts/knowledge-graph]] export become more useful when the backend and workspace layout are chosen to preserve traceability.

Because indexing depends on LLM extraction of entities and relations, observability is not optional in larger deployments. The backend should make it easy to inspect what was ingested, how it was processed, and where the resulting graph and vectors live.

## Common backend strategies

### Local-first strategy

Use the default local stores when:

- you are testing the pipeline
- data size is small
- concurrency is limited
- deployment complexity should stay low

This is the lowest-friction option, but not the best long-term choice for shared production systems.

### PostgreSQL-centered strategy

PostgreSQL can host multiple storage categories together:

- KV
- vector
- doc status
- graph, with Apache AGE support if needed

This is attractive when you want fewer moving parts and already operate PostgreSQL. It is a strong general-purpose choice, though the source notes that [[concepts/graph storage]] performance may be better with Neo4j for graph-intensive use cases.

### Neo4j-centered strategy

Neo4j is highlighted as a strong production graph backend.

Choose it when:

- graph traversal and graph performance matter
- the entity-relation graph is a core part of the application
- you want a dedicated graph database rather than a general-purpose one

This matches LightRAG’s core idea of extracting relations and reasoning over them rather than relying only on chunk similarity.

### Vector-DB-centered strategy

For large embedding corpora, use a dedicated vector database such as:

- Milvus
- Qdrant
- Faiss for local/simple indexing

Choose this when vector search scale or latency matters more than a unified setup.

This is often the best fit when the retrieval workload is dominated by similarity search, though LightRAG’s graph layer still matters for relation-aware questions.

### OpenSearch-unified strategy

OpenSearch can serve as a unified backend for all four categories:

- KV
- vector
- graph
- doc status

This is useful when you want a single storage platform and already run OpenSearch operationally. It can reduce integration complexity, though it may not always be the best specialist choice for every workload.

## Workspace as part of backend choice

Backend selection also affects how you isolate data across multiple LightRAG instances. The document describes a [[concepts/workspace isolation|workspace isolation]] model where different stores separate tenant or project data differently:

- subdirectories for local stores
- collection/table prefixes for some external databases
- labels for Neo4j and Memgraph
- index prefixes for OpenSearch
- workspace fields for PostgreSQL-backed stores

So choosing a backend is also choosing an isolation model. This is especially important if you expect multiple teams, customers, or environments to share the same infrastructure.

The README’s note about server-based operation also fits here: if you plan to expose LightRAG through WebUI or REST API, workspace separation becomes part of the service boundary, not just an internal implementation detail.

## Operational factors to consider

When selecting a backend, also account for:

- expected data volume
- number of concurrent writers
- backup and restore needs
- deployment environment
- migration difficulty
- compatibility with auth and networking requirements
- whether one backend can cover multiple storage categories
- whether cache behavior and query history should be kept separate from main data
- whether you will need export for analysis or backup later

These are especially important in production because indexing, deletion, re-indexing, and cache cleanup can be disruptive if the backend is not chosen carefully.

The README also implies that operational design should support the full LightRAG lifecycle: indexing, retrieval, server exposure, and maintenance. That makes backend choice a system-level decision rather than a database-only one.

## Advanced operational considerations

### Reranker compatibility

If you plan to use a reranker, the advanced features document recommends `mix` mode for better results. That means backend selection should support retrieval pipelines where reranking can be added without operational friction.

### Token and cache behavior

When LLM cost tracking matters, you may attach `TokenTracker` and manage cache more deliberately. Backends that make caching and cleanup easier can improve observability and cost control.

### Export and analysis workflows

If you need to inspect or back up the knowledge graph, pick a backend that makes export straightforward and predictable. The ability to export to CSV, Excel, Markdown, text, or vector-inclusive data can influence backend choice if your team relies on offline analysis.

### Multimodal ingestion

If your system ingests PDFs, Office files, images, tables, or equations through `RAG-Anything`, backend choice should account for larger and more heterogeneous inputs. In such cases, stable ingestion, workspace layout, and retrieval performance matter more than minimal setup.

### Observability and evaluation

If you need Langfuse tracing or RAGAS evaluation, the backend should fit into a workflow where query content, references, and chunk data can be inspected easily. That is often easier when the storage layout is planned up front.

## Practical guidance

A simple rule of thumb:

- **prototype**: use defaults
- **small production**: PostgreSQL or OpenSearch may simplify operations
- **graph-centric app**: choose Neo4j
- **vector-centric app**: choose Milvus, Qdrant, or Faiss depending on scale
- **multi-tenant setup**: verify workspace isolation behavior before launch
- **retrieval-quality-focused setup**: test `mix` mode plus reranker before finalizing storage
- **observability-heavy setup**: ensure cache, export, and tracing workflows are compatible with the backend

Also remember the README’s guidance: if you are integrating LightRAG into a product, the REST API path is often preferred over direct Core integration. That recommendation indirectly favors backends and storage layouts that work well behind a service boundary.

## Related concepts

- [[concepts/workspace isolation]]
- [[concepts/vector storage]]
- [[concepts/graph storage]]
- [[concepts/indexing performance]]
- [[concepts/concurrency tuning]]
- [[concepts/retrieval-quality]]
- [[concepts/rag-configuration]]
- [[concepts/llm-usage-tracking]]
- [[concepts/knowledge-graph]]
- [[concepts/lightrag-server]]
- [[concepts/lightrag-core]]
- [[summaries/05_저장소와_운영]]
- [[summaries/README]]

## Summary

Storage backend selection in LightRAG is a design-time decision that trades off simplicity, scale, retrieval quality, and operational risk. The main goal is to match each storage category with a backend that fits the workload, while preserving a clear plan for workspace isolation, concurrency, cache management, observability, and long-term maintenance. The README adds that this choice should be made with the broader LightRAG architecture in mind: entity extraction, graph construction, hybrid retrieval, and production server usage all depend on storage being chosen well from the start.

## Related Documents
- [[summaries/06_고급기능과_실전팁]]