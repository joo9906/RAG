---
sources: [summaries/README.md, summaries/06_고급기능과_실전팁.md, summaries/05_저장소와_운영.md, summaries/04_Server_WebUI_API.md, summaries/03_Core_API_활용.md]
brief: How LightRAG Core manages, exports, and curates knowledge graph data.
---

# Managing Entities, Relations, and Custom KG Data in LightRAG Core

Managing Entities, Relations, and Custom KG Data in LightRAG Core refers to the direct creation, editing, merging, deletion, import, export, and operational control of structured graph data inside the embedded API. It is a core capability for users who need fine-grained control over the extracted knowledge graph rather than relying only on automatic ingestion and retrieval. See also [[summaries/03_Core_API_활용]].

## What it covers

LightRAG Core treats the knowledge graph as a first-class data layer. This includes:

- creating entities
- creating relations between entities
- editing entity metadata
- merging duplicate or synonymous entities
- deleting entities, relations, or document-linked records
- inserting pre-built custom knowledge graphs
- exporting graph data for inspection, backup, or analysis
- tracking and tuning graph-related operations in production workflows

This makes the Core API useful for curation workflows, experimental setups, migrations, and applications where graph quality must be adjusted manually.

## Core graph operations

### Create entities

Entities can be added directly with descriptive metadata and an entity type.

Example uses include defining a company, person, organization, or domain-specific concept.

### Create relations

Relations connect two entities and can store:

- a description of the relationship
- keywords
- a numerical weight

This supports richer graph semantics than simple link storage.

### Edit entities

Existing entities can be updated, including renaming and replacing descriptions. This is useful when the graph contains noisy extraction results or when canonical names change.

### Merge entities

`merge_entities()` is used to consolidate multiple labels or aliases into a single target entity. This is especially helpful for normalizing synonyms, abbreviations, or duplicate extractions such as "AI", "Artificial Intelligence", and "Machine Intelligence".

### Delete graph data

LightRAG provides deletion methods for:

- entities
- relations
- all data associated with a document ID

Because deletion is irreversible, the document recommends backing up before performing destructive operations in production.

## Custom knowledge graph insertion

If structured KG data already exists, it can be imported directly with `insert_custom_kg()`.

The custom KG payload may contain:

- `chunks` for text content tied to the graph
- `entities` for node definitions
- `relationships` for edge definitions

Each item can include provenance fields such as `source_id` and `file_path`, which support traceability and citation.

This is especially valuable when migrating from external datasets, manually curated graphs, or preprocessing pipelines that already produce structured output.

## Graph export and external analysis

LightRAG can export knowledge graph data for analysis, backup, and offline inspection.

Common export formats include:

```python
rag.export_data("knowledge_graph.csv")
rag.export_data("knowledge_graph.xlsx", file_format="excel")
rag.export_data("knowledge_graph.md", file_format="md")
rag.export_data("knowledge_graph.txt", file_format="txt")
```

Vector data can also be included in exports:

```python
rag.export_data("complete_data.csv", include_vector_data=True)
```

This is useful when you want to audit graph structure, move data between systems, or compare graph state across environments. It also pairs well with [[concepts/knowledge-graph]] and [[concepts/retrieval]].

## Operational control and cost awareness

Graph management is not only about structure; it also affects runtime cost and maintainability.

### Token usage tracking

LLM cost can be tracked with `TokenTracker`:

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

This helps teams understand the cost of ingestion, querying, and graph-related processing. Cache hits may avoid provider calls and therefore may not increase token usage.

### Cache management

LightRAG provides APIs to clear LLM cache when graph or query behavior needs to be reset:

```python
await rag.aclear_cache()
```

or:

```python
rag.clear_cache()
```

For more selective cleanup, the official tool `lightrag.tools.clean_llm_query_cache` can be used. Cache management matters when evaluating graph updates, testing new prompts, or debugging retrieval behavior.

## Quality and retrieval impact

Knowledge graph management directly influences retrieval quality. Better entity normalization, cleaner relations, and traceable custom data can improve downstream search and answer generation.

The advanced usage guide also notes that reranking can further improve query quality, especially when used with `mix` mode. Reranker-supported providers include Cohere/vLLM-compatible rerank APIs, Jina AI, and Aliyun, and the official recommendation is to use reranker with `mix` mode when available. This connects graph quality with [[concepts/retrieval-quality]] and [[concepts/rag-configuration]].

## Multimodal and extended graph sources

LightRAG can also expand graph management beyond text-only workflows through `RAG-Anything` integration. This allows PDFs, Office documents, images, tables, and equations to be processed into multimodal knowledge structures.

This broader ingestion path supports:

- PDF, DOC/DOCX, PPT/PPTX, XLS/XLSX, and image processing
- image, table, and equation-specific processors
- multimodal knowledge graphs
- mixed retrieval over text and multimodal content

This extends the concept beyond classic entity-relation management toward [[concepts/multimodal-rag]] and richer [[concepts/knowledge-graph]] workflows.

## Observability and evaluation

When graph quality changes, observability and evaluation become important.

### Langfuse observability

OpenAI-compatible LLM calls can be traced with the observability extra:

```bash
pip install lightrag-hku[observability]
```

Relevant environment variables:

```env
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_ENABLE_TRACE=true
```

The document notes that this integration is primarily centered on OpenAI-compatible calls, so other providers may not yet be equally supported.

### RAGAS evaluation

LightRAG provides RAGAS-based evaluation support, and query options such as `include_references` and `include_chunk_content` make it easier to inspect the context used during answers:

```json
{
  "query": "What is LightRAG?",
  "mode": "mix",
  "include_references": true,
  "include_chunk_content": true
}
```

This is helpful for measuring how graph quality affects context precision and answer quality.

## Why this matters

Knowledge Graph Management gives LightRAG Core a dual role:

1. it can automatically build a graph from inserted documents, and
2. it can be manually corrected, exported, backed up, or replaced with curated graph data.

That combination makes it stronger for research, evaluation, and embedded applications where graph structure must be inspected and controlled. It also complements [[concepts/retrieval]] by improving the quality of entity and relation data used during search.

## Related ideas

- [[concepts/knowledge-graph]] — broader concept of graph-based memory and structure
- [[concepts/retrieval]] — how graph data supports search and answer generation
- [[concepts/embedding]] — vector-based indexing that works alongside the graph
- [[concepts/retrieval-quality]] — how reranking and graph cleanup improve results
- [[concepts/rag-configuration]] — operational settings that shape retrieval behavior
- [[concepts/multimodal-rag]] — graph management across text and multimodal sources
- [[concepts/observability]] — tracing and debugging production behavior
- [[concepts/rag-evaluation]] — assessing retrieval and answer quality
- [[summaries/03_Core_API_활용]] — source summary covering Core API usage and graph operations
- [[summaries/04_Server_WebUI_API]]
- [[summaries/05_저장소와_운영]]

## Related Documents
- [[summaries/06_고급기능과_실전팁]]


See also: [[summaries/README]]