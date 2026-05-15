---
doc_type: short
full_text: sources/README.md
---

# Summary: README

This document is an overview and navigation guide for a LightRAG study bundle based on the official `HKUDS/LightRAG` repository and docs. It explains what LightRAG is, how the companion documents are organized, when LightRAG is useful, and what core ideas to understand before using it.

## Main purpose

The README frames LightRAG as more than a standard [[concepts/rag]] system: it combines vector retrieval with knowledge-graph-style reasoning by extracting entities and relationships from documents. The goal is to improve answer quality for questions that depend on concepts, relations, and document-wide structure rather than isolated text chunks.

## Document structure

The repository is organized as a learning path from fundamentals to practical operation:

1. **Basic concepts** — difference between RAG and LightRAG, core architecture, indexing and querying flow.
2. **Installation and quick start** — PyPI, source, and Docker setup.
3. **Core API usage** — direct use of LightRAG Core from Python.
4. **Server, WebUI, and API** — REST API, WebUI, and Ollama-compatible interfaces.
5. **Storage and operations** — KV, vector, graph, and document-status storage plus workspace and runtime settings.
6. **Advanced features and practical tips** — reranking, multimodal support, evaluation, observability, caching, and troubleshooting.

This makes the README itself a meta-guide: it does not teach one technical feature in depth, but maps the rest of the study set and clarifies how the pieces fit together.

## Key ideas

### LightRAG as graph-enhanced RAG
The document emphasizes that LightRAG is designed to:
- extract entities and relationships during indexing,
- build a knowledge graph from documents,
- combine graph retrieval and vector retrieval during querying.

This positions LightRAG as a system for [[concepts/knowledge-graph-retrieval]] and [[concepts/graph-rag]] rather than a pure embedding search tool.

### Higher model requirements during indexing
A key warning is that LightRAG requires:
- an LLM,
- an embedding model,
- a storage backend.

Because the indexing step uses an LLM to extract entities and relations, the model requirement is stronger than in ordinary chunk-based RAG. The document also notes that changing the embedding model after indexing can break vector compatibility, so reindexing may be required.

### Recommended integration approach
For application integration, the official docs recommend using the **LightRAG Server REST API** rather than calling the Core library directly. This suggests a separation between:
- internal experimentation or local scripting with [[concepts/lightrag-core]], and
- production or system integration through [[concepts/lightrag-server]].

### Retrieval quality and reranking
The README highlights reranking as an important way to improve retrieval quality. It specifically notes that the official documentation recommends the `mix` mode when using a reranker, implying a hybrid retrieval pipeline is preferred in practice.

## When LightRAG is a good fit
The document identifies several common use cases:
- questions about relationships between concepts, people, organizations, or events,
- tasks that need global document flow or thematic understanding,
- document collections where plain chunk retrieval is insufficient,
- knowledge-base services exposed through WebUI or API,
- systems that need to integrate with external stores such as Neo4j, PostgreSQL, Milvus, Qdrant, Redis, MongoDB, or OpenSearch.

In short, LightRAG is well suited for cases where document structure and inter-entity relationships matter as much as local text similarity.

## Source references

The document anchors the study set in the official project and docs:
- GitHub repository: https://github.com/HKUDS/LightRAG
- Paper: https://arxiv.org/abs/2410.05779
- API Server guide
- Core programming guide
- Advanced features guide

## Takeaway

This README is a roadmap for learning LightRAG from fundamentals to deployment. Its central message is that LightRAG combines [[concepts/vector-retrieval]], [[concepts/entity-extraction]], and [[concepts/knowledge-graph]] techniques to support richer document understanding and more reliable question answering than standard RAG.

## Related Concepts
- [[concepts/lightrag-doc-organization]]
- [[concepts/lightrag]]
- [[concepts/installation-and-setup]]
- [[concepts/server-api-and-webui]]
- [[concepts/storage-backend-selection]]
- [[concepts/vector-store-compatibility]]
- [[concepts/knowledge-graph-management]]
- [[concepts/model-configuration]]
- [[concepts/query-parameter-control]]
- [[concepts/rag-evaluation]]
- [[concepts/authentication-and-deployment]]
- [[concepts/multimodal-rag]]
- [[concepts/workspace-isolation]]
