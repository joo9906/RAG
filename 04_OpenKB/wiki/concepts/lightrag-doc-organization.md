---
sources: [summaries/README.md]
brief: LightRAG docs are organized as a guided path from basics to deployment.
---

# LightRAG Documentation Structure

## Overview

The [[summaries/README]] presents the LightRAG documentation set as a structured learning and reference path. Rather than a single long manual, the materials are divided into focused documents that move from foundational ideas to implementation, operations, and advanced usage.

This structure helps readers learn LightRAG in a practical sequence: first understand the core idea of [[concepts/graph-rag]] and [[concepts/rag]], then install and run it, then integrate it through the Core API or Server API, and finally explore storage, tuning, and troubleshooting.

## Document organization

The README lists six companion documents:

1. **Basic concepts** — explains the difference between RAG and LightRAG, the architecture, and the indexing/query flow.
2. **Installation and quick start** — covers PyPI, source, and Docker setup, plus first execution.
3. **Core API usage** — shows how to use LightRAG Core directly from Python.
4. **Server, WebUI, and API** — describes the server, WebUI, REST API, and Ollama-compatible interface.
5. **Storage and operations** — explains KV, vector, graph, and document-status storage, plus workspace and operational settings.
6. **Advanced features and practical tips** — covers reranking, multimodal use, evaluation, observability, caching, and troubleshooting.

## Why this structure matters

The document structure reflects the way LightRAG is meant to be adopted:

- **Start with concepts** to understand why LightRAG is different from standard chunk-based retrieval.
- **Install and run** the system to verify local setup.
- **Choose an integration style** depending on whether you want direct library access or a service-based architecture.
- **Understand storage backends** because LightRAG depends on multiple stores working together.
- **Apply advanced features** once the core pipeline is working.

This progression supports both newcomers and experienced users who need a reference for a specific subsystem.

## Key design implications

### Learning path orientation
The documentation is organized as a guided learning path, not just a collection of API references. That means the docs assume readers benefit from seeing the full pipeline before diving into implementation details.

### Separation of concerns
The structure separates:
- conceptual understanding,
- setup and execution,
- API usage,
- service deployment,
- storage architecture,
- advanced tuning.

This separation makes it easier to map one topic to one document and reduces overlap across pages.

### Integration options
The README highlights both direct Core usage and Server-based usage. This creates an implicit distinction between:
- local/prototyping workflows with [[concepts/lightrag-core]], and
- production/service workflows with [[concepts/lightrag-server]].

### Operational awareness
The inclusion of a dedicated storage/operations document signals that LightRAG is not just a library but a system with real operational dependencies. Readers need to understand backend compatibility, workspace organization, and persistence behavior.

## Related concepts

- [[concepts/graph-rag]] — the broader retrieval model LightRAG belongs to.
- [[concepts/rag]] — the baseline approach LightRAG extends.
- [[concepts/lightrag-core]] — direct library-level usage.
- [[concepts/lightrag-server]] — service/API-based usage.
- [[concepts/storage-backends]] — the persistence layers used by the system.
- [[concepts/reranking]] — one of the advanced retrieval improvements mentioned in the docs.

## Summary

The LightRAG documentation structure is a staged, modular guide that leads readers from foundational concepts to practical deployment and advanced optimization. It reflects the system’s hybrid nature as both a knowledge-graph-enhanced retrieval framework and an operational service stack.