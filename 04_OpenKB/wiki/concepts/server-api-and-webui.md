---
sources: [summaries/README.md, summaries/06_고급기능과_실전팁.md, summaries/05_저장소와_운영.md, summaries/04_Server_WebUI_API.md]
brief: LightRAG Server unifies WebUI, REST API, and chat access in one service.
---

# Server API and WebUI

[[summaries/04_Server_WebUI_API]] describes LightRAG's server-first interface: a single service that exposes a WebUI, REST API, and Ollama-compatible chat endpoint for document ingestion, indexing, retrieval, inspection, and deployment.

## What this concept covers

The **Server API and WebUI** concept is LightRAG's main operational surface for real-world use. The README clarifies that LightRAG is not just a vector-search RAG system: it extracts entities and relationships during indexing, builds a knowledge graph, and then combines graph retrieval with vector retrieval to improve answer quality. The server layer is where that hybrid system is exposed as one usable service.

This concept connects several ideas:

- [[concepts/retrieval_modes]] — query modes such as `local`, `global`, `hybrid`, `mix`, and related variants
- [[concepts/ollama_compatibility]] — using the server as an Ollama-compatible backend
- [[concepts/api_security]] — protecting exposed API endpoints
- [[concepts/authentication]] — API key and JWT account-based access control
- [[concepts/reverse_proxying]] — deployment behind Nginx or similar proxies
- [[concepts/streaming_apis]] — streaming query behavior and proxy settings
- [[concepts/rag-evaluation]] — inspecting references and returned chunk content for evaluation
- [[concepts/rag-configuration]] — tuning mode, reranker, storage, and prompt behavior
- [[concepts/retrieval-quality]] — improving answer quality through retrieval and reranking
- [[concepts/graph-rag]] — graph-enhanced retrieval over extracted entities and relations
- [[concepts/knowledge-graph]] — knowledge graph creation, inspection, and export

## Core role of the server

LightRAG Server acts as the bridge between stored knowledge and user-facing interaction. It combines:

- **document upload**
- **text insertion**
- **indexing and status tracking**
- **knowledge graph visualization**
- **query execution**
- **reference and chunk inspection**
- **chat-style access through an Ollama-compatible interface**
- **advanced retrieval tuning such as reranking**
- **observability, export, and evaluation workflows**

The README adds an important architectural point: the official documentation recommends the server's REST API for project integration, rather than calling the Core library directly. That makes the server the preferred integration layer for most applications.

## WebUI as an operational console

The WebUI is not just a viewer; it is an operations console for a RAG system. It supports:

- uploading documents
- monitoring indexing progress
- exploring the knowledge graph
- selecting query modes
- viewing retrieved context and references
- debugging retrieval behavior and answer quality

Because it exposes the same workflow as the API, the WebUI is especially helpful for understanding how answers are assembled and for comparing retrieval modes during experimentation.

## REST API as the integration layer

The REST API is the preferred integration path for external systems. Its workflow is straightforward:

1. start the server
2. upload documents or insert text
3. track indexing progress
4. issue queries
5. inspect context and references if needed

Important endpoints include:

- `/documents/upload` for file upload
- `/documents/text` for a single text item
- `/documents/texts` for bulk text insertion
- `/track_status/{track_id}` for async processing status
- `/query` for standard requests
- `/query/stream` for streaming responses

This API design supports both synchronous and asynchronous use cases, including long-running indexing jobs that return a track ID for polling.

## Query modes and response shaping

The server supports explicit query mode selection. Mode may be supplied through the REST API or through Ollama-style chat prefixes.

Available modes include:

- `local`
- `global`
- `hybrid`
- `naive`
- `mix`
- `context`
- `mixcontext`
- `bypass`

A key design idea is that query intent can be separated from answer formatting. With chat-style input, a user can write:

```text
/mix[Use a table and cite sources] Explain the main entities.
```

Here, the prefix chooses the retrieval mode, while the bracketed instruction affects the final response style after retrieval. This separation makes the system flexible for both users and developers.

## Chat interoperability via Ollama compatibility

The server can present itself like an Ollama-compatible model endpoint, allowing frontends such as Open WebUI to connect with minimal configuration. This lowers integration friction and makes LightRAG easier to plug into existing chat-oriented workflows.

In practice, this means the server is not only a backend for retrieval, but also a drop-in conversational interface for RAG-enabled chat applications.

## Retrieval quality and advanced server-side tuning

The server layer is also where advanced retrieval controls are usually applied.

### Reranker support

LightRAG can rerank retrieved chunks with a more precise relevance model. The official guidance recommends using reranking with `mix` mode for better query quality.

Supported reranker providers include:

- Cohere / vLLM-compatible rerank API
- Jina AI
- Aliyun

Reranking can be enabled globally or toggled per query, which makes it useful for testing and production tuning. This is a major lever for [[concepts/retrieval-quality]] and [[concepts/rag-configuration]].

### Context inspection for evaluation

The server can return references and chunk content with a query response. That makes it easier to evaluate retrieval quality with tools such as RAGAS and to inspect whether the right context was retrieved before judging the generated answer.

This is especially useful when comparing query modes or testing reranker effects.

## Multimodal document handling

LightRAG can integrate with `RAG-Anything` to process multimodal documents such as PDFs, Office files, images, tables, and equations.

In practice:

- install `raganything`
- create a LightRAG instance
- connect it to `RAGAnything`
- use the server layer to expose the resulting multimodal retrieval workflow

This expands the server from text-only RAG into [[concepts/multimodal-rag]] and strengthens its role as a general document interface.

## Cost tracking, export, and cache control

The server is also the place where operational visibility matters.

### Token usage tracking

`TokenTracker` can be attached to LLM calls to estimate cost and monitor usage. This helps identify expensive workflows and compare query patterns over time.

### Knowledge graph export

The knowledge graph can be exported to CSV, Excel, Markdown, or text, and vector data can optionally be included. This is useful for backup, offline analysis, and external auditing of [[concepts/knowledge-graph]].

### Cache management

The server supports clearing LLM cache entirely, and query-specific cache cleanup is also available through the official cache tool. Cache behavior matters for both performance and cost analysis, especially when token usage seems unexpectedly low due to cache hits.

## Observability and evaluation

The server layer can be instrumented for operational insight.

### Langfuse observability

OpenAI-compatible LLM calls can be traced with Langfuse by installing the observability extra and configuring the required environment variables.

The document notes an important limitation: support is centered on OpenAI-compatible APIs, so integrations such as Ollama, Azure, or AWS Bedrock may not be covered in the same way yet.

### RAGAS evaluation

LightRAG provides RAGAS-based evaluation scripts, and server responses can include references and chunk content to support analysis of metrics like context precision. This makes the server useful not just for serving answers, but for iterating on retrieval quality and prompt design.

## Security and deployment concerns

The document emphasizes that the default server is open unless protected. For external exposure, both API key access and JWT account authentication should be considered.

Operationally, the system also needs deployment tuning:

- **Gunicorn + Uvicorn** is available for Linux production environments
- **Windows** does not support that Gunicorn mode
- **Nginx upload limits** may block large file uploads unless `client_max_body_size` is increased
- **streaming endpoints** should generally disable gzip compression
- large documents may be better handled in smaller chunks for safer ingestion and incremental processing

The README adds another operational caution: because the embedding model is part of the indexing pipeline, changing it later can invalidate stored vectors and require reindexing. That makes storage and model selection part of server-side planning, not just core model setup.

## Practical quality tips through the server

Several best practices are easiest to apply at the server layer:

- separate the query from the response instruction using `user_prompt`
- start with a small corpus to validate indexing quality
- choose embeddings carefully because dimension changes can require reindexing
- test reranker behavior in a production-like environment
- include source file paths for better citation traceability
- define workspace and storage policy before moving to production
- treat the server as the standard integration path rather than embedding Core directly

These align with [[concepts/document-ingestion]], [[concepts/caching]], [[concepts/rag-configuration]], and [[concepts/troubleshooting]].

## Why this concept matters

This concept captures LightRAG's main user-facing architecture:

- one service for ingestion and retrieval
- one interactive UI for inspection
- one API surface for automation
- one chat-compatible interface for frontend interoperability
- one operational layer for reranking, evaluation, observability, and export

In other words, the server layer is the system's integration hub. It turns the core graph-enhanced RAG engine into a usable application platform. The README's broader structure reinforces this by presenting the server, core, storage, and advanced features as connected parts of a single learning path.

See also: [[summaries/05_저장소와_운영]], [[summaries/06_고급기능과_실전팁]], [[summaries/README]]