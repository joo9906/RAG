---
doc_type: short
full_text: sources/04_Server_WebUI_API.md
---

# Summary: Server, WebUI, API

LightRAG provides a server layer with a WebUI and REST API for document upload, indexing, knowledge graph visualization, querying, and an Ollama-compatible chat interface. The documentation recommends using the Server REST API for integration rather than calling the Core directly.

## Server startup

The server can be started with:

```bash
lightrag-server
```

Default settings include:

| Item | Default |
|---|---|
| host | `0.0.0.0` |
| port | `9621` |
| working dir | `./rag_storage` |
| input dir | `./inputs` |
| log level | `INFO` |

API documentation is available through:

- Swagger UI: `http://localhost:9621/docs`
- ReDoc: `http://localhost:9621/redoc`

Common startup options include host, port, timeout, log level, working directory, input directory, and workspace isolation.

## WebUI capabilities

The WebUI supports the main operational tasks needed for a RAG system:

- document upload
- indexing status monitoring
- knowledge graph exploration and visualization
- RAG query execution
- query mode selection
- inspection of retrieved context and references

This makes the WebUI useful both for interactive use and for debugging retrieval behavior.

## REST API workflow

The typical REST API flow is:

1. Start the server
2. Upload documents or insert text
3. Check indexing progress
4. Run queries
5. Inspect returned context and references when debugging

Asynchronous indexing endpoints return a track ID that can be used to poll status.

Key endpoints include:

| Endpoint | Role |
|---|---|
| `/documents/upload` | Upload files |
| `/documents/text` | Insert a single text item |
| `/documents/texts` | Insert multiple text items |
| `/track_status/{track_id}` | Check processing status |
| `/query` | Standard query |
| `/query/stream` | Streaming query |

## Query configuration

The query example shows how requests can request references and chunk contents:

```json
{
  "query": "What is LightRAG?",
  "mode": "mix",
  "include_references": true,
  "include_chunk_content": true
}
```

A key detail is that `include_chunk_content=true` only has meaning when `include_references=true`. Returned chunk content may appear as an array of strings.

This section connects closely to [[concepts/retrieval_modes]] and [[concepts/citations_references]] because it emphasizes both query routing and inspectable evidence.

## Ollama-compatible chat interface

The server exposes an Ollama-compatible API, allowing frontends such as Open WebUI to use LightRAG as if it were a model like `lightrag:latest`.

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

The prefix system enables lightweight control over retrieval behavior from chat interfaces, which relates to [[concepts/query_mode_selection]] and [[concepts/ollama_compatibility]].

## Separating retrieval intent from prompt instructions

The chat interface supports square-bracket instructions after the mode prefix:

```text
/mix[Use a table and cite sources] Explain the main entities.
```

The bracketed text does not directly participate in retrieval. Instead, it guides how the model composes the final answer after search. This is useful for separating search intent from output formatting or response style.

## Authentication and exposure control

By default, the server is open and does not require authentication. When exposing it externally, the documentation recommends configuring both API key access and account-based JWT authentication for stronger protection.

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

The document warns that if only API key protection is enabled without WebUI account authentication, guest access may still be possible. For real protection, both mechanisms should be configured.

This section is relevant to [[concepts/api_security]] and [[concepts/authentication]].

## Production operation

For Linux deployments, LightRAG can run with Gunicorn + Uvicorn workers:

```bash
lightrag-gunicorn --workers 4
```

This is not supported on Windows. The document notes that multiprocess execution can help reduce query blocking when document extraction is CPU-intensive.

## Nginx reverse proxy considerations

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

## Overall takeaway

This document presents LightRAG Server as the recommended integration surface for most applications. It combines document ingestion, indexing, retrieval, visualization, chat-style interaction, and operational controls into a single service, while also documenting important production concerns such as authentication, process management, and reverse proxy tuning.

## Related Concepts
- [[concepts/server-api-and-webui]]
- [[concepts/authentication-and-deployment]]
- [[concepts/query-parameter-control]]
- [[concepts/knowledge-graph-retrieval]]
- [[concepts/installation-and-setup]]
- [[concepts/lightrag]]
- [[concepts/rag]]
- [[concepts/model-configuration]]
- [[concepts/vector-store-compatibility]]
- [[concepts/knowledge-graph-management]]
