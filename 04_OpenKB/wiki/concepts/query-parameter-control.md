---
sources: [summaries/README.md, summaries/06_고급기능과_실전팁.md, summaries/04_Server_WebUI_API.md, summaries/03_Core_API_활용.md]
brief: QueryParam controls retrieval mode, reranking, context, and answer style in LightRAG.
---

# QueryParam and Mode Control in LightRAG

`QueryParam` is the main control surface for shaping how LightRAG retrieves information, assembles context, and generates answers. It separates retrieval intent from output instructions, helps manage token and context budgets, and lets you choose the right retrieval mode for the task. In [[summaries/03_Core_API_활용]], this is presented as a core part of using the Core API effectively, and [[summaries/04_Server_WebUI_API]] extends the same ideas into server, WebUI, and chat-based workflows. The advanced guidance in [[summaries/06_고급기능과_실전팁]] adds operational tuning around reranking, evaluation, caching, and production quality.

## Why it matters

In LightRAG, a query is not only a question. It may also specify:

- how the system should search
- which retrieval mode to use
- how much context to keep
- whether reranking should be applied
- whether to return context, a prompt, or a final answer
- how the final answer should be formatted
- whether references or chunk contents should be exposed for debugging or evaluation

Careful parameter control improves retrieval quality, reduces confusion between search intent and output style, and makes the system easier to debug and evaluate. This is especially important when using [[concepts/retrieval_modes]], [[concepts/rag-evaluation]], and [[concepts/retrieval-quality]].

## Core idea: separate query from instructions

A central guideline is to keep the actual information-seeking question in `query` and place answer-style instructions in `user_prompt`.

- `query`: the factual or semantic question
- `user_prompt`: formatting, tone, language, or presentation instructions

For example, in server chat usage you can write:

```text
/mix[Use a table and cite sources] Explain the main entities.
```

Here, the retrieval question is the part after the prefix, while the bracketed instruction influences answer generation after search. This mirrors the `QueryParam.query` + `QueryParam.user_prompt` separation used in the Core API.

This separation helps avoid degraded retrieval performance that can happen when output instructions are mixed directly into the search question.

## Main `QueryParam` controls

### `mode`

Controls the retrieval strategy.

Common modes include:

- `local`
- `global`
- `hybrid`
- `naive`
- `mix`
- `context`
- `mixcontext`
- `bypass`

These modes determine how LightRAG combines graph-based retrieval, chunk retrieval, or simplified answering behavior. The server’s Ollama-compatible chat interface also uses these modes through message prefixes such as `/mix`, `/global`, or `/hybrid`.

The advanced notes recommend `mix` mode when reranking is enabled, because the combination can improve query quality and precision.

### `top_k`

Limits how many entities or relations are retrieved from the graph side of the pipeline.

### `chunk_top_k`

Limits how many chunk candidates remain after retrieval and reranking.

### `max_total_tokens`

Sets the overall token budget for the final context sent to the LLM.

### `only_need_context`

Returns the retrieved context without generating a final answer.

### `only_need_prompt`

Returns the final prompt only, which is useful for inspection and debugging.

### `response_type`

Provides a hint about the expected answer format, such as bullet points or another structured style.

### `stream`

Enables streaming response generation.

### `conversation_history`

Provides prior conversational turns to the LLM. The document notes that this is used only as LLM context, not as part of retrieval.

### `enable_rerank`

Turns reranking on or off after retrieval.

Reranking is a major quality control feature discussed in [[summaries/06_고급기능과_실전팁]]. It reorders retrieved chunks using a more precise relevance model. LightRAG can work with rerank providers such as Cohere/vLLM-compatible rerank APIs, Jina AI, and Aliyun. The official recommendation is to use reranker with `mix` mode for better quality.

## Query control in the Server and WebUI

The server layer exposes the same retrieval concepts through REST endpoints, WebUI controls, and chat-style interfaces. LightRAG Server provides document upload, indexing, knowledge graph visualization, querying, and an Ollama-compatible interface, making it easier to use query-mode selection without writing Core API code.

The server’s query endpoints support explicit control over response content such as references and chunk contents, and asynchronous indexing returns a track ID for status polling. This operational layer makes `QueryParam`-style control visible in everyday usage, not just in code.

The WebUI also supports:

- document upload
- indexing status tracking
- knowledge graph exploration and visualization
- RAG query execution
- query mode selection
- inspection of retrieved context and references

This means query control is both a programmatic concern and an interactive workflow concern.

## How advanced features affect mode control

### Reranker-aware querying

When reranking is available, `mode="mix"` is often the most effective default because it combines retrieval breadth with better candidate ordering. The reranker can also be disabled per request using `enable_rerank: false`, which is useful for comparisons and debugging.

### Evaluation-friendly queries

For RAGAS or other evaluation workflows, it helps to return evidence alongside answers. Options like `include_references` and `include_chunk_content` make it easier to inspect context precision and assess whether the retrieved evidence supports the answer.

Example:

```json
{
  "query": "What is LightRAG?",
  "mode": "mix",
  "include_references": true,
  "include_chunk_content": true
}
```

### Debugging intermediate stages

`only_need_context` and `only_need_prompt` are useful when diagnosing whether a bad answer comes from retrieval or from generation. They let you inspect what the system selected before the final answer is produced.

## Typical usage pattern

A common Core API configuration looks like this:

```python
QueryParam(
    mode="mix",
    response_type="Bullet Points",
    top_k=60,
    chunk_top_k=20,
    max_total_tokens=30000,
    only_need_context=False,
    stream=False,
    user_prompt="답변은 한국어로, 근거를 항목별로 정리해줘.",
    enable_rerank=True,
)
```

This configuration balances retrieval breadth, context quality, and output style control.

A similar idea in the server chat interface would be to use `/mix` or another prefix, then add bracketed instructions to shape the final answer without altering retrieval intent.

## Practical guidance

### Use `mode` intentionally

Choose the retrieval mode based on the task:

- graph-heavy questions may benefit from graph-oriented modes
- text-centric questions may need chunk retrieval
- mixed tasks often work well with `mix` or `hybrid`
- production retrieval often benefits from testing `mix` with reranker enabled

### Keep token budgets under control

`top_k`, `chunk_top_k`, and `max_total_tokens` should be tuned together. Larger values can improve recall but may increase noise or exceed context limits.

### Use reranking when precision matters

`enable_rerank=True` can help filter weak candidates after retrieval, especially when many chunks are returned.

The advanced document also notes that query quality can improve when reranking is combined with a suitable mode, and that reranker settings may be controlled globally or per request.

### Inspect intermediate outputs during debugging

`only_need_context` and `only_need_prompt` are useful for understanding what the system is retrieving and how the final prompt is assembled.

The server’s reference and chunk-content options serve a similar debugging purpose by letting you inspect what evidence was used to produce the answer.

### Track cost and performance

When a project needs usage visibility, `TokenTracker` can be attached to monitor LLM calls and token consumption. This is especially useful when comparing modes, reranker settings, or different prompt strategies. Cache hits may not increase token usage because they can avoid provider calls.

### Plan for production behavior

In real deployments, `QueryParam` is only one part of system quality. The surrounding setup matters too:

- choose embeddings carefully, because embedding dimension can affect storage schema
- keep source file paths for better citations and traceability
- split large documents so failures are easier to recover from
- design storage and workspace strategy early
- test reranker and cache behavior before rollout

These practices come from [[summaries/06_고급기능과_실전팁]] and reinforce that mode control is tied to broader operational design.

## Common failure mode

A recurring problem is combining the question and the desired output style in a single sentence. This can confuse retrieval and reduce answer quality.

Better practice:

- put the semantic question in `query`
- put formatting instructions in `user_prompt`

In server chat usage, the same principle applies by separating the query prefix and bracketed instructions. This is one of the most important operational lessons from [[summaries/03_Core_API_활용]] and is reinforced by [[summaries/04_Server_WebUI_API]].

## Related ideas

- [[concepts/retrieval]] — how queries influence search behavior
- [[concepts/prompt-engineering]] — how `user_prompt` shapes answer generation
- [[concepts/knowledge-graph]] — how graph entities and relations participate in retrieval
- [[concepts/retrieval_modes]] — differences among LightRAG search modes
- [[concepts/retrieval-quality]] — how reranking and mode choice affect results
- [[concepts/rag-evaluation]] — using references and chunk content to assess quality
- [[concepts/server_architecture]] — how the server exposes retrieval and chat workflows
- [[summaries/03_Core_API_활용]] — source document covering Core API usage and `QueryParam`
- [[summaries/04_Server_WebUI_API]] — source document covering server, WebUI, API, and Ollama-compatible chat control
- [[summaries/06_고급기능과_실전팁]] — source document covering reranker, evaluation, token tracking, and production tips

See also: [[summaries/README]]