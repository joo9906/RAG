---
sources: [summaries/06_고급기능과_실전팁.md]
brief: Tracking LLM token usage to measure and control RAG cost.
---

# LLM Usage Tracking

LLM usage tracking is the practice of measuring how many tokens a RAG system consumes during indexing and querying so that cost, throughput, and efficiency can be monitored over time. In LightRAG, this is done with `TokenTracker`, which can be attached to the model call layer to record usage during operations such as insertion and question answering.

## Why it matters

Token usage tracking helps teams:

- estimate operational cost
- compare prompts, modes, and model configurations
- detect unexpectedly expensive workflows
- understand the effect of caching on real provider calls
- validate whether changes to [[concepts/rag-configuration]] improve efficiency

It is especially useful in production or evaluation settings where multiple experiments need to be compared consistently.

## LightRAG approach

The source document describes using `TokenTracker` from `lightrag.utils` and passing it through `llm_model_kwargs` when creating `LightRAG`.

Typical flow:

1. Create a `TokenTracker` instance.
2. Pass it into the LLM model configuration.
3. Reset tracking before the operation you want to measure.
4. Run insertion or query calls.
5. Read usage results afterward.

Example behavior:

- token tracking can cover `ainsert()` and `aquery()` calls
- usage can be inspected after the operation with `get_usage()`
- `reset()` clears previous measurements before a new experiment

## Important detail: cache effects

The document notes that LLM cache hits may not increase token usage because cached responses do not trigger provider calls. This means tracked usage reflects actual model invocations rather than every logical request.

That distinction is important when comparing:

- uncached vs cached runs
- indexing speed vs model cost
- reranker or query-mode experiments with different cache behavior

This also connects to [[concepts/caching]] and [[concepts/troubleshooting]].

## Practical uses

Common use cases include:

- cost benchmarking during development
- measuring the effect of different query modes such as `mix`
- comparing prompt strategies
- monitoring ingestion cost for large document batches
- establishing cost baselines before deployment

For broader operational visibility, token tracking can complement [[concepts/observability]] tools such as Langfuse.

## Related ideas

- [[summaries/06_고급기능과_실전팁]]
- [[concepts/caching]]
- [[concepts/observability]]
- [[concepts/rag-configuration]]
- [[concepts/rag-evaluation]]

## Takeaway

LLM usage tracking gives LightRAG users a simple way to measure real token consumption, understand caching effects, and make more informed cost and performance decisions.