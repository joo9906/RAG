---
sources: [summaries/README.md, summaries/06_고급기능과_실전팁.md]
brief: Methods for measuring RAG retrieval and answer quality.
---

# RAG Evaluation

RAG evaluation is the practice of measuring how well a retrieval-augmented generation system finds relevant context and produces useful answers. In LightRAG, this is especially important because retrieval mode, reranking, chunking, and prompt design can all change the final quality of results.

## Why it matters

A RAG system can appear to work while still retrieving weak context or generating answers from incomplete evidence. Evaluation helps answer questions such as:

- Are the retrieved chunks actually relevant?
- Does reranking improve precision?
- Is the system returning enough supporting context?
- Do changes in embedding or storage settings affect output quality?

This makes evaluation a core part of [[concepts/retrieval-quality]], [[concepts/rag-configuration]], and [[concepts/troubleshooting]].

## LightRAG evaluation support

The source document notes that LightRAG provides RAGAS-based evaluation scripts. RAGAS is useful for assessing retrieval and generation behavior with metrics such as context precision and related quality indicators.

A practical advantage in LightRAG is that evaluation becomes easier when query APIs return references and chunk content. That allows you to inspect not only the final answer, but also the evidence behind it.

Example query options mentioned in the source:

```json
{
  "query": "What is LightRAG?",
  "mode": "mix",
  "include_references": true,
  "include_chunk_content": true
}
```

These options help during debugging and make it easier to compare retrieval behavior across different modes.

## What to evaluate

Common evaluation targets include:

### Retrieval quality
- Whether the system finds the right chunks
- Whether the chunks are ranked appropriately
- Whether reranking improves the results

### Context quality
- Whether retrieved evidence is sufficient
- Whether the retrieved context is too broad, too narrow, or irrelevant
- Whether the answer is grounded in the returned references

### End-to-end answer quality
- Whether the final answer is accurate
- Whether it reflects the evidence in the context
- Whether formatting and task instructions are followed correctly

## Evaluation workflow in practice

A useful workflow is:

1. Run a fixed set of representative queries.
2. Compare retrieval modes such as `naive`, `local`, `global`, `hybrid`, and `mix`.
3. Enable `include_references` and `include_chunk_content` for inspection.
4. Check whether reranking improves context precision.
5. Use RAGAS scripts to score the results consistently over time.
6. Repeat after changing embeddings, chunking, prompts, or storage settings.

This aligns with the operational advice in [[summaries/06_고급기능과_실전팁]].

## Relationship to other concepts

- [[concepts/retrieval-quality]] — evaluation is how retrieval quality is measured and compared.
- [[concepts/llm-usage-tracking]] — cost tracking complements quality evaluation.
- [[concepts/observability]] — traces and logs help explain why a result changed.
- [[concepts/rag-configuration]] — configuration changes should be validated through evaluation.
- [[concepts/caching]] — cache effects can influence observed usage and repeatability.

## Practical takeaway

RAG evaluation is not a one-time benchmark; it is an ongoing part of tuning and operating a system. In LightRAG, RAGAS scripts plus reference-returning queries provide a practical way to measure whether changes in reranking, chunking, embeddings, or prompts actually improve retrieval-grounded answers.

See also: [[summaries/README]]