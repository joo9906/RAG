---
sources: [summaries/06_고급기능과_실전팁.md, summaries/05_저장소와_운영.md]
brief: Tuning parallelism to balance indexing speed, cost, and merge safety.
---

# Concurrency Tuning

Concurrency tuning is the practice of choosing processing limits that improve LightRAG indexing throughput without causing excessive cost, instability, or entity merge conflicts. It matters most during document ingestion, where performance is often constrained by LLM request capacity rather than storage speed.

## Why it matters

In LightRAG, indexing work can run in parallel at several layers:

- multiple files can be inserted at the same time
- multiple asynchronous LLM calls can be active at once
- multiple server workers can serve requests concurrently

If these limits are too low, ingestion becomes slow. If they are too high, systems may hit API limits, increase latency, or create more conflicts during entity merging.

## Key settings

The main concurrency-related settings mentioned in [[summaries/05_저장소와_운영]] are:

- `WORKERS`: number of server worker processes
- `MAX_PARALLEL_INSERT`: number of files processed concurrently
- `MAX_ASYNC`: total number of concurrent LLM requests

These settings work together rather than independently. For example, increasing insert parallelism without enough async capacity may not improve throughput much.

## Practical guidance

The source document recommends starting conservatively:

- begin with `MAX_PARALLEL_INSERT` around 2–10
- increase gradually while observing system behavior
- watch for entity merge conflicts, which can rise when parallelism is too aggressive

This makes concurrency tuning an iterative operational task, not a one-time configuration choice.

## Relationship to other concerns

Concurrency tuning is closely connected to:

- [[concepts/indexing performance]] — throughput during ingestion
- [[concepts/LLM cost control]] — higher concurrency can increase usage and expense
- [[concepts/storage backend selection]] — backend choice can affect scaling behavior
- [[concepts/workspace isolation]] — multi-workspace deployments may need stricter limits

It also supports safer operation alongside [[summaries/05_저장소와_운영]], especially when preparing a production deployment.

## Operational takeaway

A good concurrency configuration balances speed and safety. Start small, measure ingestion behavior, and scale up only as long as throughput improves without increasing merge errors or operational strain.

See also: [[summaries/06_고급기능과_실전팁]]