---
sources: [summaries/README.md, summaries/06_고급기능과_실전팁.md, summaries/05_저장소와_운영.md]
brief: Workspace isolation separates LightRAG data across tenants or projects.
---

# Workspace Isolation

Workspace isolation is the mechanism LightRAG uses to keep data from different instances, teams, or projects logically separated while sharing the same application framework and, in some cases, the same backend service.

It is a core operational idea for multi-instance deployments and is discussed in [[summaries/05_저장소와_운영]].

## Why it matters

LightRAG can store documents, embeddings, graphs, and processing state in several different backends. Without a clear isolation strategy, multiple deployments could interfere with one another by writing into the same tables, indexes, collections, or directories.

Workspace isolation helps prevent:

- data collisions between environments
- accidental mixing of documents from different teams
- index or table name conflicts
- operational risk during scaling or multi-tenant use

This makes it a practical [[concepts/storage backend selection|storage backend selection]] concern rather than just a naming convention.

## What a workspace is

A **workspace** is a logical namespace used to partition LightRAG data. For example, two server instances can run with different workspace names:

```bash
lightrag-server --port 9621 --workspace team-a
lightrag-server --port 9622 --workspace team-b
```

Even if they use the same underlying storage technology, their data can remain separated by workspace.

## How isolation is implemented

Isolation depends on the backend type:

- **Json / NetworkX / NanoVectorDB / Faiss**: use a workspace-specific subdirectory
- **Redis / Milvus / Mongo / PGGraph**: use collection or table prefixes
- **Qdrant**: uses payload filtering for multitenancy
- **PGKV / PGVector / PGDocStatus**: store workspace in a table `workspace` field
- **Neo4j / Memgraph**: use labels
- **OpenSearch**: use an index-name prefix

This means workspace isolation is not one universal mechanism; it is adapted to each storage backend’s native model.

## Configuration precedence

LightRAG allows backend-specific workspace environment variables to override the common `WORKSPACE` setting. For example:

```env
REDIS_WORKSPACE=team-a
MILVUS_WORKSPACE=team-a
QDRANT_WORKSPACE=team-a
MONGODB_WORKSPACE=team-a
POSTGRES_WORKSPACE=team-a
NEO4J_WORKSPACE=team-a
MEMGRAPH_WORKSPACE=team-a
OPENSEARCH_WORKSPACE=team-a
```

This is important when a deployment uses multiple storage systems and each needs a consistent namespace.

## Practical guidance

Workspace isolation is most useful when:

- multiple environments share infrastructure
- different customers or teams share a backend cluster
- you want to keep dev, staging, and production data separate
- you are running multiple LightRAG servers on the same machine or service

Good workspace naming should be:

- stable
- unique
- easy to recognize
- consistent across all connected storages

## Related ideas

Workspace isolation connects closely to:

- [[concepts/storage backend selection]]
- [[concepts/concurrency tuning]]
- [[concepts/indexing performance]]

It also depends on operational discipline, because workspace names must be chosen carefully before ingestion begins. As noted in [[summaries/05_저장소와_운영]], changing storage implementations after data has been added is not fully supported, so namespace design should be planned early.

## Summary

Workspace isolation is LightRAG’s multi-tenant separation strategy. It prevents cross-environment data mixing by mapping a workspace name into backend-specific isolation mechanisms such as folders, prefixes, fields, labels, or index names.

See also: [[summaries/06_고급기능과_실전팁]]

See also: [[summaries/README]]