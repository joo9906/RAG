---
sources: [summaries/README.md]
brief: RAG that augments vector search with entity and relationship graphs.
---

# Graph-Enhanced RAG

## Overview

Graph-Enhanced RAG is a retrieval-augmented generation approach that combines traditional [[concepts/rag]] with explicit graph structure, usually built from extracted entities and relationships. Instead of relying only on chunk similarity, it uses a knowledge graph to capture how concepts connect across a document set.

In the [[summaries/README]], LightRAG is presented as a graph-enhanced RAG system: it extracts entities and relations during indexing, builds a knowledge graph, and then uses both graph retrieval and vector retrieval at query time.

## Core idea

Standard RAG answers questions by retrieving text chunks that are semantically similar to the query. Graph-Enhanced RAG adds another layer:

- identify entities in the source material,
- extract relationships between those entities,
- store the resulting structure as a graph,
- use graph traversal or graph-aware retrieval alongside embedding search.

This helps the system reason about:

- relationships between people, organizations, and events,
- document-wide themes and dependencies,
- indirect connections that may not appear in a single chunk,
- broader context around a question.

## Why it matters

Graph-enhanced retrieval is useful when the answer depends on structure, not just wording. A vector search may find locally similar passages, but a graph can reveal:

- who is connected to whom,
- how ideas are organized,
- what entities co-occur across multiple sections,
- which paths link one topic to another.

This is especially valuable for knowledge bases, technical documentation, research corpora, and other collections where explicit relations improve retrieval quality.

## How LightRAG uses it

According to [[summaries/README]], LightRAG applies graph-enhanced RAG as part of its core design:

1. **Indexing**
   - Documents are processed by an LLM.
   - Entities and relationships are extracted.
   - A graph representation is created.

2. **Retrieval**
   - Vector search finds semantically similar content.
   - Graph retrieval explores entity and relation structure.
   - The system combines these signals to improve answer quality.

3. **Generation**
   - Retrieved graph-linked context is passed to the LLM to produce a response.

This makes LightRAG closer to a [[concepts/knowledge-graph]]-aware retrieval system than a simple chunk-based assistant.

## Key properties

### 1. Better relational reasoning
Graph structure helps answer questions like:
- What is the relationship between A and B?
- Which concepts connect these two topics?
- What is the broader context around this entity?

### 2. More robust document understanding
By modeling cross-document links, the system can support queries that require synthesis across multiple passages.

### 3. Hybrid retrieval
Graph-enhanced RAG typically works best when it is combined with [[concepts/vector-retrieval]] rather than replacing it. LightRAG explicitly uses both.

### 4. Higher indexing cost
Because entity and relation extraction usually requires an LLM, indexing is more expensive than in ordinary RAG systems.

## Tradeoffs

Graph-Enhanced RAG improves structure-aware retrieval, but it introduces extra complexity:

- more preprocessing during indexing,
- dependence on LLM quality for extraction,
- possible graph noise if extraction is inaccurate,
- additional storage and orchestration requirements.

The benefits are strongest when the corpus contains meaningful entities and relations, and when the questions depend on those connections.

## Related concepts

- [[concepts/rag]] — baseline retrieval-augmented generation
- [[concepts/vector-retrieval]] — semantic chunk retrieval
- [[concepts/knowledge-graph]] — graph representation of extracted entities and relations
- [[concepts/entity-extraction]] — identifying entities from source text
- [[concepts/relation-extraction]] — detecting links between entities
- [[concepts/lightrag-core]] — direct programmatic use of LightRAG
- [[concepts/lightrag-server]] — recommended deployment interface for integration

## Source note

This concept is derived from the LightRAG overview in [[summaries/README]], where the system is described as combining graph construction with vector retrieval to improve answer quality for relational and document-wide questions.