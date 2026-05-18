---
sources: [summaries/2_1_1 24년 하반기 신입사원_조인스_이재은_학술교재_0002.md, summaries/2_1_1 24년 하반기 신입사원_조인스_이재은_학술교재_0001.md, summaries/README.md, summaries/06_고급기능과_실전팁.md, summaries/04_Server_WebUI_API.md, summaries/03_Core_API_활용.md, summaries/02_설치와_빠른시작.md, summaries/01_기초개념.md]
brief: RAG grounds LLM answers in retrieved external context before generation.
---

# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a method for improving LLM responses by searching external sources before generation. Instead of relying only on the model’s internal parameters, the system retrieves relevant documents or chunks from a knowledge source and inserts them into the prompt as context.

This concept is introduced in [[summaries/01_기초개념]], where it serves as the baseline architecture that [[concepts/lightrag]] extends. The installation and startup guide in [[summaries/02_설치와_빠른시작]] shows the practical requirements for running a RAG system: choosing an installation mode, configuring the LLM and embedding models, selecting storage, and keeping vector data compatible with the embedding setup.

## How RAG works

A standard RAG pipeline usually follows these steps:

1. Split documents into smaller chunks.
2. Convert each chunk into an embedding vector.
3. Convert the user question into an embedding vector.
4. Retrieve the most similar chunks using vector similarity.
5. Place the retrieved chunks into the LLM prompt.
6. Generate the final answer with the added context.

This workflow depends on several implementation choices, especially the embedding model, embedding dimension, and vector storage layer. In systems like LightRAG, these settings must be aligned before indexing begins.

## Why RAG is useful

RAG helps reduce hallucination and makes answers more grounded in source material. It is especially useful when the model must answer from private documents, internal knowledge bases, or rapidly changing information that may not be well represented in the model’s training data.

It also gives teams more control over the knowledge source: the model can stay general-purpose while retrieval supplies domain-specific context at query time.

## Strengths

- Simple to implement
- Works well for factual lookup
- Provides source-grounded responses
- Can be applied to many document collections
- Separates knowledge storage from model parameters

## Limitations

The basic RAG approach is effective, but it has important weaknesses:

- It mainly relies on chunk similarity, so it may miss broader document structure.
- It does not naturally model relationships between entities.
- It can struggle with questions that require cross-document reasoning.
- Retrieval quality depends heavily on chunking and embedding quality.
- Changing embedding models or dimensions can invalidate previously indexed vectors and require reindexing.

These limitations motivate graph-enhanced approaches such as [[concepts/lightrag]], which add entity and relationship extraction on top of standard retrieval.

## Operational considerations

A practical RAG setup requires more than retrieval logic alone. The startup guide in [[summaries/02_설치와_빠른시작]] highlights several configuration decisions that directly affect RAG quality and stability:

- **LLM provider and model** — determines generation quality and context handling
- **Embedding provider and model** — determines retrieval quality
- **Embedding dimension** — must match the stored vector data
- **Reranker** — can improve retrieval precision after initial search
- **Storage backend** — supports document, vector, and graph persistence

The guide also recommends using strong models for different phases of the pipeline:

- LLMs of **32B or larger** are recommended
- context length should be at least **32K**, ideally **64K**
- reasoning models are better avoided during indexing
- stronger models can be used at query time for better answers
- embedding models such as `BAAI/bge-m3` and `text-embedding-3-large` are commonly recommended
- rerankers such as `BAAI/bge-reranker-v2-m3` or Jina models can improve ranking quality

These choices reflect a key RAG principle: retrieval quality and generation quality both matter, and the system must be configured so that they work together.

## Related concepts

- [[concepts/lightrag]] — graph-enhanced retrieval that extends basic RAG
- [[concepts/knowledge-graph]] — structure for modeling entities and relationships
- [[concepts/vector-search]] — embedding-based similarity retrieval
- [[concepts/model-configuration]] — provider, model, and environment setup for RAG systems
- [[concepts/embeddings]] — vector representations used for retrieval
- [[concepts/storage-backends]] — persistence layers for documents and vectors
- [[summaries/01_기초개념]] — source summary covering the foundational explanation
- [[summaries/02_설치와_빠른시작]] — setup guide covering installation and required configuration

## Takeaway

RAG is the foundation of retrieval-based LLM systems: retrieve relevant external context first, then generate an answer using that context.

See also: [[summaries/03_Core_API_활용]]

See also: [[summaries/04_Server_WebUI_API]]

See also: [[summaries/06_고급기능과_실전팁]]

See also: [[summaries/README]]

See also: [[summaries/2_1_1 24년 하반기 신입사원_조인스_이재은_학술교재_0001]]

See also: [[summaries/2_1_1 24년 하반기 신입사원_조인스_이재은_학술교재_0002]]