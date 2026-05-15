---
sources: [summaries/README.md, summaries/06_고급기능과_실전팁.md]
brief: RAG that retrieves and reasons over text plus images, tables, and documents.
---

# Multimodal RAG

Multimodal RAG is a retrieval-augmented generation approach that can ingest, index, and search across multiple content types, not just plain text. In the context of LightRAG, it extends the system to handle PDFs, Office files, images, tables, and equations while still supporting standard text retrieval and generation.

See also [[summaries/06_고급기능과_실전팁]] for the source discussion of the RAG-Anything integration.

## What makes it multimodal

A multimodal RAG system can work with:

- PDF documents
- DOC / DOCX files
- PPT / PPTX files
- XLS / XLSX spreadsheets
- Images
- Tables
- Mathematical formulas and equations

Instead of treating everything as plain text, the system can use modality-specific processors to extract and represent different content types more effectively.

## LightRAG + RAG-Anything

The source document describes integration between LightRAG and `RAG-Anything` as the practical path for multimodal support.

Typical workflow:

1. Create an existing LightRAG instance.
2. Connect it to `RAGAnything`.
3. Use the combined system to process and search multimodal documents.

This setup enables:

- modality-aware extraction
- multimodal knowledge graph construction
- joint search over text and non-text content

## Key capabilities

### 1. Document-type coverage
RAG-Anything expands LightRAG beyond text-only ingestion to support office documents, scanned-style materials, and visual content.

### 2. Specialized processors
The integration includes dedicated processors for images, tables, and formulas, which helps preserve structure and meaning during ingestion.

### 3. Multimodal knowledge graph
A knowledge graph can be built from both textual and multimodal content, allowing richer relationships to be represented and queried.

### 4. Unified retrieval
Users can search across text and multimodal content together, which improves coverage for real-world documents where important facts may live in figures, tables, or slides.

## Why it matters

Multimodal RAG is useful when important information is not contained in plain text alone. Many business, research, and technical documents rely on visual or structured elements, so a text-only RAG pipeline can miss critical context.

This concept is closely related to:

- [[concepts/knowledge-graph]] for structured relationship storage
- [[concepts/rag-configuration]] for system setup choices
- [[concepts/document-ingestion]] for processing varied file types
- [[concepts/retrieval-quality]] for improving answer completeness

## Practical considerations

- Multimodal ingestion usually requires more preprocessing than text-only ingestion.
- Specialized parsers and processors can improve quality but may add complexity.
- The quality of retrieval depends on how well tables, figures, and formulas are extracted and represented.
- Multimodal systems are especially valuable for analytics, manuals, reports, slide decks, and scientific documents.

## Summary

Multimodal RAG extends standard RAG pipelines to handle heterogeneous document content. In LightRAG, the `RAG-Anything` integration provides a practical way to process files such as PDFs, spreadsheets, slides, images, tables, and equations while preserving searchable structure and enabling a multimodal knowledge graph.

See also: [[summaries/README]]