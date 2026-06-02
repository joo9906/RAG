"""Ingest markdown documents into both Weaviate and Elasticsearch.

Usage:
    python ingest.py
    python ingest.py --reset   # drop and recreate indices
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langsmith import traceable

from vector_store.weaviate_store import WeaviateStore
from vector_store.elasticsearch_store import ElasticsearchStore

load_dotenv()

DOCS_DIR = Path(__file__).parent / "docs"
_embedder = OpenAIEmbeddings(model="text-embedding-3-small")


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_markdown(file_path: Path) -> list[dict]:
    """Split a markdown file by level-2 (##) headers into chunks."""
    text = file_path.read_text(encoding="utf-8")
    source = file_path.name

    # Split on ## headers
    sections = re.split(r"\n(?=## )", text)

    chunks: list[dict] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract header as section title
        lines = section.splitlines()
        header = lines[0].lstrip("#").strip() if lines else "intro"
        body = "\n".join(lines).strip()

        # Skip very short sections
        if len(body) < 80:
            continue

        chunks.append(
            {
                "content": body,
                "source": source,
                "section": header,
            }
        )

    return chunks


def load_all_docs() -> list[dict]:
    docs: list[dict] = []
    for md_file in sorted(DOCS_DIR.glob("*.md")):
        chunks = chunk_markdown(md_file)
        for i, chunk in enumerate(chunks):
            chunk["chunk_id"] = len(docs) + i
        docs.extend(chunks)
        print(f"  Loaded {len(chunks)} chunks from {md_file.name}")
    return docs


# ── Embedding ─────────────────────────────────────────────────────────────────

@traceable(run_type="chain", name="batch_embed_documents")
def add_embeddings(docs: list[dict]) -> list[dict]:
    texts = [d["content"] for d in docs]
    print(f"Computing {len(texts)} embeddings via OpenAI text-embedding-3-small…")
    embeddings = _embedder.embed_documents(texts)
    for doc, emb in zip(docs, embeddings):
        doc["embedding"] = emb
    return docs


# ── Main ──────────────────────────────────────────────────────────────────────

@traceable(run_type="chain", name="document_ingestion_pipeline")
def main(reset: bool = False) -> None:
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    es_url = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

    print("=" * 60)
    print("📄 Loading documents…")
    docs = load_all_docs()
    print(f"  Total chunks: {len(docs)}")

    print("\n🔢 Embedding…")
    docs = add_embeddings(docs)

    print("\n🔵 Weaviate indexing…")
    with WeaviateStore(weaviate_url) as wstore:
        wstore.create_collection(reset=reset)
        if reset or wstore.count() == 0:
            wstore.add_documents(docs)
        else:
            print(f"  Already has {wstore.count()} objects, skipping (use --reset to overwrite).")

    print("\n🟠 Elasticsearch indexing…")
    with ElasticsearchStore(es_url) as estore:
        estore.create_index(reset=reset)
        if reset or estore.count() == 0:
            estore.add_documents(docs)
        else:
            print(f"  Already has {estore.count()} docs, skipping (use --reset to overwrite).")

    print("\n✅ Ingestion complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest financial docs into vector stores.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate indices.")
    args = parser.parse_args()
    main(reset=args.reset)
