"""Weaviate vector store — semantic (dense) search using OpenAI embeddings."""

from __future__ import annotations

import os
from typing import Any

import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import MetadataQuery
from langsmith import traceable

COLLECTION_NAME = "FinancialDocument"


class WeaviateStore:
    def __init__(self, url: str = "http://localhost:8080") -> None:
        self.url = url
        self.client: weaviate.WeaviateClient | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    def connect(self) -> None:
        self.client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051,
        )

    def close(self) -> None:
        if self.client:
            self.client.close()

    def __enter__(self) -> "WeaviateStore":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── schema ────────────────────────────────────────────────────────────

    def create_collection(self, *, reset: bool = False) -> None:
        assert self.client is not None
        if reset and self.client.collections.exists(COLLECTION_NAME):
            self.client.collections.delete(COLLECTION_NAME)

        if not self.client.collections.exists(COLLECTION_NAME):
            self.client.collections.create(
                name=COLLECTION_NAME,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="section", data_type=DataType.TEXT),
                    Property(name="chunk_id", data_type=DataType.INT),
                ],
            )
            print(f"[Weaviate] Collection '{COLLECTION_NAME}' created.")
        else:
            print(f"[Weaviate] Collection '{COLLECTION_NAME}' already exists.")

    # ── indexing ──────────────────────────────────────────────────────────

    @traceable(run_type="tool", name="weaviate_index_documents")
    def add_documents(self, documents: list[dict]) -> None:
        """
        documents: list of {content, source, section, chunk_id, embedding}
        embedding must be a list[float] (OpenAI text-embedding-3-small = 1536 dims)
        """
        assert self.client is not None
        collection = self.client.collections.get(COLLECTION_NAME)

        with collection.batch.dynamic() as batch:
            for doc in documents:
                batch.add_object(
                    properties={
                        "content": doc["content"],
                        "source": doc["source"],
                        "section": doc.get("section", ""),
                        "chunk_id": doc["chunk_id"],
                    },
                    vector=doc["embedding"],
                )
        print(f"[Weaviate] Indexed {len(documents)} documents.")

    # ── retrieval ─────────────────────────────────────────────────────────

    @traceable(run_type="retriever", name="weaviate_semantic_search")
    def search(self, query_embedding: list[float], k: int = 3) -> list[dict]:
        """Dense (cosine similarity) search. Returns top-k results."""
        assert self.client is not None
        collection = self.client.collections.get(COLLECTION_NAME)

        response = collection.query.near_vector(
            near_vector=query_embedding,
            limit=k,
            return_metadata=MetadataQuery(distance=True),
            return_properties=["content", "source", "section", "chunk_id"],
        )

        results = []
        for obj in response.objects:
            results.append(
                {
                    "content": obj.properties["content"],
                    "source": obj.properties["source"],
                    "section": obj.properties.get("section", ""),
                    "chunk_id": obj.properties.get("chunk_id", 0),
                    "score": round(1 - (obj.metadata.distance or 0), 4),
                    "method": "weaviate_semantic",
                }
            )
        return results

    def count(self) -> int:
        assert self.client is not None
        collection = self.client.collections.get(COLLECTION_NAME)
        response = collection.aggregate.over_all(total_count=True)
        return response.total_count or 0
