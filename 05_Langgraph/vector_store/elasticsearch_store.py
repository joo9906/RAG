"""Elasticsearch vector store — BM25 keyword search (+ optional dense vector)."""

from __future__ import annotations

from typing import Any

from elasticsearch import Elasticsearch, NotFoundError
from langsmith import traceable

INDEX_NAME = "financial_documents"

_MAPPING = {
    "mappings": {
        "properties": {
            "content": {"type": "text", "analyzer": "standard"},
            "source": {"type": "keyword"},
            "section": {"type": "keyword"},
            "chunk_id": {"type": "integer"},
            "embedding": {
                "type": "dense_vector",
                "dims": 1536,
                "index": True,
                "similarity": "cosine",
            },
        }
    }
}


class ElasticsearchStore:
    def __init__(self, url: str = "http://localhost:9200") -> None:
        self.url = url
        self.es: Elasticsearch | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    def connect(self) -> None:
        self.es = Elasticsearch(self.url, verify_certs=False, ssl_show_warn=False)
        if not self.es.ping():
            raise ConnectionError(f"Cannot connect to Elasticsearch at {self.url}")

    def close(self) -> None:
        if self.es:
            self.es.close()

    def __enter__(self) -> "ElasticsearchStore":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ── schema ────────────────────────────────────────────────────────────

    def create_index(self, *, reset: bool = False) -> None:
        assert self.es is not None
        if reset:
            try:
                self.es.indices.delete(index=INDEX_NAME)
                print(f"[ES] Index '{INDEX_NAME}' deleted.")
            except NotFoundError:
                pass

        if not self.es.indices.exists(index=INDEX_NAME):
            self.es.indices.create(index=INDEX_NAME, body=_MAPPING)
            print(f"[ES] Index '{INDEX_NAME}' created.")
        else:
            print(f"[ES] Index '{INDEX_NAME}' already exists.")

    # ── indexing ──────────────────────────────────────────────────────────

    @traceable(run_type="tool", name="es_index_documents")
    def add_documents(self, documents: list[dict]) -> None:
        """
        documents: list of {content, source, section, chunk_id, embedding}
        """
        assert self.es is not None
        for doc in documents:
            self.es.index(
                index=INDEX_NAME,
                document={
                    "content": doc["content"],
                    "source": doc["source"],
                    "section": doc.get("section", ""),
                    "chunk_id": doc["chunk_id"],
                    "embedding": doc["embedding"],
                },
            )
        self.es.indices.refresh(index=INDEX_NAME)
        print(f"[ES] Indexed {len(documents)} documents.")

    # ── retrieval ─────────────────────────────────────────────────────────

    @traceable(run_type="retriever", name="es_bm25_search")
    def search_bm25(self, query: str, k: int = 3) -> list[dict]:
        """Classic BM25 full-text search."""
        assert self.es is not None
        response = self.es.search(
            index=INDEX_NAME,
            body={
                "size": k,
                "query": {"match": {"content": {"query": query, "operator": "or"}}},
            },
        )
        return self._parse_hits(response, method="es_bm25")

    @traceable(run_type="retriever", name="es_knn_search")
    def search_knn(self, query_embedding: list[float], k: int = 3) -> list[dict]:
        """Dense vector kNN search."""
        assert self.es is not None
        response = self.es.search(
            index=INDEX_NAME,
            body={
                "size": k,
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": k,
                    "num_candidates": k * 10,
                },
            },
        )
        return self._parse_hits(response, method="es_knn")

    def count(self) -> int:
        assert self.es is not None
        return self.es.count(index=INDEX_NAME)["count"]

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_hits(response: Any, method: str) -> list[dict]:
        results = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            results.append(
                {
                    "content": src["content"],
                    "source": src["source"],
                    "section": src.get("section", ""),
                    "chunk_id": src.get("chunk_id", 0),
                    "score": round(hit["_score"] or 0, 4),
                    "method": method,
                }
            )
        return results
