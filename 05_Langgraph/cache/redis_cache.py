"""Redis semantic cache — cosine similarity threshold 0.8.

Stores (query_embedding, answer) pairs. On lookup, embeds the incoming
query and returns the cached answer when the best cosine similarity
exceeds the configured threshold.
"""

from __future__ import annotations

import base64
import uuid
from typing import Optional

import numpy as np
import redis.asyncio as aioredis
from langchain_openai import OpenAIEmbeddings

_INDEX_KEY = "sem_cache:ids"
_ENTRY_PFX = "sem_cache:e:"
_TTL = 86_400  # 24 h


class SemanticCache:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        threshold: float = 0.8,
    ) -> None:
        self._r = aioredis.from_url(redis_url, decode_responses=False)
        self._threshold = threshold
        self._embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    async def _embed(self, text: str) -> np.ndarray:
        vec = await self._embedder.aembed_query(text)
        return np.asarray(vec, dtype=np.float32)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        n = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / n) if n > 0 else 0.0

    @staticmethod
    def _pack(v: np.ndarray) -> bytes:
        return base64.b64encode(v.tobytes())

    @staticmethod
    def _unpack(raw: bytes) -> np.ndarray:
        return np.frombuffer(base64.b64decode(raw), dtype=np.float32)

    async def get(self, query: str) -> tuple[Optional[str], float]:
        """Return (cached_answer, similarity) on hit, (None, best_score) on miss."""
        try:
            qv = await self._embed(query)
            ids = await self._r.smembers(_INDEX_KEY)
            best_score, best_ans = 0.0, None

            for eid in ids:
                key = _ENTRY_PFX + eid.decode()
                raw_emb = await self._r.hget(key, "emb")
                raw_ans = await self._r.hget(key, "ans")
                if not raw_emb or not raw_ans:
                    await self._r.srem(_INDEX_KEY, eid)  # 죽은 eid를 발견 시 삭제
                    continue
                score = self._cosine(qv, self._unpack(raw_emb))
                if score > best_score:
                    best_score, best_ans = score, raw_ans.decode()

            if best_score >= self._threshold and best_ans:
                return best_ans, best_score
            return None, best_score
        except Exception:
            return None, 0.0  # Redis 불가 시 cache miss로 처리

    async def set(self, query: str, answer: str) -> None:
        """Store (query, answer) pair with its embedding in Redis."""
        try:
            qv = await self._embed(query)
            eid = str(uuid.uuid4())
            key = _ENTRY_PFX + eid
            await self._r.hset(key, mapping={
                "q": query.encode(),
                "ans": answer.encode(),
                "emb": self._pack(qv),
            })
            await self._r.expire(key, _TTL)
            await self._r.sadd(_INDEX_KEY, eid)
        except Exception:
            pass  # Redis 불가 시 캐싱 생략

    async def close(self) -> None:
        await self._r.aclose()
