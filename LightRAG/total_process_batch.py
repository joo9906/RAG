"""
total_process_batch.py — LightRAG + OpenAI Batch API 통합 파이프라인

  total_process.py 에서 실시간으로 호출하던 LLM 을
  OpenAI Batch API 로 대체하여 비용을 최대 50% 절감합니다.

  ┌─────────────────────────────────────────────────────────────────┐
  │  Batch API 적용 범위                                             │
  │  삽입  : 청크별 엔티티 추출 호출 → 배치 제출, 완료 후 재사용    │
  │  힐링  : 전략 C(LLM 관계 제안), D(소스 재추출) → 배치 제출     │
  │  비적용: 전략 A(삭제), B(임베딩), 쿼리 → 실시간 유지           │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  실시간 vs Batch API 비용 비교 (gpt-4o-mini 기준)               │
  │  입력  $0.150/1M  →  $0.075/1M  (50% 절감)                     │
  │  출력  $0.600/1M  →  $0.300/1M  (50% 절감)                     │
  │  처리  즉시       →  최대 24시간 (비동기)                        │
  └─────────────────────────────────────────────────────────────────┘

사용 예)
  # ① 전체 파이프라인 (배치 삽입 + 배치 힐링, 완료까지 폴링 대기)
  python total_process_batch.py

  # ② 배치 제출만 하고 즉시 종료 (나중에 결과 확인)
  python total_process_batch.py --submit-only

  # ③ 이전에 제출한 배치가 완료되면 이어서 처리
  python total_process_batch.py --resume insert   # 삽입 배치 재개
  python total_process_batch.py --resume heal     # 힐링 배치 재개

  # ④ 배치 상태 확인
  python total_process_batch.py --batch-status

  # ⑤ 힐링만 배치로 실행
  python total_process_batch.py --heal --batch-heal

  # ⑥ 폴링 간격 조정 (기본 60초)
  python total_process_batch.py --poll-interval 120

  # ⑦ 실시간 폴백 (배치 없이 즉시 실행 — total_process.py 동작과 동일)
  python total_process_batch.py --no-batch

  # ⑧ 단건 쿼리 (배치 무관, 항상 실시간)
  python total_process_batch.py -q 기넥신 누구한테 써?
"""

import os
import sys
import html
import copy
import asyncio
import argparse
import time
import json
import hashlib
import re
import shutil
import numpy as np
from datetime import datetime
from typing import Optional
import networkx as nx
from pyvis.network import Network
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, gpt_4o_complete
from lightrag.utils import EmbeddingFunc
from openai import AsyncOpenAI

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import tiktoken
    _enc = tiktoken.encoding_for_model("gpt-4o-mini")
except Exception:
    _enc = None

# LightRAG 내부 프롬프트 (배치 삽입 시 엔티티 추출 프롬프트 재현용)
try:
    from lightrag.prompt import PROMPTS as _LR_PROMPTS
    _TUPLE_DELIM      = _LR_PROMPTS["DEFAULT_TUPLE_DELIMITER"]       # "<|#|>"
    _COMPLETE_DELIM   = _LR_PROMPTS["DEFAULT_COMPLETION_DELIMITER"]  # "<|COMPLETE|>"
    _SYS_TMPL         = _LR_PROMPTS["entity_extraction_system_prompt"]
    _USR_TMPL         = _LR_PROMPTS["entity_extraction_user_prompt"]
    _EXAMPLES_LIST    = _LR_PROMPTS.get("entity_extraction_examples", [])
    _LR_PROMPTS_OK    = True
except Exception:
    _TUPLE_DELIM    = "<|#|>"
    _COMPLETE_DELIM = "<|COMPLETE|>"
    _LR_PROMPTS_OK  = False

# ==============================================================================
# CONFIG — 이 블록만 수정하면 전체 파이프라인이 맞춰 동작합니다
# ==============================================================================

# [1] 데이터 경로
ENV_JSON_PATH     = "../.env.json"
_BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR       = os.path.join(_BASE_DIR, "lightrag_before_chunk_test")
MD_DIR            = os.path.join(_BASE_DIR, "../chunked_docs")
QDRANT_URL        = "http://localhost:6333"
QDRANT_COLLECTION = "lightrag_before_chunk_test"

# [2] LLM 모델
LLM_MODEL   = "mini"   # "mini" = gpt-4o-mini | "4o" = gpt-4o
_COST_TABLE = {
    "mini": {
        "in": 0.000150, "out": 0.000600, "name": "gpt-4o-mini",
        "batch_in": 0.000075, "batch_out": 0.000300,       # 50% 할인
    },
    "4o": {
        "in": 0.002500, "out": 0.010000, "name": "gpt-4o",
        "batch_in": 0.001250, "batch_out": 0.005000,
    },
}

# [3] 임베딩 모델
EMB_MODEL      = "text-embedding-3-large"
EMB_DIM        = 2048
_EMB_COST      = 0.000130
EMB_MAX_TOKENS = 8192

# [4] 청크 설정
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150

# [5] 쿼리 캐시
CACHE_SIMILARITY_THRESHOLD = 0.85

# [6] 힐링 기본값
HEAL_PRUNE_MIN_DESC  = 10
HEAL_EMBED_THRESHOLD = 0.75
HEAL_EMBED_TOP_K     = 2
HEAL_LLM_LIMIT       = 50
HEAL_LLM_MIN_CONF    = 0.5
HEAL_RELINK_LIMIT    = 30
HEAL_LLM_BATCH_CANDS = 20

# [7] 배치 API 설정
BATCH_POLL_INTERVAL  = 60    # 폴링 간격 (초)
BATCH_STATE_FILE     = os.path.join(WORKING_DIR, "batch_state.json")
BATCH_COMPLETION_WIN = "24h" # OpenAI 배치 완료 창

# ==============================================================================
# CONFIG 끝
# ==============================================================================

with open(os.path.join(_BASE_DIR, ENV_JSON_PATH), "r", encoding="utf-8") as _f:
    _env = json.load(_f)
OPENAI_API_KEY = _env["openai_api_key"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["QDRANT_URL"]     = QDRANT_URL

os.makedirs(WORKING_DIR, exist_ok=True)

_oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ==============================================================================
# 토큰 카운터 & 비용
# ==============================================================================
_phase    = "insert"
_call_log = []
_file_log = []
_usage    = {
    "insert":       {"llm_in": 0, "llm_out": 0, "emb": 0},
    "query":        {"llm_in": 0, "llm_out": 0, "emb": 0},
    "heal":         {"llm_in": 0, "llm_out": 0, "emb": 0},
    "batch_insert": {"llm_in": 0, "llm_out": 0},   # 배치 삽입 집계
    "batch_heal":   {"llm_in": 0, "llm_out": 0},   # 배치 힐링 집계
}
_file_summary: list = []


def _count(text: str) -> int:
    if _enc and text:
        return len(_enc.encode(str(text)))
    return len(str(text)) // 4


def _llm_cost_rates(batch: bool = False):
    r = _COST_TABLE.get(LLM_MODEL, _COST_TABLE["mini"])
    if batch:
        return {"in": r["batch_in"], "out": r["batch_out"]}
    return {"in": r["in"], "out": r["out"]}


def _cost(in_tok, out_tok, emb_tok=0, batch=False):
    r = _llm_cost_rates(batch)
    return (in_tok  / 1000 * r["in"]
            + out_tok / 1000 * r["out"]
            + emb_tok / 1000 * _EMB_COST)


# ==============================================================================
# 임베딩 헬퍼
# ==============================================================================
async def _raw_embed(texts: list[str]) -> np.ndarray:
    kwargs = {"model": EMB_MODEL, "input": texts}
    if "large" in EMB_MODEL:
        kwargs["dimensions"] = EMB_DIM
    resp = await _oai_client.embeddings.create(**kwargs)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


# ==============================================================================
# [NEW] BatchJobManager — OpenAI Batch API 생명주기 관리
# ==============================================================================
class BatchJobManager:
    """
    OpenAI Batch API 를 통해 LLM 요청을 비동기로 제출·조회·수신합니다.

    사용 흐름:
        mgr = BatchJobManager()
        mgr.add_request("req_001", messages)
        batch_id = await mgr.submit("삽입 배치")
        results  = await mgr.poll_until_done(batch_id)
        # results: dict[custom_id → response_text]
    """

    def __init__(self):
        self._requests: list[dict] = []

    # ── 요청 추가 ────────────────────────────────────────────────
    def add_request(
        self,
        custom_id: str,
        messages: list[dict],
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> str:
        model_name = _COST_TABLE.get(LLM_MODEL, _COST_TABLE["mini"])["name"]
        self._requests.append({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model":       model_name,
                "messages":    messages,
                "temperature": temperature,
                "max_tokens":  max_tokens,
            },
        })
        return custom_id

    @property
    def count(self) -> int:
        return len(self._requests)

    def reset(self):
        self._requests.clear()

    # ── 배치 제출 ────────────────────────────────────────────────
    async def submit(self, description: str = "") -> str:
        """JSONL 업로드 후 배치 생성 → batch_id 반환."""
        if not self._requests:
            raise ValueError("제출할 요청이 없습니다.")

        jsonl = "\n".join(
            json.dumps(r, ensure_ascii=False) for r in self._requests
        ).encode("utf-8")

        print(f"  [Batch] 파일 업로드 중 ({len(self._requests)}개 요청) ...")
        file_resp = await _oai_client.files.create(
            file=("batch_requests.jsonl", jsonl),
            purpose="batch",
        )
        print(f"  [Batch] 파일 ID: {file_resp.id}")

        batch = await _oai_client.batches.create(
            input_file_id=file_resp.id,
            endpoint="/v1/chat/completions",
            completion_window=BATCH_COMPLETION_WIN,
            metadata={"description": description},
        )
        print(f"  [Batch] 배치 ID: {batch.id}  상태: {batch.status}")
        return batch.id

    # ── 상태 확인 ────────────────────────────────────────────────
    async def check_status(self, batch_id: str) -> dict:
        """배치 현재 상태 반환."""
        batch = await _oai_client.batches.retrieve(batch_id)
        rc = batch.request_counts
        return {
            "status":    batch.status,
            "total":     rc.total,
            "completed": rc.completed,
            "failed":    rc.failed,
            "output_file_id": batch.output_file_id,
        }

    # ── 완료까지 폴링 ────────────────────────────────────────────
    async def poll_until_done(
        self,
        batch_id: str,
        poll_interval: int = BATCH_POLL_INTERVAL,
    ) -> dict[str, str]:
        """
        배치 완료(completed)까지 poll_interval 초마다 상태를 확인합니다.
        완료되면 {custom_id: response_text} 딕셔너리를 반환합니다.
        """
        print(f"\n  [Batch] 폴링 시작  batch_id={batch_id}  간격={poll_interval}초")
        t_start = time.time()

        while True:
            info = await self.check_status(batch_id)
            elapsed = time.time() - t_start
            print(
                f"  [Batch] {info['status']:<14} "
                f"완료 {info['completed']}/{info['total']}  "
                f"실패 {info['failed']}  "
                f"경과 {elapsed/60:.1f}분"
            )

            if info["status"] == "completed":
                break
            if info["status"] in ("failed", "expired", "cancelled"):
                raise RuntimeError(
                    f"배치 {batch_id} 이 {info['status']} 상태로 종료되었습니다."
                )

            await asyncio.sleep(poll_interval)

        return await self._download_results(batch_id, info["output_file_id"])

    # ── 결과 다운로드 ────────────────────────────────────────────
    async def _download_results(
        self, batch_id: str, output_file_id: str
    ) -> dict[str, str]:
        print(f"  [Batch] 결과 다운로드 중 (file_id={output_file_id}) ...")
        content  = await _oai_client.files.content(output_file_id)
        results  = {}
        errors   = 0
        for line in content.text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            cid  = item.get("custom_id", "")
            resp = item.get("response", {})
            if resp.get("status_code") == 200:
                text = (
                    resp["body"]["choices"][0]["message"]["content"] or ""
                )
                results[cid] = text
                # 비용 집계
                usage = resp["body"].get("usage", {})
                _usage["batch_insert"]["llm_in"]  += usage.get("prompt_tokens", 0)
                _usage["batch_insert"]["llm_out"] += usage.get("completion_tokens", 0)
            else:
                errors += 1
        print(
            f"  [Batch] 수신 {len(results)}개  오류 {errors}개  "
            f"비용 예상 ${_cost(_usage['batch_insert']['llm_in'], _usage['batch_insert']['llm_out'], batch=True):.5f}"
        )
        return results

    # ── 상태 파일 저장/로드 ──────────────────────────────────────
    @staticmethod
    def save_state(batch_type: str, batch_id: str, extra: dict = None):
        state = {}
        if os.path.exists(BATCH_STATE_FILE):
            try:
                with open(BATCH_STATE_FILE, encoding="utf-8") as f:
                    state = json.load(f)
            except Exception:
                pass
        state[batch_type] = {
            "batch_id":     batch_id,
            "submitted_at": datetime.now().isoformat(),
            "status":       "submitted",
            **(extra or {}),
        }
        with open(BATCH_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"  [Batch] 상태 저장 → {BATCH_STATE_FILE}")

    @staticmethod
    def load_state(batch_type: str) -> Optional[dict]:
        if not os.path.exists(BATCH_STATE_FILE):
            return None
        with open(BATCH_STATE_FILE, encoding="utf-8") as f:
            state = json.load(f)
        return state.get(batch_type)

    @staticmethod
    def clear_state(batch_type: str):
        if not os.path.exists(BATCH_STATE_FILE):
            return
        with open(BATCH_STATE_FILE, encoding="utf-8") as f:
            state = json.load(f)
        state.pop(batch_type, None)
        with open(BATCH_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)


# ==============================================================================
# [NEW] BatchCachedLLM — 배치 결과를 캐시로 써서 LightRAG 삽입에 재주입
# ==============================================================================
class BatchCachedLLM:
    """
    entity extraction 호출: 배치 결과 캐시에서 즉시 반환 (API 비용 없음)
    gleaning / merge 호출 : 실시간 API 폴백 (비율 낮음)

    캐시 키 = hash(user_message) — 동일 청크 텍스트는 동일 해시
    """

    def __init__(self, cache: dict[str, str]):
        # {user_msg_hash → batch_response}
        self.cache  = cache
        self.hits   = 0
        self.misses = 0

    async def __call__(
        self,
        prompt,
        system_prompt=None,
        history_messages=[],
        **kwargs,
    ) -> str:
        # LightRAG 내부에서 _priority 등 내부 kwarg를 전달하는 경우 제거
        kwargs.pop("_priority", None)
        kwargs.pop("_timeout", None)
        kwargs.pop("_queue_timeout", None)
        key = hashlib.md5(prompt.encode("utf-8", errors="replace")).hexdigest()

        if key in self.cache:
            self.hits += 1
            result   = self.cache[key]
            in_tok   = _count(system_prompt or "") + _count(prompt)
            out_tok  = _count(result)
            # batch 단가로 집계
            _usage["batch_insert"]["llm_in"]  += in_tok
            _usage["batch_insert"]["llm_out"] += out_tok
            _file_log.append({
                "kind": "llm_batch", "in": in_tok, "out": out_tok, "sec": 0,
            })
            call_no = sum(1 for e in _file_log if e["kind"] in ("llm","llm_batch"))
            print(
                f"    LLM #{call_no:<3} [BATCH HIT] | "
                f"입력 {in_tok:>6,}tok | 출력 {out_tok:>5,}tok | "
                f"${_cost(in_tok, out_tok, batch=True):.5f}"
            )
            return result

        # 배치 캐시에 없으면 실시간 폴백
        self.misses += 1
        return await tracked_llm(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    def report(self):
        total = self.hits + self.misses
        ratio = self.hits / total * 100 if total else 0
        print(
            f"  [BatchCachedLLM] 캐시 히트 {self.hits}/{total} ({ratio:.0f}%)"
            f"  폴백(실시간) {self.misses}회"
        )


# ==============================================================================
# LightRAG 래퍼 (실시간 — 폴백용)
# ==============================================================================
async def tracked_llm(prompt, system_prompt=None, history_messages=[], **kwargs):
    # LightRAG 내부에서 _priority 등 내부 kwarg를 전달하는 경우 제거
    kwargs.pop("_priority", None)
    kwargs.pop("_timeout", None)
    kwargs.pop("_queue_timeout", None)
    in_tok = _count(system_prompt or "") + _count(prompt)
    for m in (history_messages or []):
        in_tok += _count(m.get("content", ""))

    t = time.time()
    if LLM_MODEL == "4o":
        result = await gpt_4o_complete(
            prompt, system_prompt=system_prompt,
            history_messages=history_messages, **kwargs)
    else:
        result = await gpt_4o_mini_complete(
            prompt, system_prompt=system_prompt,
            history_messages=history_messages, **kwargs)
    dur     = time.time() - t
    out_tok = _count(result)

    _usage[_phase]["llm_in"]  += in_tok
    _usage[_phase]["llm_out"] += out_tok

    if _phase == "insert":
        call_no = sum(1 for e in _file_log if e["kind"] in ("llm","llm_batch")) + 1
        _file_log.append({"kind": "llm", "in": in_tok, "out": out_tok, "sec": dur})
        print(
            f"    LLM #{call_no:<3} [LIVE]       | "
            f"입력 {in_tok:>6,}tok | 출력 {out_tok:>5,}tok | "
            f"${_cost(in_tok, out_tok):.5f} | {dur:.1f}초"
        )
    else:
        _call_log.append({
            "type": "llm", "in": in_tok, "out": out_tok, "sec": dur,
            "prompt_preview": prompt[:800],
            "result_preview": result[:300],
        })
    return result


async def tracked_embed(texts, **kwargs):
    t      = time.time()
    result = await _raw_embed(list(texts) if not isinstance(texts, list) else texts)
    dur    = time.time() - t
    tok    = sum(_count(x) for x in (texts if isinstance(texts, list) else [texts]))
    _usage[_phase]["emb"] += tok
    if _phase == "insert":
        emb_no = sum(1 for e in _file_log if e["kind"] == "emb") + 1
        _file_log.append({"kind": "emb", "in": tok, "out": 0, "sec": dur})
        print(f"    EMB #{emb_no:<3} | {tok:>6,}tok | {dur:.2f}초")
    else:
        _call_log.append({"type": "emb", "in": tok, "out": 0, "sec": dur})
    return result


# ==============================================================================
# [NEW] 배치 삽입 파이프라인
# ==============================================================================

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """tiktoken 기반 청크 분할 (LightRAG 와 동일 로직)."""
    if _enc is None:
        # tiktoken 없으면 문자 기반 근사
        step = max(1, chunk_size * 4 - overlap * 4)
        return [text[i:i + chunk_size * 4] for i in range(0, len(text), step)]
    tokens = _enc.encode(text)
    chunks = []
    step   = max(1, chunk_size - overlap)
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start:start + chunk_size]
        if not chunk_tokens:
            break
        chunks.append(_enc.decode(chunk_tokens))
    return chunks


def _build_entity_extraction_messages(
    chunk_content: str,
    language: str = "Korean",
) -> tuple[list[dict], str]:
    """
    LightRAG 와 동일한 entity extraction 프롬프트를 생성합니다.
    반환: (messages, user_msg_hash)
    """
    entity_types = [
        "Person", "Creature", "Organization", "Location", "Event",
        "Concept", "Method", "Content", "Data", "Artifact", "NaturalObject",
    ]

    if _LR_PROMPTS_OK:
        # LightRAG 실제 프롬프트 사용
        try:
            example_ctx = dict(
                tuple_delimiter=_TUPLE_DELIM,
                completion_delimiter=_COMPLETE_DELIM,
                language=language,
            )
            if isinstance(_EXAMPLES_LIST, list):
                examples_str = "\n\n".join(
                    ex.format(**example_ctx) for ex in _EXAMPLES_LIST
                )
            else:
                examples_str = str(_EXAMPLES_LIST).format(**example_ctx)

            ctx = dict(
                tuple_delimiter=_TUPLE_DELIM,
                completion_delimiter=_COMPLETE_DELIM,
                entity_types=",".join(entity_types),
                examples=examples_str,
                language=language,
            )
            system_msg = _SYS_TMPL.format(**ctx)
            user_msg   = _USR_TMPL.format(
                **{**ctx, "input_text": chunk_content}
            )
        except Exception as e:
            print(f"  [경고] LightRAG 프롬프트 포맷 실패: {e} — 폴백 사용")
            system_msg, user_msg = _fallback_extraction_messages(
                chunk_content, entity_types, language
            )
    else:
        system_msg, user_msg = _fallback_extraction_messages(
            chunk_content, entity_types, language
        )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg},
    ]
    # 캐시 키 = user 메시지 해시 (entity_extraction 호출 식별에 사용)
    user_hash = hashlib.md5(user_msg.encode("utf-8", errors="replace")).hexdigest()
    return messages, user_hash


def _fallback_extraction_messages(
    chunk_content: str, entity_types: list, language: str
) -> tuple[str, str]:
    """LightRAG 프롬프트를 로드하지 못한 경우 사용하는 폴백."""
    td, cd = _TUPLE_DELIM, _COMPLETE_DELIM
    system_msg = (
        f"You are a Knowledge Graph Specialist. "
        f"Extract entities and relationships from the text. "
        f"Entity types: {','.join(entity_types)}. "
        f"Output format: entity{td}name{td}type{td}description\n"
        f"For relations: relation{td}src{td}tgt{td}keywords{td}description\n"
        f"End with {cd}. Language: {language}."
    )
    user_msg = (
        f"Extract entities and relationships from the following text.\n\n"
        f"{chunk_content}\n\n{cd}"
    )
    return system_msg, user_msg


async def collect_and_submit_insert_batch(
    md_files: list[str],
    submit_only: bool = False,
    poll_interval: int = BATCH_POLL_INTERVAL,
) -> Optional[dict[str, str]]:
    """
    1) 모든 MD 파일을 청크로 분할
    2) 청크별 entity extraction 프롬프트를 Batch API 로 제출
    3) submit_only=True 면 제출 후 None 반환 (나중에 resume)
       False 면 완료까지 폴링하여 {user_hash: response} 반환
    """
    mgr       = BatchJobManager()
    hash_map  = {}   # custom_id → user_hash (결과 매핑용)
    req_count = 0

    print(f"\n  [Batch Insert] MD 파일 {len(md_files)}개 청크 분할 중 ...")
    for fname in md_files:
        with open(fname, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue

        chunks = _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        base   = os.path.basename(fname)
        print(f"    {base[:50]:<50}  → {len(chunks)}개 청크")

        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            messages, user_hash = _build_entity_extraction_messages(chunk)
            cid = f"{hashlib.md5(fname.encode()).hexdigest()[:8]}_{idx:04d}"
            mgr.add_request(cid, messages, max_tokens=4096, temperature=0)
            hash_map[cid] = user_hash
            req_count += 1

    print(f"\n  [Batch Insert] 총 {req_count}개 요청 준비 완료")
    est_in  = req_count * 800   # 청크당 평균 입력 토큰 추정
    est_out = req_count * 500   # 추출 결과 평균 추정
    rates   = _llm_cost_rates(batch=True)
    est_cost = (est_in / 1000 * rates["in"] + est_out / 1000 * rates["out"])
    print(f"  예상 비용 (배치): ${est_cost:.4f}  vs 실시간: ${est_cost*2:.4f}")

    batch_id = await mgr.submit(f"LightRAG 엔티티 추출 배치 — {len(md_files)}개 파일")
    BatchJobManager.save_state("insert", batch_id, {
        "req_count": req_count,
        "hash_map":  hash_map,
        "md_files":  [os.path.basename(f) for f in md_files],
    })

    if submit_only:
        print(f"\n  [Batch Insert] 제출 완료. 배치 ID: {batch_id}")
        print(f"  완료 후 재개하려면:")
        print(f"    python total_process_batch.py --resume insert")
        return None

    # 폴링 대기
    raw_results = await mgr.poll_until_done(batch_id, poll_interval)

    # custom_id → user_hash → response 로 재매핑
    cache: dict[str, str] = {}
    for cid, response in raw_results.items():
        uhash = hash_map.get(cid)
        if uhash:
            cache[uhash] = response

    print(f"  [Batch Insert] 캐시 구축 완료: {len(cache)}개 청크 응답")
    BatchJobManager.clear_state("insert")
    return cache


async def resume_insert_batch(
    poll_interval: int = BATCH_POLL_INTERVAL,
) -> Optional[dict[str, str]]:
    """이전에 제출한 삽입 배치를 재개하여 결과를 반환합니다."""
    state = BatchJobManager.load_state("insert")
    if not state:
        print("  [Batch Insert] 저장된 삽입 배치 없음.")
        return None

    batch_id = state["batch_id"]
    hash_map = state.get("hash_map", {})
    print(f"  [Batch Insert] 배치 재개: {batch_id}")
    print(f"  제출 시각: {state.get('submitted_at')}")
    print(f"  요청 수  : {state.get('req_count')}")

    mgr         = BatchJobManager()
    raw_results = await mgr.poll_until_done(batch_id, poll_interval)

    cache: dict[str, str] = {}
    for cid, response in raw_results.items():
        uhash = hash_map.get(cid)
        if uhash:
            cache[uhash] = response

    print(f"  [Batch Insert] 캐시 구축 완료: {len(cache)}개")
    BatchJobManager.clear_state("insert")
    return cache


def _build_rag(llm_func=None) -> LightRAG:
    """LightRAG 인스턴스 생성. llm_func 미지정 시 tracked_llm 사용."""
    lf = llm_func or tracked_llm
    print(f"  [청크] size={CHUNK_SIZE}tok  overlap={CHUNK_OVERLAP}tok")
    print(f"  [임베딩] {EMB_MODEL}  dim={EMB_DIM}")
    return LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=lf,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMB_DIM,
            max_token_size=EMB_MAX_TOKENS,
            func=tracked_embed,
            model_name=EMB_MODEL,
        ),
        chunk_token_size=CHUNK_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP,
        entity_extract_max_gleaning=2,
        force_llm_summary_on_merge=3,
        addon_params={"language": "Korean"},
        vector_storage="QdrantVectorDBStorage",
        vector_db_storage_cls_kwargs={"collection_name": QDRANT_COLLECTION},
    )


async def insert_documents(rag: LightRAG, batch_cache: Optional[dict] = None) -> None:
    """
    MD 파일 삽입.
    batch_cache 가 있으면 BatchCachedLLM 을 통해 배치 결과를 재주입하고,
    없으면 tracked_llm (실시간) 을 사용합니다.
    """
    global _file_log

    if batch_cache is not None:
        cached_llm = BatchCachedLLM(batch_cache)
        # LightRAG 의 llm_model_func 를 교체
        rag.llm_model_func = cached_llm
        print(f"  [Batch Insert] 캐시 {len(batch_cache)}개 응답으로 삽입 재실행")
    else:
        rag.llm_model_func = tracked_llm

    md_files = sorted(f for f in os.listdir(MD_DIR) if f.endswith(".md"))
    if not md_files:
        print(f"  MD 파일 없음: {MD_DIR}")
        return

    total = len(md_files)
    for idx, fname in enumerate(md_files, 1):
        _file_log = []
        fpath = os.path.join(MD_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            document = f.read().strip()
        if not document:
            print(f"  [{idx}/{total}] 건너뜀 (빈 파일): {fname}")
            continue

        print(f"\n  [{idx}/{total}] > {fname}  ({len(document):,} chars)")
        print(f"  {'─'*72}")
        t_file = time.time()
        await rag.ainsert(document)
        file_sec = time.time() - t_file

        llm_live  = [e for e in _file_log if e["kind"] == "llm"]
        llm_batch = [e for e in _file_log if e["kind"] == "llm_batch"]
        emb_calls = [e for e in _file_log if e["kind"] == "emb"]

        in_live  = sum(e["in"]  for e in llm_live)
        out_live = sum(e["out"] for e in llm_live)
        in_b     = sum(e["in"]  for e in llm_batch)
        out_b    = sum(e["out"] for e in llm_batch)
        emb_tok  = sum(e["in"]  for e in emb_calls)

        cost_live  = _cost(in_live,  out_live)
        cost_batch = _cost(in_b, out_b, batch=True)

        print(f"  {'─'*72}")
        print(
            f"  [{idx}/{total}] done  "
            f"실시간 LLM {len(llm_live)}회 ${cost_live:.5f}  |  "
            f"배치 캐시 {len(llm_batch)}회 ${cost_batch:.5f}  |  "
            f"EMB {emb_tok:,}tok  |  {file_sec:.1f}초"
        )
        _file_summary.append({
            "name": fname[:42],
            "calls_live":  len(llm_live),  "calls_batch": len(llm_batch),
            "in_live": in_live, "out_live": out_live,
            "in_b":    in_b,    "out_b":    out_b,
            "emb": emb_tok, "sec": file_sec,
        })

    if batch_cache is not None:
        rag.llm_model_func.report()


def print_insert_summary(elapsed: float) -> None:
    W = 110
    print(f"\n{'='*W}")
    print(f"{'[ 삽입 토큰 사용량 요약 ]':^{W}}")
    print(f"{'='*W}")
    print(f"  {'파일명':<43}| {'실시간':>5} | {'배치':>5} | "
          f"{'입력(실시간)':>12} | {'입력(배치)':>10} | {'비용($)':>10} | {'시간':>6}")
    print(f"  {'─'*105}")
    t_cl = t_cb = t_il = t_ol = t_ib = t_ob = t_emb = 0
    t_cost = t_sec = 0.0
    for s in _file_summary:
        c = _cost(s["in_live"], s["out_live"]) + _cost(s["in_b"], s["out_b"], batch=True)
        print(f"  {s['name']:<43}| {s['calls_live']:>5} | {s['calls_batch']:>5} | "
              f"{s['in_live']:>12,} | {s['in_b']:>10,} | {c:>10.5f} | {s['sec']:>5.1f}초")
        t_cl += s["calls_live"];  t_cb  += s["calls_batch"]
        t_il += s["in_live"];     t_ol  += s["out_live"]
        t_ib += s["in_b"];        t_ob  += s["out_b"]
        t_emb += s["emb"];        t_cost += c; t_sec += s["sec"]
    print(f"  {'─'*105}")
    print(f"  {'합계':<43}| {t_cl:>5} | {t_cb:>5} | "
          f"{t_il:>12,} | {t_ib:>10,} | {t_cost:>10.5f} | {t_sec:>5.1f}초")
    if t_il + t_ib > 0:
        batch_ratio = t_ib / (t_il + t_ib) * 100
        print(f"\n  배치 처리 비율: {batch_ratio:.0f}%  "
              f"(실시간 ${_cost(t_il,t_ol):.5f} + 배치 ${_cost(t_ib,t_ob,batch=True):.5f})")
    print(f"{'='*W}\n")


# ==============================================================================
# [NEW] 배치 힐링 파이프라인
# ==============================================================================

_HEAL_PRUNE_MIN_DESC = HEAL_PRUNE_MIN_DESC

# ── 공통 유틸 ────────────────────────────────────────────────────
def _get_isolated(G: nx.Graph) -> list[str]:
    return [n for n, d in G.degree() if d == 0]


def _node_text(attrs: dict) -> str:
    name = attrs.get("entity_name", "")
    desc = attrs.get("description", "")
    return f"{name}: {desc}".strip(": ") or "unknown"


async def _batch_embed_texts(texts: list[str], batch: int = 128) -> np.ndarray:
    parts = []
    for i in range(0, len(texts), batch):
        emb = await _raw_embed(texts[i:i+batch])
        parts.append(emb)
    return np.vstack(parts)


def _parse_json_array(raw: str) -> list[dict]:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    try:
        d = json.loads(raw)
        if isinstance(d, list):
            return d
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return []


# ── 전략 A: Prune ────────────────────────────────────────────────
def _strategy_prune(
    G: nx.Graph, min_desc_len: int = HEAL_PRUNE_MIN_DESC, dry_run: bool = False
) -> tuple[nx.Graph, int]:
    isolated  = _get_isolated(G)
    to_remove = []
    for nid in isolated:
        attrs = G.nodes[nid]
        etype = attrs.get("entity_type", "UNKNOWN")
        desc  = (attrs.get("description") or "").strip()
        if not desc:
            to_remove.append((nid, "설명없음"))
        elif etype == "UNKNOWN" and len(desc) < min_desc_len:
            to_remove.append((nid, f"UNKNOWN+짧음({len(desc)}자)"))

    W = 64
    print(f"\n  {'─'*W}")
    print(f"  [전략 A: Prune]  고립 {len(isolated)}개 중 삭제 후보 {len(to_remove)}개")
    for nid, reason in to_remove[:15]:
        print(f"    삭제: {G.nodes[nid].get('entity_name', nid)[:42]}  ({reason})")
    if dry_run:
        return G, 0
    G2 = copy.deepcopy(G)
    for nid, _ in to_remove:
        G2.remove_node(nid)
    print(f"  완료: {len(to_remove)}개 삭제")
    return G2, len(to_remove)


# ── 전략 B: Embed (배치 API 불필요, 임베딩 API 사용) ─────────────
async def _strategy_embed(
    G: nx.Graph,
    threshold: float = HEAL_EMBED_THRESHOLD,
    top_k: int = HEAL_EMBED_TOP_K,
    dry_run: bool = False,
) -> tuple[nx.Graph, int]:
    isolated  = _get_isolated(G)
    connected = [n for n in G.nodes() if n not in set(isolated)]
    W = 64
    print(f"\n  {'─'*W}")
    if not isolated or not connected:
        print("  [전략 B: Embed]  스킵")
        return G, 0
    print(f"  [전략 B: Embed]  고립 {len(isolated)}개 × 연결 {len(connected)}개")

    iso_embs  = await _batch_embed_texts([_node_text(G.nodes[n]) for n in isolated])
    conn_embs = await _batch_embed_texts([_node_text(G.nodes[n]) for n in connected])

    G2 = copy.deepcopy(G)
    added = 0
    for i, iso_id in enumerate(isolated):
        sims = [(connected[j], _cosine_sim(iso_embs[i], conn_embs[j]))
                for j in range(len(connected))]
        sims.sort(key=lambda x: -x[1])
        for conn_id, sim in [(c, s) for c, s in sims if s >= threshold][:top_k]:
            print(f"    {G.nodes[iso_id].get('entity_name',iso_id)[:30]:<30}"
                  f" ─({sim:.3f})─> {G.nodes[conn_id].get('entity_name',conn_id)[:30]}")
            if not dry_run:
                G2.add_edge(iso_id, conn_id,
                            relation_name="semantic_similarity",
                            keywords="semantic_similarity",
                            description=f"임베딩 유사도 (cosine={sim:.4f})",
                            weight=round(float(sim), 4),
                            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                added += 1
    print(f"  완료: 엣지 {added}개 추가")
    return G2, added


# ── 전략 C: LLM (배치 API 버전) ─────────────────────────────────
_HEAL_C_SYSTEM = """\
당신은 의료·제약 지식그래프 전문가입니다.
[고립 노드] 목록의 각 엔티티와 [후보 노드] 목록 중 실제로 관계가 있을 법한 후보를 골라 JSON으로 반환하세요.

출력 형식:
[
  {
    "isolated_id": "고립 노드 node_id",
    "candidate_id": "후보 node_id",
    "relation": "관계 종류 (한국어, 15자 이내)",
    "description": "관계 설명 (50자 이내)",
    "confidence": 0.0~1.0
  }
]
규칙: confidence<0.5 제외, 반드시 JSON 배열만 출력."""

_HEAL_D_SYSTEM = """\
당신은 지식그래프 전문가입니다.
[텍스트]에서 [타겟 엔티티]와 관련된 다른 엔티티들과의 관계를 추출하세요.

출력 형식:
[
  {
    "other_entity": "관련 엔티티 이름",
    "relation": "관계 종류 (한국어, 15자 이내)",
    "description": "관계 설명 (50자 이내)",
    "confidence": 0.0~1.0
  }
]
규칙: confidence<0.6 제외, 반드시 JSON 배열만 출력."""


async def _strategy_llm_batch(
    G: nx.Graph,
    llm_limit: int = HEAL_LLM_LIMIT,
    min_confidence: float = HEAL_LLM_MIN_CONF,
    dry_run: bool = False,
    submit_only: bool = False,
    poll_interval: int = BATCH_POLL_INTERVAL,
) -> tuple[nx.Graph, int]:
    """전략 C — Batch API 버전: 모든 배치 요청을 한 번에 제출."""
    isolated  = _get_isolated(G)
    connected = [n for n in G.nodes() if n not in set(isolated)]
    W = 64
    print(f"\n  {'─'*W}")
    if not isolated:
        print("  [전략 C: LLM Batch]  고립 노드 없음 — 스킵")
        return G, 0

    top_cands = sorted(connected, key=lambda n: G.degree(n), reverse=True)[:HEAL_LLM_BATCH_CANDS]
    target    = isolated[:llm_limit]
    CHUNK     = 10

    def _fmt(nid, attrs):
        name  = attrs.get("entity_name", nid)
        etype = attrs.get("entity_type", "?")
        desc  = (attrs.get("description") or "")[:80].replace("\n", " ")
        return f"  id={nid}  name={name}  type={etype}  desc={desc}"

    cand_block = "\n".join(_fmt(n, G.nodes[n]) for n in top_cands)

    mgr      = BatchJobManager()
    chunk_ids: list[tuple[str, list[str]]] = []  # (batch_req_id, [iso_ids])

    for start in range(0, len(target), CHUNK):
        chunk    = target[start:start+CHUNK]
        iso_block = "\n".join(_fmt(n, G.nodes[n]) for n in chunk)
        user_msg  = f"[고립 노드]\n{iso_block}\n\n[후보 노드]\n{cand_block}"
        cid       = f"heal_c_{start:04d}"
        mgr.add_request(cid, [
            {"role": "system", "content": _HEAL_C_SYSTEM},
            {"role": "user",   "content": user_msg},
        ], max_tokens=2048, temperature=0.2)
        chunk_ids.append((cid, chunk))

    print(f"  [전략 C: LLM Batch]  {mgr.count}개 요청 제출 중 ...")
    batch_id = await mgr.submit("힐링 전략 C — LLM 관계 제안")
    BatchJobManager.save_state("heal_c", batch_id)

    if submit_only:
        print(f"  제출 완료: {batch_id}")
        return G, 0

    raw = await mgr.poll_until_done(batch_id, poll_interval)
    BatchJobManager.clear_state("heal_c")

    # 결과 적용
    G2 = copy.deepcopy(G)
    added = 0
    for req_id, iso_ids in chunk_ids:
        response   = raw.get(req_id, "[]")
        suggestions = _parse_json_array(response)
        valid = [
            s for s in suggestions
            if isinstance(s, dict)
            and s.get("confidence", 0) >= min_confidence
            and s.get("isolated_id")  in G.nodes
            and s.get("candidate_id") in G.nodes
        ]
        for sug in valid:
            iso_n  = G.nodes[sug["isolated_id"]].get("entity_name",  sug["isolated_id"])
            cand_n = G.nodes[sug["candidate_id"]].get("entity_name", sug["candidate_id"])
            print(f"    [{sug['confidence']:.2f}] {iso_n[:28]:<28} ─[{sug['relation']}]─> {cand_n[:28]}")
            _usage["batch_heal"]["llm_in"]  += _count(_HEAL_C_SYSTEM)
            _usage["batch_heal"]["llm_out"] += _count(response)
            if not dry_run:
                G2.add_edge(
                    sug["isolated_id"], sug["candidate_id"],
                    relation_name=sug.get("relation", "관련"),
                    keywords=sug.get("relation", "관련"),
                    description=sug.get("description", "LLM 배치 제안"),
                    weight=round(float(sug.get("confidence", 0.5)), 4),
                    created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                added += 1

    healed = len({s["isolated_id"] for rr in [_parse_json_array(raw.get(r,"[]")) for r,_ in chunk_ids] for s in rr if s.get("confidence",0)>=min_confidence})
    print(f"  완료: 엣지 {added}개 추가 (고립 노드 ~{healed}개 해소)")
    return G2, added


async def _strategy_relink_batch(
    G: nx.Graph,
    relink_limit: int = HEAL_RELINK_LIMIT,
    dry_run: bool = False,
    submit_only: bool = False,
    poll_interval: int = BATCH_POLL_INTERVAL,
) -> tuple[nx.Graph, int]:
    """전략 D — Batch API 버전: 소스 청크 재추출을 배치로 제출."""
    W = 64
    print(f"\n  {'─'*W}")

    kv_files = [
        os.path.join(WORKING_DIR, f)
        for f in os.listdir(WORKING_DIR)
        if "chunk" in f.lower() and f.endswith(".json")
    ]
    if not kv_files:
        print("  [전략 D: Re-link Batch]  청크 KV 없음 — 스킵")
        return G, 0

    SEP = "<SEP>"
    chunk_map: dict[str, str] = {}
    for kp in kv_files:
        try:
            with open(kp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for cid, cval in data.items():
                    text = (cval.get("content") or cval.get("text") or ""
                            if isinstance(cval, dict) else str(cval))
                    if text:
                        chunk_map[cid] = text
        except Exception:
            pass

    isolated = _get_isolated(G)
    print(f"  [전략 D: Re-link Batch]  고립 {len(isolated)}개 / 청크 {len(chunk_map)}개")

    name_to_id: dict[str, str] = {}
    for nid, attrs in G.nodes(data=True):
        name = (attrs.get("entity_name") or nid).strip().lower()
        name_to_id[name] = nid

    mgr       = BatchJobManager()
    node_list = []   # (nid, req_id) — 응답 매핑용

    for nid in isolated[:relink_limit]:
        attrs   = G.nodes[nid]
        name    = attrs.get("entity_name", nid)
        src_ids = [s.strip() for s in (attrs.get("source_id") or "").split(SEP)
                   if s.strip() in chunk_map]
        if not src_ids:
            continue
        chunk_text = "\n---\n".join(chunk_map[s] for s in src_ids[:3])[:3000]
        user_msg   = f"[타겟 엔티티]\n{name}\n\n[텍스트]\n{chunk_text}"
        req_id     = f"heal_d_{hashlib.md5(nid.encode()).hexdigest()[:8]}"
        mgr.add_request(req_id, [
            {"role": "system", "content": _HEAL_D_SYSTEM},
            {"role": "user",   "content": user_msg},
        ], max_tokens=1024, temperature=0.2)
        node_list.append((nid, req_id, name))

    if not node_list:
        print("  [전략 D] 소스 청크 있는 고립 노드 없음 — 스킵")
        return G, 0

    print(f"  [전략 D: Re-link Batch]  {mgr.count}개 요청 제출 중 ...")
    batch_id = await mgr.submit("힐링 전략 D — 소스 청크 재추출")
    BatchJobManager.save_state("heal_d", batch_id)

    if submit_only:
        print(f"  제출 완료: {batch_id}")
        return G, 0

    raw = await mgr.poll_until_done(batch_id, poll_interval)
    BatchJobManager.clear_state("heal_d")

    G2 = copy.deepcopy(G)
    added = 0
    for nid, req_id, name in node_list:
        response    = raw.get(req_id, "[]")
        suggestions = _parse_json_array(response)
        valid = [s for s in suggestions
                 if isinstance(s, dict) and s.get("confidence", 0) >= 0.6]
        _usage["batch_heal"]["llm_in"]  += _count(_HEAL_D_SYSTEM)
        _usage["batch_heal"]["llm_out"] += _count(response)
        for sug in valid:
            other_name = (sug.get("other_entity") or "").strip().lower()
            target_id  = name_to_id.get(other_name)
            if not target_id or target_id == nid:
                continue
            other_disp = G.nodes.get(target_id, {}).get("entity_name", target_id)
            conf       = float(sug.get("confidence", 0.6))
            rel        = sug.get("relation", "관련")
            print(f"    + {name[:25]} ─[{rel}]─> {other_disp[:25]}  (conf={conf:.2f})")
            if not dry_run:
                G2.add_edge(nid, target_id,
                            relation_name=rel, keywords=rel,
                            description=sug.get("description", ""),
                            weight=round(conf, 4),
                            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                added += 1

    print(f"  완료: 엣지 {added}개 추가")
    return G2, added


def _find_graphml() -> str:
    files = [os.path.join(WORKING_DIR, f)
             for f in os.listdir(WORKING_DIR) if f.endswith(".graphml")]
    if not files:
        raise FileNotFoundError(f"GraphML 없음: {WORKING_DIR}")
    return max(files, key=os.path.getmtime)


async def run_heal_batch(
    do_prune: bool = True,
    do_embed: bool = True,
    do_llm: bool   = True,
    do_relink: bool = False,
    dry_run: bool   = False,
    submit_only: bool = False,
    poll_interval: int = BATCH_POLL_INTERVAL,
    embed_threshold: float = HEAL_EMBED_THRESHOLD,
    embed_top_k: int = HEAL_EMBED_TOP_K,
    llm_limit: int  = HEAL_LLM_LIMIT,
    llm_min_confidence: float = HEAL_LLM_MIN_CONF,
    relink_limit: int = HEAL_RELINK_LIMIT,
    prune_min_desc: int = HEAL_PRUNE_MIN_DESC,
    isolated_detail: bool = False,
) -> None:
    global _phase
    _phase = "heal"

    try:
        gpath = _find_graphml()
    except FileNotFoundError as e:
        print(f"\n  오류: {e}")
        return

    G = nx.read_graphml(gpath)
    print(f"\n  GraphML: {os.path.basename(gpath)}")
    print_graph_stats(G, "힐링 전")

    if isolated_detail:
        print_isolated_detail(G)

    n_iso_before = sum(1 for _, d in G.degree() if d == 0)
    if n_iso_before == 0:
        print("  고립 노드 없음 — 힐링 스킵")
        return

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak  = gpath + f".bak_{ts}"
    shutil.copy2(gpath, bak)
    print(f"  백업: {os.path.basename(bak)}\n")

    total_changed = 0

    if do_prune:
        G, n = _strategy_prune(G, prune_min_desc, dry_run)
        total_changed += n
    if do_embed:
        G, n = await _strategy_embed(G, embed_threshold, embed_top_k, dry_run)
        total_changed += n
    if do_llm:
        G, n = await _strategy_llm_batch(
            G, llm_limit, llm_min_confidence, dry_run, submit_only, poll_interval)
        total_changed += n
    if do_relink:
        G, n = await _strategy_relink_batch(
            G, relink_limit, dry_run, submit_only, poll_interval)
        total_changed += n

    if not dry_run and not submit_only and total_changed > 0:
        nx.write_graphml(G, gpath)
        print(f"\n  그래프 저장: {os.path.basename(gpath)}")
        print_graph_stats(G, "힐링 후")
        n_iso_after = sum(1 for _, d in G.degree() if d == 0)
        resolved    = n_iso_before - n_iso_after
        if n_iso_before:
            print(f"  [힐링 요약] 고립 노드: {n_iso_before} → {n_iso_after}"
                  f"  (해소 {resolved}개, {resolved/n_iso_before*100:.1f}%)")
    elif dry_run:
        print("\n  [dry_run] 저장 없음")
    elif submit_only:
        print("\n  [submit_only] 배치 제출 완료. --resume heal 로 재개하세요.")


# ==============================================================================
# 그래프 통계 & 시각화
# ==============================================================================
def print_graph_stats(G: nx.Graph = None, label: str = "") -> None:
    if G is None:
        try:
            gpath = _find_graphml()
            G = nx.read_graphml(gpath)
        except FileNotFoundError as e:
            print(f"  {e}")
            return
    n_nodes  = G.number_of_nodes()
    n_edges  = G.number_of_edges()
    degrees  = [d for _, d in G.degree()]
    avg_deg  = sum(degrees) / len(degrees) if degrees else 0.0
    max_deg  = max(degrees) if degrees else 0
    isolated = sum(1 for d in degrees if d == 0)
    comps    = (list(nx.weakly_connected_components(G))
                if G.is_directed()
                else list(nx.connected_components(G)))
    type_dist: dict[str, int] = {}
    for _, attrs in G.nodes(data=True):
        etype = attrs.get("entity_type", "UNKNOWN")
        type_dist[etype] = type_dist.get(etype, 0) + 1
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:10]
    W = 64
    print(f"\n{'='*W}")
    print(f"  그래프 통계{f'  [{label}]' if label else ''}")
    print(f"{'='*W}")
    print(f"  노드: {n_nodes:,}개  |  엣지: {n_edges:,}개")
    iso_pct = f" ({isolated/n_nodes*100:.1f}%)" if n_nodes else ""
    print(f"  컴포넌트: {len(comps)}개  |  고립 노드: {isolated}개{iso_pct}")
    print(f"  평균 연결도: {avg_deg:.2f}  |  최대 연결도: {max_deg}")
    print()
    print("  [타입 분포]")
    for etype, cnt in sorted(type_dist.items(), key=lambda x: -x[1])[:15]:
        bar = "#" * min(30, cnt)
        print(f"    {etype:<22} {cnt:>5}개  {bar}")
    print()
    print("  [연결도 상위 10]")
    for nid, deg in top_nodes:
        attrs = G.nodes[nid]
        name  = attrs.get("entity_name", nid)[:35]
        etype = attrs.get("entity_type", "?")
        print(f"    {name:<36} [{etype:<12}] 연결={deg}")
    print(f"{'='*W}\n")


def print_isolated_detail(G: nx.Graph, limit: int = 50) -> None:
    isolated = [n for n, d in G.degree() if d == 0]
    print(f"\n  고립 노드 (총 {len(isolated)}개, 최대 {limit}개)")
    print(f"  {'노드 ID':<36} {'타입':<14} {'길이':>6}  미리보기")
    print(f"  {'-'*80}")
    for nid in isolated[:limit]:
        attrs = G.nodes[nid]
        etype = attrs.get("entity_type", "?")
        desc  = attrs.get("description", "")
        print(f"  {nid[:36]:<36} {etype:<14} {len(desc):>6}자  {desc[:45].replace(chr(10),' ')}")
    if len(isolated) > limit:
        print(f"  ... 외 {len(isolated)-limit}개")
    print()


def visualize_graph(graphml_path=None, output_html=None, max_nodes=1000):
    if output_html is None:
        output_html = os.path.join(WORKING_DIR, "knowledge_graph.html")
    if graphml_path is None:
        try:
            graphml_path = _find_graphml()
        except FileNotFoundError:
            print("  GraphML 없음 — 시각화 스킵")
            return
    print(f"  [시각화] {os.path.basename(graphml_path)}")
    G = nx.read_graphml(graphml_path)
    print(f"  노드: {G.number_of_nodes():,}  엣지: {G.number_of_edges():,}")
    if G.number_of_nodes() > max_nodes:
        top = sorted(G.degree, key=lambda x: x[1], reverse=True)[:max_nodes]
        G   = G.subgraph([n for n, _ in top]).copy()
    net = Network(height="900px", width="100%", bgcolor="#0f0f1a",
                  font_color="#e0e0e0", directed=G.is_directed(), notebook=False)
    net.set_options("""{"physics":{"barnesHut":{"gravitationalConstant":-8000,
      "springLength":150,"springConstant":0.04,"damping":0.09},"minVelocity":0.75},
      "edges":{"smooth":{"type":"dynamic"},"color":{"inherit":"both"},"width":1.5},
      "nodes":{"shape":"dot","font":{"size":13,"color":"#ffffff"},"borderWidth":2},
      "interaction":{"hover":true,"navigationButtons":true,"keyboard":true}}""")
    type_colors: dict = {}
    palette = ["#4fc3f7","#81c784","#ffb74d","#e57373","#ba68c8",
               "#4db6ac","#f06292","#aed581","#ff8a65","#90a4ae"]
    def get_color(e):
        if e not in type_colors:
            type_colors[e] = palette[len(type_colors) % len(palette)]
        return type_colors[e]
    for nid, attrs in G.nodes(data=True):
        label = attrs.get("entity_name", str(nid))
        etype = attrs.get("entity_type", "UNKNOWN")
        deg   = G.degree(nid)
        net.add_node(str(nid), label=str(label)[:40],
                     title=f"<b>[{etype}]</b> {label}<br/>{attrs.get('description','')[:200]}",
                     color={"background": get_color(etype), "border": "#ffffff"},
                     size=max(10, min(40, deg * 3)),
                     font={"color": "#ffffff", "size": 13})
    for src, dst, attrs in G.edges(data=True):
        rel = attrs.get("relation_name", attrs.get("relation", ""))
        try:    w = float(attrs.get("weight", 1.0))
        except: w = 1.0
        net.add_edge(str(src), str(dst),
                     title=f"{rel}<br/>{attrs.get('keywords','')}",
                     label=str(rel)[:25] if rel else "",
                     width=max(1.0, min(5.0, w * 2)),
                     color={"opacity": 0.7})
    legend = ("<div style='position:fixed;top:10px;right:10px;background:#1a1a2e;"
              "padding:12px;border-radius:8px;font-family:sans-serif;font-size:12px;"
              "color:#fff;z-index:999;'><b>Entity Types</b><br/>")
    for etype, color in type_colors.items():
        legend += (f"<span style='display:inline-block;width:12px;height:12px;"
                   f"background:{color};border-radius:50%;margin-right:5px;'></span>"
                   f"{html.escape(etype)}<br/>")
    legend += "</div>"
    net.save_graph(output_html)
    with open(output_html, "r", encoding="utf-8") as f:
        content = f.read()
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(content.replace("</body>", f"{legend}</body>"))
    print(f"  시각화 완료 → {os.path.abspath(output_html)}")


# ==============================================================================
# 쿼리 파이프라인 (실시간 — 배치 불필요)
# ==============================================================================
class QueryCache:
    def __init__(self, cache_path: str):
        self.path = cache_path
        self.data: dict = {}
        self._load()
    def _load(self):
        if not os.path.exists(self.path):
            self.data = {}; return
        with open(self.path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        sample_dim = next((len(e["embedding"]) for e in loaded.values()
                           if e.get("embedding")), None)
        if sample_dim and sample_dim != EMB_DIM:
            bak = self.path + f".dim{sample_dim}.bak"
            shutil.copy2(self.path, bak)
            print(f"  [캐시] 차원 불일치 — 초기화 (백업: {os.path.basename(bak)})")
            self.data = {}; self.save()
        else:
            self.data = loaded
            print(f"  [캐시] {len(self.data)}건 로드됨")
    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    @staticmethod
    def _cosine_sim(a, b):
        a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        if a.shape != b.shape: return 0.0
        n = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / n) if n else 0.0
    def search(self, query_emb, threshold=None):
        if threshold is None: threshold = CACHE_SIMILARITY_THRESHOLD
        best_sim, best_entry = 0.0, None
        for e in self.data.values():
            emb = e.get("embedding")
            if not emb or len(emb) != len(query_emb): continue
            s = self._cosine_sim(query_emb, emb)
            if s > best_sim: best_sim, best_entry = s, e
        return (best_entry, best_sim) if best_sim >= threshold else (None, best_sim)
    def store(self, query, embedding, answer, mode, llm_in, llm_out, emb_tok, cost, sec):
        cid = hashlib.md5(query.encode()).hexdigest()[:12]
        self.data[cid] = {"query": query, "embedding": embedding,
            "emb_dim": len(embedding), "answer": answer, "mode": mode,
            "llm_in_tok": llm_in, "llm_out_tok": llm_out, "emb_tok": emb_tok,
            "cost": round(cost, 6), "sec": round(sec, 2),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        self.save()
    def summary(self):
        n = len(self.data)
        if n == 0: return "캐시 비어 있음"
        return f"캐시 {n}건 | 누적 비용 ${sum(e.get('cost',0) for e in self.data.values()):.5f}"


_query_cache: QueryCache | None = None
DEFAULT_QUERIES = [
    "기넥신 누구한테 영업할까?",
    "외과에 어떤 약을 추천할까",
    "류마티스 관절염에 도움이 되는 약",
    "기넥신을 누구한테 쓰면 안돼?",
]
MODES = ["naive", "local", "global", "hybrid"]


async def _embed_query(q: str) -> list:
    return (await _raw_embed([q]))[0].tolist()


def _extract_sources(call_log):
    sources = []
    for entry in call_log:
        if entry.get("type") != "llm": continue
        p = entry.get("prompt_preview", "")
        names = re.findall(r'entity[_\s]*name["\s:]*([^"\n,]+)', p, re.IGNORECASE)
        names += re.findall(r'"([A-Z][A-Za-z0-9\s\-]{2,30})"', p)
        for n in names:
            n = n.strip()
            if n and n not in sources and len(n) > 1: sources.append(n)
    return sources[:15]


async def _run_one_query(rag, query, mode="hybrid", silent=False):
    global _call_log, _query_cache
    _call_log = []
    u_before  = {k: _usage["query"][k] for k in _usage["query"]}
    if not silent:
        print(f"\n{'='*60}\n[쿼리] {query}  (mode={mode})\n{'='*60}")
    t0        = time.time()
    query_emb = await _embed_query(query)
    emb_sec   = time.time() - t0
    emb_tok   = _count(query)
    _usage["query"]["emb"] += emb_tok
    if not silent:
        print(f"  [임베딩] {emb_tok}tok | {emb_sec:.2f}초")
    if _query_cache is not None and mode == "hybrid":
        cached, sim = _query_cache.search(query_emb)
        if cached is not None:
            if not silent:
                print(f"\n  ** 캐시 히트 ** (유사도: {sim:.4f})\n{cached['answer']}")
            return {"query": query, "mode": mode, "answer": cached["answer"],
                    "cache_hit": True, "similarity": sim, "sec": emb_sec,
                    "llm_in": 0, "llm_out": 0, "emb_tok": emb_tok, "cost": 0, "sources": []}
        if not silent and sim > 0:
            print(f"  [캐시] 유사도 {sim:.4f} (임계값 {CACHE_SIMILARITY_THRESHOLD} 미달)")
    t_total   = time.time()
    result    = await rag.aquery(query, param=QueryParam(mode=mode, enable_rerank=False))
    total_sec = time.time() - t_total
    if not silent: print(result)
    emb_calls = [c for c in _call_log if c["type"] == "emb"]
    llm_calls = [c for c in _call_log if c["type"] == "llm"]
    gen_llm   = llm_calls[-1] if llm_calls else None
    t_emb = sum(c["sec"] for c in emb_calls)
    t_gen = gen_llm["sec"] if gen_llm else 0
    q_in  = _usage["query"]["llm_in"]  - u_before["llm_in"]
    q_out = _usage["query"]["llm_out"] - u_before["llm_out"]
    q_emb = _usage["query"]["emb"]     - u_before["emb"]
    q_cost = _cost(q_in, q_out, q_emb)
    sources = _extract_sources(_call_log)
    if not silent:
        print(f"\n  ┌─ [타이밍] 임베딩 {t_emb:.2f}초  답변생성 {t_gen:.2f}초  합계 {total_sec:.2f}초")
        print(f"  └─ 비용: ${q_cost:.5f}  (= ₩{q_cost*1380:,.0f})")
        if sources: print(f"  [출처] {', '.join(sources[:8])}")
    if _query_cache is not None and mode == "hybrid":
        _query_cache.store(query=query, embedding=query_emb, answer=result,
                           mode=mode, llm_in=q_in, llm_out=q_out, emb_tok=q_emb,
                           cost=q_cost, sec=total_sec)
        if not silent: print(f"  [캐시] 저장 ({_query_cache.summary()})")
    return {"query": query, "mode": mode, "answer": result, "cache_hit": False,
            "sec": total_sec, "llm_in": q_in, "llm_out": q_out,
            "emb_tok": q_emb, "cost": q_cost, "sources": sources}


async def run_queries(rag):
    global _phase
    _phase = "query"
    for q in DEFAULT_QUERIES:
        await _run_one_query(rag, q)


async def run_mode_compare(rag, query):
    global _phase
    _phase = "query"
    print(f"\n{'='*70}\n  [모드 비교] {query}\n{'='*70}")
    results = []
    for mode in MODES:
        res = await _run_one_query(rag, query, mode=mode, silent=True)
        results.append(res)
        print(f"  {mode:<8} | {res['sec']:>6.2f}초 | ${res['cost']:.5f}")
    print(f"\n  {'─'*60}")
    for r in results:
        print(f"  [{r['mode']}] {r['answer'].replace(chr(10),' ')[:500]}...")
    print()


async def run_batch_queries(rag, batch_file):
    global _phase
    _phase = "query"
    with open(batch_file, "r", encoding="utf-8") as f:
        queries = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    if not queries:
        return
    print(f"\n  [배치 쿼리] {len(queries)}개")
    results = []
    for i, q in enumerate(queries, 1):
        print(f"\n  [{i}/{len(queries)}] {q}")
        res = await _run_one_query(rag, q, silent=True)
        results.append(res)
        print(f"    → {res['sec']:.2f}초 | ${res['cost']:.5f}")
    out_path = os.path.join(WORKING_DIR, "batch_result.md")
    lines = [f"# 배치 쿼리 결과\n\n생성: {datetime.now():%Y-%m-%d %H:%M:%S}\n",
             "| # | 질문 | 시간 | 비용 |", "|---|------|------|------|"]
    for i, r in enumerate(results, 1):
        lines.append(f"| {i} | {r['query'][:40]} | {r['sec']:.2f}초 | ${r['cost']:.5f} |")
    lines.append("")
    for i, r in enumerate(results, 1):
        lines.append(f"## Q{i}. {r['query']}\n\n{r['answer']}\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  결과 저장: {out_path}")


# ==============================================================================
# 비용 최종 요약
# ==============================================================================
def print_total_cost(total_elapsed: float) -> None:
    ins = _usage["insert"]
    qry = _usage["query"]
    hea = _usage["heal"]
    bi  = _usage["batch_insert"]
    bh  = _usage["batch_heal"]

    ic  = _cost(ins["llm_in"], ins["llm_out"], ins["emb"])        # 실시간 삽입
    bic = _cost(bi["llm_in"],  bi["llm_out"],  batch=True)        # 배치 삽입
    qc  = _cost(qry["llm_in"], qry["llm_out"], qry["emb"])        # 쿼리
    hc  = _cost(hea["llm_in"], hea["llm_out"], hea["emb"])        # 힐링 실시간
    bhc = _cost(bh["llm_in"],  bh["llm_out"],  batch=True)        # 힐링 배치

    # 배치로 처리한 토큰이 실시간으로 처리됐다면 들었을 가상 비용
    bic_rt = _cost(bi["llm_in"], bi["llm_out"])
    bhc_rt = _cost(bh["llm_in"], bh["llm_out"])
    saved  = (bic_rt - bic) + (bhc_rt - bhc)

    tot = ic + bic + qc + hc + bhc
    print(f"\n{'='*56}")
    print("[전체 비용 최종 요약]")
    print(f"  모델: {_COST_TABLE[LLM_MODEL]['name']}")
    print(f"  삽입 (실시간 LLM)  : ${ic:.5f}  (= ₩{ic*1380:,.0f})")
    if bic > 0:
        print(f"  삽입 (배치 API)    : ${bic:.5f}  (= ₩{bic*1380:,.0f})")
    print(f"  힐링 (실시간 LLM)  : ${hc:.5f}  (= ₩{hc*1380:,.0f})")
    if bhc > 0:
        print(f"  힐링 (배치 API)    : ${bhc:.5f}  (= ₩{bhc*1380:,.0f})")
    print(f"  쿼리               : ${qc:.5f}  (= ₩{qc*1380:,.0f})")
    print(f"  {'─'*44}")
    print(f"  합계               : ${tot:.5f}  (= ₩{tot*1380:,.0f})")
    if saved > 0.0001:
        print(f"  배치 절감          : ${saved:.5f}  (= ₩{saved*1380:,.0f})  "
              f"({saved/(tot+saved)*100:.0f}% 절감)")
    print(f"  총 소요 시간       : {total_elapsed:.1f}초")
    print(f"{'='*56}\n")


# ==============================================================================
# 배치 상태 출력
# ==============================================================================
async def print_batch_status():
    if not os.path.exists(BATCH_STATE_FILE):
        print("  저장된 배치 없음.")
        return
    with open(BATCH_STATE_FILE, encoding="utf-8") as f:
        state = json.load(f)
    if not state:
        print("  저장된 배치 없음.")
        return
    mgr = BatchJobManager()
    print(f"\n  [저장된 배치 상태]")
    for btype, info in state.items():
        print(f"\n  [{btype}]  batch_id={info['batch_id']}")
        print(f"    제출: {info.get('submitted_at')}")
        try:
            status = await mgr.check_status(info["batch_id"])
            print(f"    상태: {status['status']}  완료: {status['completed']}/{status['total']}")
        except Exception as e:
            print(f"    상태 확인 실패: {e}")


# ==============================================================================
# CLI & 메인
# ==============================================================================
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="total_process_batch.py",
        description="LightRAG + OpenAI Batch API 통합 파이프라인 (비용 50% 절감)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # 공통
    p.add_argument("--llm", choices=["mini", "4o"], default=LLM_MODEL)
    p.add_argument("--chunk-size",    type=int, default=CHUNK_SIZE)
    p.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)

    # 배치 제어
    p.add_argument("--no-batch",     action="store_true",
                   help="배치 API 사용 안 함 (실시간 모드, total_process.py 동작과 동일)")
    p.add_argument("--submit-only",  action="store_true",
                   help="배치 제출 후 즉시 종료 (폴링 안 함)")
    p.add_argument("--resume",       choices=["insert", "heal", "all"],
                   help="이전 배치를 재개: insert=삽입, heal=힐링, all=둘 다")
    p.add_argument("--poll-interval", type=int, default=BATCH_POLL_INTERVAL,
                   help=f"배치 폴링 간격 초 (기본: {BATCH_POLL_INTERVAL})")
    p.add_argument("--batch-status", action="store_true",
                   help="저장된 배치 상태 확인 후 종료")

    # 삽입 제어
    p.add_argument("--skip-insert",  action="store_true")

    # 힐링
    p.add_argument("--heal",         action="store_true")
    p.add_argument("--heal-all",     action="store_true")
    p.add_argument("--heal-prune",   action="store_true")
    p.add_argument("--heal-embed",   action="store_true")
    p.add_argument("--heal-llm",     action="store_true")
    p.add_argument("--heal-relink",  action="store_true")
    p.add_argument("--batch-heal",   action="store_true",
                   help="힐링 C+D를 Batch API로 실행 (--heal 과 함께 사용)")
    p.add_argument("--dry-run",      action="store_true")
    p.add_argument("--isolated-detail", action="store_true")
    p.add_argument("--prune-min-desc",  type=int, default=HEAL_PRUNE_MIN_DESC)
    p.add_argument("--embed-threshold", type=float, default=HEAL_EMBED_THRESHOLD)
    p.add_argument("--embed-top-k",     type=int,   default=HEAL_EMBED_TOP_K)
    p.add_argument("--llm-limit",       type=int,   default=HEAL_LLM_LIMIT)
    p.add_argument("--llm-min-confidence", type=float, default=HEAL_LLM_MIN_CONF)
    p.add_argument("--relink-limit",    type=int,   default=HEAL_RELINK_LIMIT)

    # 통계 / 시각화
    p.add_argument("--stats",     action="store_true")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--max-nodes", type=int, default=1000)

    # 쿼리
    p.add_argument("-q", "--query",    nargs="+", metavar="WORD")
    p.add_argument("--mode", choices=MODES, default="hybrid")
    p.add_argument("--mode-compare",   nargs="+", metavar="WORD")
    p.add_argument("--batch",          metavar="FILE")
    p.add_argument("--no-cache",       action="store_true")
    p.add_argument("--cache-threshold", type=float, default=CACHE_SIMILARITY_THRESHOLD)
    p.add_argument("--show-cache",     action="store_true")
    return p


async def main() -> None:
    global _phase, _query_cache, CHUNK_SIZE, CHUNK_OVERLAP, LLM_MODEL, CACHE_SIMILARITY_THRESHOLD

    parser = _build_parser()
    args   = parser.parse_args()

    CHUNK_SIZE    = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    LLM_MODEL     = args.llm
    if args.cache_threshold != CACHE_SIMILARITY_THRESHOLD:
        CACHE_SIMILARITY_THRESHOLD = args.cache_threshold

    use_batch = not args.no_batch
    rates = _COST_TABLE[LLM_MODEL]
    print(f"  [모델] {rates['name']}  "
          f"(실시간: in=${rates['in']*1000:.3f}/1M  out=${rates['out']*1000:.3f}/1M)")
    if use_batch:
        print(f"  [배치] 활성화  "
              f"(배치: in=${rates['batch_in']*1000:.3f}/1M  out=${rates['batch_out']*1000:.3f}/1M  50% 할인)")

    cache_path = os.path.join(WORKING_DIR, "query_cache.json")
    if not args.no_cache:
        _query_cache = QueryCache(cache_path)
    else:
        _query_cache = None

    t_start = time.time()

    # ── 즉시 종료 단축 경로 ─────────────────────────────────────
    if args.batch_status:
        await print_batch_status()
        return

    if args.show_cache:
        if _query_cache and _query_cache.data:
            for cid, e in _query_cache.data.items():
                print(f"  [{cid}] {e['query']}\n    {e['created_at']} | ${e['cost']:.5f}")
        else:
            print("  캐시 비어 있음")
        return

    if args.stats:
        print_graph_stats()
        return

    if args.visualize:
        visualize_graph(max_nodes=args.max_nodes)
        return

    # ── 배치 재개 ────────────────────────────────────────────────
    if args.resume:
        rag = _build_rag()
        await rag.initialize_storages()
        batch_cache = None

        if args.resume in ("insert", "all"):
            batch_cache = await resume_insert_batch(args.poll_interval)
            if batch_cache:
                _phase = "insert"
                await insert_documents(rag, batch_cache)
                print_insert_summary(time.time() - t_start)

        if args.resume in ("heal", "all"):
            state_c = BatchJobManager.load_state("heal_c")
            state_d = BatchJobManager.load_state("heal_d")
            if state_c or state_d:
                await run_heal_batch(
                    do_prune=False, do_embed=False,
                    do_llm=bool(state_c), do_relink=bool(state_d),
                    poll_interval=args.poll_interval,
                )
            else:
                print("  저장된 힐링 배치 없음.")

        print_total_cost(time.time() - t_start)
        return

    # ── 힐링 단독 ────────────────────────────────────────────────
    heal_requested = (args.heal or args.heal_all or args.heal_prune
                      or args.heal_embed or args.heal_llm or args.heal_relink)

    if heal_requested and not (args.query or args.mode_compare or args.batch):
        if args.heal_all:
            do_prune, do_embed, do_llm, do_relink = True, True, True, True
        elif any([args.heal_prune, args.heal_embed, args.heal_llm, args.heal_relink]):
            do_prune, do_embed, do_llm, do_relink = (
                args.heal_prune, args.heal_embed, args.heal_llm, args.heal_relink)
        else:
            do_prune, do_embed, do_llm, do_relink = True, True, True, False

        await run_heal_batch(
            do_prune=do_prune, do_embed=do_embed,
            do_llm=do_llm,     do_relink=do_relink,
            dry_run=args.dry_run,
            submit_only=args.submit_only and use_batch,
            poll_interval=args.poll_interval,
            embed_threshold=args.embed_threshold,
            embed_top_k=args.embed_top_k,
            llm_limit=args.llm_limit,
            llm_min_confidence=args.llm_min_confidence,
            relink_limit=args.relink_limit,
            prune_min_desc=args.prune_min_desc,
            isolated_detail=args.isolated_detail,
        )
        print_total_cost(time.time() - t_start)
        return

    # ── RAG 인스턴스 ─────────────────────────────────────────────
    rag = _build_rag()
    await rag.initialize_storages()

    if args.mode_compare:
        await run_mode_compare(rag, " ".join(args.mode_compare))
        print_total_cost(time.time() - t_start)
        return
    if args.batch:
        await run_batch_queries(rag, args.batch)
        print_total_cost(time.time() - t_start)
        return
    if args.query:
        _phase = "query"
        await _run_one_query(rag, " ".join(args.query), mode=args.mode)
        print_total_cost(time.time() - t_start)
        return

    # ════════════════════════════════════════════════════════════
    # 전체 파이프라인: 배치 삽입 → 배치 힐링 → 시각화 → 통계 → 쿼리
    # ════════════════════════════════════════════════════════════
    print("\ntotal_process_batch.py — 전체 파이프라인 시작\n")

    # 1. 삽입
    if not args.skip_insert:
        _phase = "insert"
        t1 = time.time()
        if use_batch:
            md_files = sorted(
                os.path.join(MD_DIR, f)
                for f in os.listdir(MD_DIR) if f.endswith(".md")
            )
            batch_cache = await collect_and_submit_insert_batch(
                md_files,
                submit_only=args.submit_only,
                poll_interval=args.poll_interval,
            )
            if args.submit_only:
                print("\n  --submit-only: 삽입 배치 제출 완료. 종료합니다.")
                print("  재개: python total_process_batch.py --resume insert")
                return
            await insert_documents(rag, batch_cache)
        else:
            await insert_documents(rag, batch_cache=None)
        print_insert_summary(time.time() - t1)
    else:
        print("  [삽입 건너뜀]\n")

    # 2. 힐링 (배치)
    await run_heal_batch(
        do_prune=True, do_embed=True, do_llm=True,
        do_relink=args.heal_all,
        dry_run=args.dry_run,
        submit_only=args.submit_only and use_batch,
        poll_interval=args.poll_interval,
        embed_threshold=args.embed_threshold,
        embed_top_k=args.embed_top_k,
        llm_limit=args.llm_limit,
        llm_min_confidence=args.llm_min_confidence,
        relink_limit=args.relink_limit,
        prune_min_desc=args.prune_min_desc,
        isolated_detail=args.isolated_detail,
    )

    if args.submit_only and use_batch:
        print("\n  --submit-only: 힐링 배치 제출 완료. 종료합니다.")
        print("  재개: python total_process_batch.py --resume heal")
        print_total_cost(time.time() - t_start)
        return

    # 3. 시각화
    print()
    visualize_graph(max_nodes=args.max_nodes)

    # 4. 통계
    print_graph_stats(label="최종")

    # 5. 쿼리
    await run_queries(rag)
    if _query_cache:
        print(f"\n  [쿼리 캐시] {_query_cache.summary()}")

    print_total_cost(time.time() - t_start)


if __name__ == "__main__":
    asyncio.run(main())
