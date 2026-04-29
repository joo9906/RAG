"""
total_test.py — 전체 단위/통합 테스트
======================================

대상:
  - LLMReranker          (src/supervisors/v3/tools/reranker.py)
  - HistoryManager       (src/supervisors/v3/history_manager.py)
  - ContextEnhancer      (src/supervisors/v3/supervisor/supervisor.py)
  - _build_enhanced_input / _prepare_request_context  (supervisor_v3.py)

실행:
  venv/Scripts/python total_test.py          # 전체
  venv/Scripts/python total_test.py rerank   # Reranker만
  venv/Scripts/python total_test.py history  # HistoryManager만
  venv/Scripts/python total_test.py context  # ContextEnhancer만
  venv/Scripts/python total_test.py prepare  # _prepare_request_context만
"""

# ── 경로 설정 (venv/Scripts/python 또는 일반 python 모두 대응)
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 외부 로그 노이즈 억제
for _noisy in ("httpx", "httpcore", "openai", "urllib3", "langchain", "langsmith"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

import copy
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 공통 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_doc(content: str, label: str = "") -> Document:
    return Document(page_content=content, metadata={"label": label} if label else {})


def make_rerank_response(relevance=5, specificity=5, completeness=5,
                         practicality=5, constraint_fit=5) -> MagicMock:
    m = MagicMock()
    m.relevance = relevance
    m.specificity = specificity
    m.completeness = completeness
    m.practicality = practicality
    m.constraint_fit = constraint_fit
    return m


def make_llm_response(content: str) -> MagicMock:
    m = MagicMock()
    m.content = content
    return m


SEP = "─" * 70


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. LLMReranker 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestLLMRerankerWeightedScore(unittest.TestCase):
    """_calculate_weighted_score 단위 테스트 (LLM 불필요)"""

    def _make_reranker_no_llm(self):
        """LLM 초기화 없이 LLMReranker 인스턴스 생성"""
        from supervisors.v3.tools.reranker import LLMReranker
        with patch("supervisors.v3.tools.reranker.LangchainLoader"), \
             patch("supervisors.v3.tools.reranker.PromptManager"):
            r = LLMReranker.__new__(LLMReranker)
            r.threshold = 0.3
            r.rerank_prompt = "dummy"
            r.llm = MagicMock()
            return r

    def test_perfect_score(self):
        """모든 기준 10점 → 가중합 1.0"""
        from supervisors.v3.tools.reranker import LLMReranker
        r = self._make_reranker_no_llm()
        scores = {k: 10 for k in LLMReranker.WEIGHTS}
        result = r._calculate_weighted_score(scores)
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_zero_score(self):
        """모든 기준 0점 → 0.0"""
        from supervisors.v3.tools.reranker import LLMReranker
        r = self._make_reranker_no_llm()
        scores = {k: 0 for k in LLMReranker.WEIGHTS}
        result = r._calculate_weighted_score(scores)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_only_relevance(self):
        """관련성(weight=0.6)만 10점, 나머지 0 → 0.6"""
        r = self._make_reranker_no_llm()
        scores = {
            "relevance_score": 10,
            "specificity_score": 0,
            "completeness_score": 0,
            "practicality_score": 0,
            "constraint_fit_score": 0,
        }
        result = r._calculate_weighted_score(scores)
        self.assertAlmostEqual(result, 0.6, places=6)

    def test_missing_key_treated_as_zero(self):
        """없는 키 → 0으로 처리"""
        r = self._make_reranker_no_llm()
        result = r._calculate_weighted_score({})
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_half_score(self):
        """모든 기준 5점 → 0.5"""
        from supervisors.v3.tools.reranker import LLMReranker
        r = self._make_reranker_no_llm()
        scores = {k: 5 for k in LLMReranker.WEIGHTS}
        result = r._calculate_weighted_score(scores)
        self.assertAlmostEqual(result, 0.5, places=6)

    def test_weights_sum_to_one(self):
        """가중치 합 = 1.0"""
        from supervisors.v3.tools.reranker import LLMReranker
        total = sum(LLMReranker.WEIGHTS.values())
        self.assertAlmostEqual(total, 1.0, places=6)


class TestLLMRerankerRerank(unittest.TestCase):
    """rerank() 동작 검증 (Mock LLM)"""

    def _make_reranker(self, threshold=0.3, prompt="dummy {user_query} {search_query} {document}"):
        from supervisors.v3.tools.reranker import LLMReranker, RerankResponse
        with patch("supervisors.v3.tools.reranker.LangchainLoader"), \
             patch("supervisors.v3.tools.reranker.PromptManager"):
            r = LLMReranker.__new__(LLMReranker)
            r.threshold = threshold
            r.rerank_prompt = prompt
            r.MAX_WORKERS = 10
            r.BATCH_TIMEOUT_SEC = 30
            r.SINGLE_DOC_TIMEOUT_SEC = 5
            r.EPSILON = 1e-9
            r.WEIGHTS = LLMReranker.WEIGHTS
            r.llm = MagicMock()
        return r

    def test_empty_docs_returns_empty(self):
        r = self._make_reranker()
        result = r.rerank([], "쿼리")
        self.assertEqual(result, [])

    def test_no_prompt_returns_original(self):
        """프롬프트 미로드 → 원본 반환"""
        r = self._make_reranker()
        r.rerank_prompt = None
        docs = [make_doc("문서1"), make_doc("문서2")]
        result = r.rerank(docs, "쿼리")
        self.assertEqual(result, docs)

    def test_threshold_filters_low_score(self):
        """threshold 미만 문서 제거"""
        r = self._make_reranker(threshold=0.5)
        # score 낮은 응답 (relevance=2 → 가중합 ≈ 0.12)
        r.llm.invoke.return_value = make_rerank_response(relevance=2, specificity=2,
                                                          completeness=2, practicality=2,
                                                          constraint_fit=2)
        docs = [make_doc("낮은 관련성 문서")]
        result = r.rerank(docs, "쿼리")
        self.assertEqual(len(result), 0)

    def test_threshold_passes_high_score(self):
        """threshold 이상 문서 통과"""
        r = self._make_reranker(threshold=0.3)
        # score 높은 응답 (relevance=8 → 가중합 ≈ 0.64)
        r.llm.invoke.return_value = make_rerank_response(relevance=8, specificity=8,
                                                          completeness=8, practicality=8,
                                                          constraint_fit=8)
        docs = [make_doc("관련성 높은 문서")]
        result = r.rerank(docs, "쿼리")
        self.assertEqual(len(result), 1)

    def test_sorted_by_score_descending(self):
        """높은 점수 문서가 앞에 와야 함"""
        r = self._make_reranker(threshold=0.0)
        responses = [
            make_rerank_response(relevance=3),   # 낮은 점수
            make_rerank_response(relevance=9),   # 높은 점수
            make_rerank_response(relevance=6),   # 중간 점수
        ]
        r.llm.invoke.side_effect = responses
        docs = [make_doc(f"문서{i+1}") for i in range(3)]
        result = r.rerank(docs, "쿼리")

        self.assertEqual(len(result), 3)
        scores = [d.metadata["overall_score"] for d in result]
        self.assertEqual(scores, sorted(scores, reverse=True), "점수 내림차순 정렬 실패")

    def test_rank_metadata_added(self):
        """rank 메타데이터가 1부터 순서대로 추가되어야 함"""
        r = self._make_reranker(threshold=0.0)
        r.llm.invoke.return_value = make_rerank_response(relevance=5)
        docs = [make_doc(f"문서{i+1}") for i in range(3)]
        result = r.rerank(docs, "쿼리")

        ranks = [d.metadata["rank"] for d in result]
        self.assertEqual(ranks, list(range(1, len(result) + 1)))

    def test_overall_score_metadata_added(self):
        """overall_score 메타데이터가 추가되어야 함"""
        r = self._make_reranker(threshold=0.0)
        r.llm.invoke.return_value = make_rerank_response(relevance=7)
        result = r.rerank([make_doc("문서")], "쿼리")
        self.assertIn("overall_score", result[0].metadata)

    def test_search_query_defaults_to_user_query(self):
        """search_query 미전달 시 user_query로 대체"""
        r = self._make_reranker(threshold=0.0)
        r.llm.invoke.return_value = make_rerank_response(relevance=5)
        # 예외 없이 실행되면 OK
        result = r.rerank([make_doc("문서")], user_query="쿼리")
        self.assertEqual(len(result), 1)

    def test_llm_error_doc_filtered(self):
        """LLM 오류 발생 시 해당 문서 점수=0 → threshold 미만이면 제거"""
        r = self._make_reranker(threshold=0.1)
        r.llm.invoke.side_effect = RuntimeError("LLM 오류")
        result = r.rerank([make_doc("오류 문서")], "쿼리")
        self.assertEqual(len(result), 0)

    def test_sub_scores_stored_in_metadata(self):
        """5개 세부 점수가 모두 metadata에 저장되어야 함"""
        r = self._make_reranker(threshold=0.0)
        r.llm.invoke.return_value = make_rerank_response(
            relevance=7, specificity=6, completeness=8, practicality=5, constraint_fit=9
        )
        result = r.rerank([make_doc("문서")], "쿼리")
        meta = result[0].metadata
        for key in ["relevance_score", "specificity_score", "completeness_score",
                    "practicality_score", "constraint_fit_score"]:
            self.assertIn(key, meta, f"{key} 메타데이터 누락")

    def test_multiple_docs_all_pass(self):
        """여러 문서 모두 threshold 이상이면 전부 반환"""
        r = self._make_reranker(threshold=0.0)
        r.llm.invoke.return_value = make_rerank_response(relevance=5)
        docs = [make_doc(f"문서{i}") for i in range(5)]
        result = r.rerank(docs, "쿼리")
        self.assertEqual(len(result), 5)

    def test_original_docs_not_modified_on_no_prompt(self):
        """프롬프트 없을 때 원본 doc 객체 그대로 반환 (참조 동일)"""
        r = self._make_reranker()
        r.rerank_prompt = None
        docs = [make_doc("문서")]
        result = r.rerank(docs, "쿼리")
        self.assertIs(result[0], docs[0])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. HistoryManager 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestHistoryManagerTokenCount(unittest.TestCase):
    """토큰 카운팅 단위 테스트"""

    def _make_manager(self):
        from supervisors.v3.history_manager import HistoryManager
        model = MagicMock()
        return HistoryManager(model=model)

    def test_empty_messages(self):
        m = self._make_manager()
        self.assertEqual(m._count_tokens([]), 0)

    def test_single_message_token_overhead(self):
        """각 메시지 4토큰 오버헤드 포함"""
        m = self._make_manager()
        msg = HumanMessage(content="안녕")
        tokens = m._count_tokens([msg])
        text_tokens = m._count_text_tokens("안녕")
        self.assertEqual(tokens, text_tokens + 4)

    def test_multiple_messages(self):
        m = self._make_manager()
        msgs = [HumanMessage(content="안녕"), AIMessage(content="반갑습니다")]
        tokens = m._count_tokens(msgs)
        expected = (
            m._count_text_tokens("안녕") + 4 +
            m._count_text_tokens("반갑습니다") + 4
        )
        self.assertEqual(tokens, expected)

    def test_empty_content(self):
        m = self._make_manager()
        msg = HumanMessage(content="")
        tokens = m._count_tokens([msg])
        self.assertEqual(tokens, 4)  # 오버헤드만


class TestHistoryManagerBuild(unittest.IsolatedAsyncioTestCase):
    """build() 및 get_history() 동작 테스트"""

    def _make_manager(self):
        from supervisors.v3.history_manager import HistoryManager
        from unittest.mock import AsyncMock
        model = MagicMock()
        model.ainvoke = AsyncMock(return_value=make_llm_response("요약된 내용입니다."))
        return HistoryManager(model=model)

    async def test_build_stores_messages(self):
        m = self._make_manager()
        msgs = [HumanMessage(content="질문1"), AIMessage(content="답변1")]
        await m.build(msgs)
        self.assertEqual(m.recent_messages, msgs)

    async def test_get_history_no_summary(self):
        """요약 없을 때 recent_messages 그대로 반환"""
        m = self._make_manager()
        msgs = [HumanMessage(content="질문1"), AIMessage(content="답변1")]
        await m.build(msgs)
        result = m.get_history()
        self.assertEqual(result, msgs)
        # SystemMessage 없어야 함
        self.assertFalse(any(isinstance(r, SystemMessage) for r in result))

    async def test_get_history_with_summary(self):
        """요약 있을 때 SystemMessage가 첫 번째로 prepend"""
        m = self._make_manager()
        m.summary = "이전 요약"
        m.recent_messages = [HumanMessage(content="최근 질문")]
        result = m.get_history()
        self.assertIsInstance(result[0], SystemMessage)
        self.assertIn("이전 요약", result[0].content)
        self.assertIn("[이전 대화 요약]", result[0].content)
        self.assertEqual(len(result), 2)

    async def test_empty_history_no_summary(self):
        m = self._make_manager()
        await m.build([])
        self.assertEqual(m.get_history(), [])

    async def test_build_replaces_previous_messages(self):
        """build()를 재호출하면 이전 메시지 교체"""
        m = self._make_manager()
        await m.build([HumanMessage(content="이전")])
        await m.build([HumanMessage(content="새로운")])
        result = m.get_history()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "새로운")


class TestHistoryManagerSummarize(unittest.IsolatedAsyncioTestCase):
    """토큰 초과 시 요약 트리거 테스트"""

    def _make_manager_with_model(self, summary_text="요약"):
        from supervisors.v3.history_manager import HistoryManager
        from unittest.mock import AsyncMock
        model = MagicMock()
        model.ainvoke = AsyncMock(return_value=make_llm_response(summary_text))
        return HistoryManager(model=model)

    async def test_under_limit_no_summary(self):
        """120K 미만이면 요약 안 함"""
        m = self._make_manager_with_model()
        msgs = [HumanMessage(content="짧은 메시지"), AIMessage(content="짧은 답변")]
        await m.build(msgs)
        self.assertEqual(m.summary, "")
        m.model.ainvoke.assert_not_called()

    async def test_over_limit_triggers_summarize(self):
        """120K 초과 시 model.ainvoke 호출"""
        from supervisors.v3.history_manager import HistoryManager, MAX_HISTORY_TOKENS
        from unittest.mock import AsyncMock
        model = MagicMock()
        model.ainvoke = AsyncMock(return_value=make_llm_response("긴 히스토리 요약"))
        m = HistoryManager(model=model)

        # 토큰 카운트를 강제로 초과시키기 위해 패치
        with patch.object(m, '_count_tokens', return_value=MAX_HISTORY_TOKENS + 1), \
             patch.object(m, '_count_text_tokens', return_value=0), \
             patch.object(m, '_find_split_index', return_value=5):
            msgs = [HumanMessage(content=f"메시지{i}") for i in range(10)]
            m.recent_messages = msgs
            await m._summarize_history()

        model.ainvoke.assert_called_once()
        self.assertNotEqual(m.summary, "")

    async def test_summary_stored_after_trigger(self):
        """요약 후 self.summary에 결과 저장"""
        from supervisors.v3.history_manager import HistoryManager, MAX_HISTORY_TOKENS
        from unittest.mock import AsyncMock
        model = MagicMock()
        model.ainvoke = AsyncMock(return_value=make_llm_response("통합 요약 결과"))
        m = HistoryManager(model=model)

        with patch.object(m, '_count_tokens', return_value=MAX_HISTORY_TOKENS + 1), \
             patch.object(m, '_count_text_tokens', return_value=0), \
             patch.object(m, '_find_split_index', return_value=5):
            msgs = [HumanMessage(content=f"msg{i}") for i in range(10)]
            m.recent_messages = msgs
            await m._summarize_history()

        self.assertEqual(m.summary, "통합 요약 결과")

    async def test_oldest_messages_removed_after_summarize(self):
        """요약된 메시지는 recent_messages에서 제거되어야 함"""
        from supervisors.v3.history_manager import HistoryManager, MAX_HISTORY_TOKENS
        from unittest.mock import AsyncMock
        model = MagicMock()
        model.ainvoke = AsyncMock(return_value=make_llm_response("요약"))
        m = HistoryManager(model=model)

        n = 10
        msgs = [HumanMessage(content=f"msg{i}") for i in range(n)]

        with patch.object(m, '_count_tokens', return_value=MAX_HISTORY_TOKENS + 1), \
             patch.object(m, '_count_text_tokens', return_value=0), \
             patch.object(m, '_find_split_index', return_value=5):
            m.recent_messages = msgs[:]
            await m._summarize_history()

        self.assertLess(len(m.recent_messages), n)

    async def test_summarize_llm_failure_fallback(self):
        """LLM 실패 시 빈 문자열(기존 요약 유지) 반환"""
        from supervisors.v3.history_manager import HistoryManager
        from unittest.mock import AsyncMock
        model = MagicMock()
        model.ainvoke = AsyncMock(side_effect=RuntimeError("LLM 연결 실패"))
        m = HistoryManager(model=model)
        msgs = [HumanMessage(content="원본 메시지")]
        result = await m._summarize(msgs)
        self.assertEqual(result, "")

    async def test_rolling_summary_integrates_previous(self):
        """두 번째 요약 시 이전 요약이 프롬프트에 포함"""
        from supervisors.v3.history_manager import HistoryManager
        from unittest.mock import AsyncMock
        model = MagicMock()
        model.ainvoke = AsyncMock(return_value=make_llm_response("통합 요약"))
        m = HistoryManager(model=model)
        m.summary = "이전 요약"

        msgs = [HumanMessage(content="새 메시지")]
        await m._summarize(msgs)

        # 실제 ainvoke에 전달된 프롬프트 확인
        call_args = model.ainvoke.call_args[0][0]
        prompt_text = call_args[0].content  # HumanMessage.content
        self.assertIn("이전 요약", prompt_text)
        self.assertIn("통합", prompt_text)

    def test_build_summary_prompt_no_previous(self):
        """이전 요약 없을 때 프롬프트에 '통합' 없음"""
        from supervisors.v3.history_manager import HistoryManager
        m = HistoryManager(model=MagicMock())
        m.summary = ""
        prompt = m._build_summary_prompt([HumanMessage(content="질문")])
        self.assertNotIn("통합", prompt)
        self.assertIn("다음 대화를 요약", prompt)

    def test_format_for_summary_role_labels(self):
        """role 레이블이 한국어로 변환"""
        from supervisors.v3.history_manager import HistoryManager
        m = HistoryManager(model=MagicMock())
        msgs = [
            HumanMessage(content="질문"),
            AIMessage(content="답변"),
            SystemMessage(content="시스템"),
        ]
        text = m._format_for_summary(msgs)
        self.assertIn("[사용자]", text)
        self.assertIn("[어시스턴트]", text)
        self.assertIn("[시스템]", text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. ContextEnhancer 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestContextEnhancerInit(unittest.TestCase):
    """ContextEnhancer 초기화 테스트"""

    def _make_enhancer(self, product_list=None, hospital_list=None, model=None,
                       comp_path=None):
        from supervisors.v3.supervisor.supervisor import ContextEnhancer
        if comp_path is None:
            comp_path = str(PROJECT_ROOT / "src" / "supervisors" / "v3" /
                           "competitive_product_list.txt")
        with patch("project_paths.CE_PRODUCT_LIST_TXT_PATH", "dummy", create=True), \
             patch("project_paths.CE_COMPETITIVE_PRODUCT_LIST_TXT_PATH", comp_path, create=True):
            return ContextEnhancer(
                product_list=product_list or [],
                hospital_list=hospital_list or [],
                model=model,
            )

    def test_product_list_set(self):
        ce = self._make_enhancer(product_list=["제품A", "제품B"])
        self.assertEqual(ce.product_list, ["제품A", "제품B"])

    def test_hospital_list_set(self):
        ce = self._make_enhancer(hospital_list=["한국병원", "서울의원"])
        self.assertEqual(ce.hospital_list, ["한국병원", "서울의원"])

    def test_model_stored(self):
        mock_model = MagicMock()
        ce = self._make_enhancer(model=mock_model)
        self.assertIs(ce.model, mock_model)


class TestContextEnhancerDetection(unittest.TestCase):
    """제품/병원/경쟁제품 감지 테스트"""

    def _make_enhancer(self, products=None, hospitals=None, comp_products=None):
        from supervisors.v3.supervisor.supervisor import ContextEnhancer
        comp_path = str(PROJECT_ROOT / "src" / "supervisors" / "v3" /
                       "competitive_product_list.txt")
        with patch("project_paths.CE_PRODUCT_LIST_TXT_PATH", "dummy", create=True), \
             patch("project_paths.CE_COMPETITIVE_PRODUCT_LIST_TXT_PATH", comp_path, create=True):
            ce = ContextEnhancer(
                product_list=products or [],
                hospital_list=hospitals or [],
                model=None,
            )
        if comp_products is not None:
            ce.competitive_product_list = comp_products
        return ce

    def test_detect_product_exact_match(self):
        ce = self._make_enhancer(products=["아세리스", "리리카"])
        result = ce._detect_products("아세리스 처방 현황 알려줘")
        self.assertIn("아세리스", result)

    def test_detect_product_case_insensitive(self):
        ce = self._make_enhancer(products=["ACERYS"])
        result = ce._detect_products("acerys 제품 정보")
        self.assertIn("ACERYS", result)

    def test_detect_product_not_in_text(self):
        ce = self._make_enhancer(products=["아세리스"])
        result = ce._detect_products("리리카 처방 현황")
        self.assertNotIn("아세리스", result)

    def test_detect_hospital(self):
        ce = self._make_enhancer(hospitals=["스마트정형외과의원"])
        result = ce._detect_hospitals("스마트정형외과의원 방문 계획")
        self.assertIn("스마트정형외과의원", result)

    def test_detect_competitive_product(self):
        ce = self._make_enhancer(comp_products=["경쟁제품X"])
        result = ce._detect_competitive_products("경쟁제품X 비교 분석")
        self.assertIn("경쟁제품X", result)

    def test_no_match_returns_empty(self):
        ce = self._make_enhancer(products=["아세리스"])
        result = ce._detect_products("오늘 날씨 어때?")
        self.assertEqual(result, [])


class TestContextEnhancerAnalyzeQuestion(unittest.TestCase):
    """_analyze_question_and_history 테스트"""

    def _make_enhancer_with_model(self, model):
        from supervisors.v3.supervisor.supervisor import ContextEnhancer
        comp_path = str(PROJECT_ROOT / "src" / "supervisors" / "v3" /
                       "competitive_product_list.txt")
        with patch("project_paths.CE_PRODUCT_LIST_TXT_PATH", "dummy", create=True), \
             patch("project_paths.CE_COMPETITIVE_PRODUCT_LIST_TXT_PATH", comp_path, create=True):
            return ContextEnhancer(product_list=[], hospital_list=[], model=model)

    def test_no_history_returns_original(self):
        """히스토리 없으면 original_input 그대로"""
        ce = self._make_enhancer_with_model(MagicMock())
        result = ce._analyze_question_and_history("오늘 매출은?", history=[])
        self.assertEqual(result.analyzed_question, "오늘 매출은?")

    def test_no_model_returns_original(self):
        """모델 없으면 original_input 그대로"""
        ce = self._make_enhancer_with_model(None)
        result = ce._analyze_question_and_history(
            "거기 방문 계획",
            history=[HumanMessage(content="스마트정형 방문했어"), AIMessage(content="네")]
        )
        self.assertEqual(result.analyzed_question, "거기 방문 계획")

    def test_with_history_calls_llm(self):
        """히스토리 있으면 LLM 호출"""
        from supervisors.v3.supervisor.supervisor import QuestionAnalysisResult
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = QuestionAnalysisResult(
            analyzed_question="스마트정형외과의원 방문 계획"
        )
        mock_model.with_structured_output.return_value = mock_structured
        ce = self._make_enhancer_with_model(mock_model)

        result = ce._analyze_question_and_history(
            "거기 방문 계획",
            history=[HumanMessage(content="스마트정형 얘기"), AIMessage(content="네")]
        )
        mock_structured.invoke.assert_called_once()
        self.assertEqual(result.analyzed_question, "스마트정형외과의원 방문 계획")

    def test_llm_error_returns_original(self):
        """LLM 오류 → original_input fallback"""
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = RuntimeError("LLM 오류")
        mock_model.with_structured_output.return_value = mock_structured
        ce = self._make_enhancer_with_model(mock_model)

        result = ce._analyze_question_and_history(
            "원본 질문",
            history=[HumanMessage(content="히스토리"), AIMessage(content="응답")]
        )
        self.assertEqual(result.analyzed_question, "원본 질문")


class TestContextEnhancerCall(unittest.TestCase):
    """ContextEnhancer.__call__ 통합 테스트"""

    def _make_enhancer(self, model=None, products=None, hospitals=None):
        from supervisors.v3.supervisor.supervisor import ContextEnhancer
        comp_path = str(PROJECT_ROOT / "src" / "supervisors" / "v3" /
                       "competitive_product_list.txt")
        with patch("project_paths.CE_PRODUCT_LIST_TXT_PATH", "dummy", create=True), \
             patch("project_paths.CE_COMPETITIVE_PRODUCT_LIST_TXT_PATH", comp_path, create=True):
            ce = ContextEnhancer(
                product_list=products or [],
                hospital_list=hospitals or [],
                model=model,
            )
        return ce

    def test_returns_required_keys(self):
        ce = self._make_enhancer()
        result = ce("테스트 질문")
        for key in ["original_input", "analyzed_question", "detected_products",
                    "detected_competitive_products", "detected_hospitals"]:
            self.assertIn(key, result, f"키 누락: {key}")

    def test_original_input_preserved(self):
        ce = self._make_enhancer()
        result = ce("오늘 매출은?")
        self.assertEqual(result["original_input"], "오늘 매출은?")

    def test_no_history_analyzed_equals_original(self):
        """히스토리·모델 없으면 analyzed_question == original_input"""
        ce = self._make_enhancer(model=None)
        result = ce("오늘 매출은?", history=[])
        self.assertEqual(result["analyzed_question"], "오늘 매출은?")

    def test_detected_products_populated(self):
        ce = self._make_enhancer(products=["아세리스"])
        result = ce("아세리스 처방 현황")
        self.assertIn("아세리스", result["detected_products"])

    def test_product_list_override_per_call(self):
        """호출 시 product_list 오버라이드 후 원복"""
        ce = self._make_enhancer(products=["기존제품"])
        result = ce("임시제품 정보", product_list=["임시제품"])
        self.assertIn("임시제품", result["detected_products"])
        # 호출 후 원복 확인
        self.assertEqual(ce.product_list, ["기존제품"])

    def test_hospital_list_override_per_call(self):
        ce = self._make_enhancer(hospitals=["기존병원"])
        result = ce("임시병원 방문", hospital_list=["임시병원"])
        self.assertIn("임시병원", result["detected_hospitals"])
        self.assertEqual(ce.hospital_list, ["기존병원"])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. _build_enhanced_input 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestBuildEnhancedInput(unittest.TestCase):
    """supervisor_v3._build_enhanced_input 단위 테스트"""

    def _make_supervisor_stub(self):
        """SupervisorV3의 _build_enhanced_input만 테스트하기 위한 stub"""
        from supervisors.v3.supervisor_v3 import SupervisorV3
        # 무거운 __init__ 우회
        obj = SupervisorV3.__new__(SupervisorV3)
        return obj

    def test_original_input_always_included(self):
        sv = self._make_supervisor_stub()
        result = sv._build_enhanced_input("원본", "원본", [], [])
        self.assertIn("original user input: 원본", result)

    def test_products_appended_when_detected(self):
        sv = self._make_supervisor_stub()
        result = sv._build_enhanced_input("원본", "원본", ["아세리스", "리리카"], [])
        self.assertIn("detected company products", result)
        self.assertIn("아세리스", result)

    def test_no_products_section_absent(self):
        sv = self._make_supervisor_stub()
        result = sv._build_enhanced_input("원본", "원본", [], [])
        self.assertNotIn("detected company products", result)

    def test_competitive_products_appended(self):
        sv = self._make_supervisor_stub()
        result = sv._build_enhanced_input("원본", "원본", [], ["경쟁품A"])
        self.assertIn("detected competitive products", result)

    def test_history_section_when_analyzed_differs(self):
        """analyzed_question이 original과 다를 때만 history 섹션 포함"""
        sv = self._make_supervisor_stub()
        result = sv._build_enhanced_input("거기 방문", "스마트정형외과의원 방문", [], [])
        self.assertIn("history information question", result)
        self.assertIn("스마트정형외과의원", result)

    def test_no_history_section_when_same(self):
        """analyzed_question == original → history 섹션 없음"""
        sv = self._make_supervisor_stub()
        result = sv._build_enhanced_input("동일 질문", "동일 질문", [], [])
        self.assertNotIn("history information question", result)

    def test_full_combination(self):
        sv = self._make_supervisor_stub()
        result = sv._build_enhanced_input(
            "거기 매출은?",
            "스마트정형외과의원 매출은?",
            ["아세리스"],
            ["경쟁품A"],
        )
        self.assertIn("original user input", result)
        self.assertIn("detected company products", result)
        self.assertIn("detected competitive products", result)
        self.assertIn("history information question", result)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. _prepare_request_context 통합 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestPrepareRequestContext(unittest.IsolatedAsyncioTestCase):
    """supervisor_v3._prepare_request_context 통합 테스트 (외부 의존성 Mock)"""

    def _make_supervisor_with_mocks(self):
        from supervisors.v3.supervisor_v3 import SupervisorV3
        sv = SupervisorV3.__new__(SupervisorV3)

        # ContextEnhancer mock
        ce_mock = MagicMock()
        ce_mock.return_value = {
            "original_input": "원본 질문",
            "analyzed_question": "분석된 질문",
            "detected_products": ["아세리스"],
            "detected_competitive_products": [],
            "detected_hospitals": ["스마트정형"],
        }
        sv._context_enhancer = ce_mock

        # HistoryManager mock
        from supervisors.v3.history_manager import HistoryManager
        from unittest.mock import AsyncMock
        hm_mock = MagicMock(spec=HistoryManager)
        hm_mock.build = AsyncMock()
        hm_mock.get_history.return_value = [HumanMessage(content="이전 질문")]
        sv._nano_model = MagicMock()

        # product/hospital loader mock
        sv._load_products = MagicMock(return_value=(["아세리스"], ["스마트정형"]))
        return sv, hm_mock

    async def test_returns_required_keys(self):
        from supervisors.v3.supervisor_v3 import SupervisorV3
        sv, hm_mock = self._make_supervisor_with_mocks()

        with patch("supervisors.v3.tools.product_loader.load_products_from_rdb",
                   return_value=("아세리스,리리카", 2)), \
             patch("supervisors.v3.tools.product_loader.load_hospitals_from_rdb",
                   return_value=("스마트정형외과의원", 1)), \
             patch("supervisors.v3.supervisor_v3.HistoryManager", return_value=hm_mock), \
             patch("supervisors.v3.supervisor_v3.convert_to_messages",
                   side_effect=lambda msgs: [HumanMessage(content=m["content"]) for m in msgs]):
            result = await sv._prepare_request_context(
                history=[],
                user_input="원본 질문",
                employee_ID="EMP001",
            )

        for key in ["all_messages", "known_products", "known_hospitals",
                    "original_input", "analyzed_question",
                    "detected_company_products", "detected_competitive_products",
                    "detected_hospitals"]:
            self.assertIn(key, result, f"키 누락: {key}")

    async def test_history_role_normalized_to_lowercase(self):
        """history role이 소문자로 정규화되는지 확인"""
        from supervisors.v3.supervisor_v3 import SupervisorV3
        sv, hm_mock = self._make_supervisor_with_mocks()

        captured = []

        def fake_convert(msgs):
            captured.extend(msgs)
            return [HumanMessage(content=m["content"]) for m in msgs]

        with patch("supervisors.v3.tools.product_loader.load_products_from_rdb",
                   return_value=("아세리스,리리카", 2)), \
             patch("supervisors.v3.tools.product_loader.load_hospitals_from_rdb",
                   return_value=("스마트정형외과의원,한국병원", 2)), \
             patch("supervisors.v3.supervisor_v3.HistoryManager", return_value=hm_mock), \
             patch("supervisors.v3.supervisor_v3.convert_to_messages",
                   side_effect=fake_convert):
            await sv._prepare_request_context(
                history=[{"role": "User", "content": "질문"}, {"role": "AI", "content": "답변"}],
                user_input="새 질문",
                employee_ID="EMP001",
            )

        # 첫 번째 convert_to_messages 호출 (history normalize)
        if captured:
            for msg in captured[:2]:
                self.assertEqual(msg["role"], msg["role"].lower())

    async def test_context_enhancer_receives_history(self):
        """ContextEnhancer가 history_messages를 받는지 확인"""
        from supervisors.v3.supervisor_v3 import SupervisorV3
        sv, hm_mock = self._make_supervisor_with_mocks()

        with patch("supervisors.v3.tools.product_loader.load_products_from_rdb",
                   return_value=("아세리스,리리카", 2)), \
             patch("supervisors.v3.tools.product_loader.load_hospitals_from_rdb",
                   return_value=("스마트정형외과의원,한국병원", 2)), \
             patch("supervisors.v3.supervisor_v3.HistoryManager", return_value=hm_mock), \
             patch("supervisors.v3.supervisor_v3.convert_to_messages",
                   side_effect=lambda msgs: [HumanMessage(content=m["content"]) for m in msgs]):
            await sv._prepare_request_context(
                history=[{"role": "user", "content": "이전 질문"}],
                user_input="새 질문",
                employee_ID="EMP001",
            )

        call_kwargs = sv._context_enhancer.call_args.kwargs
        self.assertIn("history", call_kwargs)

    async def test_analyzed_question_in_final_message(self):
        """analyzed_question이 최종 all_messages의 마지막 메시지에 포함"""
        from supervisors.v3.supervisor_v3 import SupervisorV3
        sv, hm_mock = self._make_supervisor_with_mocks()
        sv._context_enhancer.return_value["analyzed_question"] = "분석된 질문 내용"
        hm_mock.get_history.return_value = []

        with patch("supervisors.v3.tools.product_loader.load_products_from_rdb",
                   return_value=("아세리스,리리카", 2)), \
             patch("supervisors.v3.tools.product_loader.load_hospitals_from_rdb",
                   return_value=("스마트정형외과의원,한국병원", 2)), \
             patch("supervisors.v3.supervisor_v3.HistoryManager", return_value=hm_mock), \
             patch("supervisors.v3.supervisor_v3.convert_to_messages",
                   side_effect=lambda msgs: [HumanMessage(content=m["content"]) for m in msgs]):
            result = await sv._prepare_request_context(
                history=[],
                user_input="원본 질문",
                employee_ID="EMP001",
            )

        last_msg = result["all_messages"][-1]
        # _build_enhanced_input을 통해 final_input이 구성됨
        self.assertIn("원본 질문", last_msg.content)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실행 진입점
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUITE_MAP = {
    "rerank": [
        TestLLMRerankerWeightedScore,
        TestLLMRerankerRerank,
    ],
    "history": [
        TestHistoryManagerTokenCount,
        TestHistoryManagerBuild,
        TestHistoryManagerSummarize,
    ],
    "context": [
        TestContextEnhancerInit,
        TestContextEnhancerDetection,
        TestContextEnhancerAnalyzeQuestion,
        TestContextEnhancerCall,
    ],
    "prepare": [
        TestBuildEnhancedInput,
        TestPrepareRequestContext,
    ],
}


def run_suites(names: list[str]) -> bool:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for name in names:
        for cls in SUITE_MAP.get(name, []):
            suite.addTests(loader.loadTestsFromTestCase(cls))

    print(f"\n{'=' * 70}")
    print(f"  TOTAL TEST  |  suites: {', '.join(names)}")
    print(f"{'=' * 70}\n")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if arg == "all":
        targets = list(SUITE_MAP.keys())
    elif arg in SUITE_MAP:
        targets = [arg]
    else:
        print(f"알 수 없는 인자: {arg}")
        print(f"사용법: venv/Scripts/python total_test.py [{'|'.join(SUITE_MAP)} | all]")
        sys.exit(1)

    ok = run_suites(targets)
    sys.exit(0 if ok else 1)
