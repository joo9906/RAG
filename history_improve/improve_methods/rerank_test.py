"""
LLMReranker 동작 검증 테스트
==============================

목적:
    - 0~3 정수 척도 변경 후 점수 계산/필터링/정렬이 올바른지 검증
    - 프롬프트 로드 → 점수 계산 → Mock LLM → 실제 LLM 순서로 계층화
    - 실제 채팅 시나리오 5문서 상세 평가 데모

실행 방법 (프로젝트 루트에서):
    python improve_methods/rerank_test.py          # 전체 (통합 + 데모 포함)
    python improve_methods/rerank_test.py unit     # 단위/프롬프트만 (API 불필요)
    python improve_methods/rerank_test.py demo     # 데모만 실행
"""
# isort:imports-stdlib
import copy
import logging
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# ── 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
for _p in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# isort:imports-thirdparty
from langchain_core.documents import Document

# ── 외부 라이브러리 로그 억제 (httpx 요청/응답 노이즈 제거)
for _noisy in ("httpx", "httpcore", "openai", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 공통 헬퍼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_doc(content: str) -> Document:
    return Document(page_content=content, metadata={})


def make_mock_response(relevance: int, specificity: int = 0,
                       completeness: int = 0, practicality: int = 0,
                       constraint_fit: int = 0) -> MagicMock:
    m = MagicMock()
    m.relevance = relevance
    m.specificity = specificity
    m.completeness = completeness
    m.practicality = practicality
    m.constraint_fit = constraint_fit
    return m


# 가중치 정보 (reranker.py WEIGHTS와 동일 순서)
_SCORE_FIELDS = [
    ("relevance_score",     0.60, "relevance    "),
    ("specificity_score",   0.15, "specificity  "),
    ("completeness_score",  0.10, "completeness "),
    ("practicality_score",  0.10, "practicality "),
    ("constraint_fit_score",0.05, "constraint   "),
]


def _bar(score: int, max_score: int = 10, width: int = 12) -> str:
    """점수를 블록 바 문자열로 변환. score/max_score 비율로 채움."""
    filled = round(score / max_score * width)
    return "#" * filled + "." * (width - filled)


def print_rerank_detail(
    query: str,
    search_query: str,
    all_docs: list,
    result: list,
    elapsed: float,
    threshold: float,
    model: str = "gpt-4.1-mini",
) -> None:
    """
    rerank 결과를 문서별 상세 점수와 함께 출력한다.

    Parameters
    ----------
    all_docs : rerank()에 넘긴 문서 리스트.
               _score_document()가 in-place로 sub-score를 metadata에 기록하므로
               호출 후에도 필터링된 문서의 점수를 읽을 수 있다.
    result   : rerank()가 반환한 통과 문서 리스트 (순위순).
               rank, overall_score 메타데이터가 추가된 상태.
    """
    W = 74
    result_by_content = {doc.page_content: doc for doc in result}

    print()
    print("=" * W)
    print("  RERANKER DETAIL")
    print(f"  query  : {query}")
    if search_query != query:
        print(f"  search : {search_query}")
    print(f"  model  : {model}  |  threshold={threshold}  |  elapsed={elapsed:.2f}s")
    print(f"  result : {len(all_docs)} in -> {len(result)} PASS / "
          f"{len(all_docs) - len(result)} FILTERED")
    print("=" * W)

    for i, doc in enumerate(all_docs):
        meta = doc.metadata
        r_doc = result_by_content.get(doc.page_content)   # None if filtered
        label = meta.get("label", "")

        if r_doc is not None:
            rank    = r_doc.metadata["rank"]
            overall = r_doc.metadata["overall_score"]
            status  = f"[PASS]  rank={rank}  score={overall:.3f}"
        elif "relevance_score" in meta:
            # 필터링됐지만 _score_document가 점수를 채운 경우
            total_w = sum(w for _, w, _ in _SCORE_FIELDS)
            overall = sum((meta.get(k, 0) / 3.0) * w
                          for k, w, _ in _SCORE_FIELDS) / total_w
            status  = f"[FILTERED]  score={overall:.3f} (< {threshold})"
        else:
            status  = "[FILTERED]  (timeout or LLM error)"

        print()
        print(f"  doc {i+1}  {status}  {label}")

        # 문서 내용 미리보기
        preview = doc.page_content.replace("\n", " | ")[:150]
        if len(doc.page_content) > 150:
            preview += "..."
        print(f"  {'-' * (W - 4)}")
        print(f"  {preview}")
        print()

        # 점수 바 — sub-score가 있는 경우에만 출력
        score_src = r_doc.metadata if r_doc else meta
        if "relevance_score" in score_src:
            for field, weight, lbl in _SCORE_FIELDS:
                score = score_src.get(field, 0)
                bar   = _bar(score)
                pct   = f"{weight*100:.0f}%"
                print(f"    {lbl}  {bar}  {score}/10  (weight {pct:>4})")
        else:
            print("    (no scores)")

    print()
    print("-" * W)
    print(f"  total: {elapsed:.2f}s  ({len(all_docs)} docs parallel)")
    print("=" * W)
    print()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Suite 1: 단위 테스트 — Mock LLM, 외부 의존 없음
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRerankUnit(unittest.TestCase):
    """LLMReranker 단위 테스트 — API 호출 없음."""

    @classmethod
    def setUpClass(cls):
        from supervisors.v3.tools.reranker import LLMReranker
        cls.LLMReranker = LLMReranker

    def _make_reranker(self, threshold: float = 0.3, prompt: str = None):
        r = self.LLMReranker.__new__(self.LLMReranker)
        r.threshold = threshold
        r.llm = MagicMock()
        r.prompt_manager = MagicMock()
        r.rerank_prompt = prompt
        return r

    def test_weighted_score_all_max(self):
        """모든 기준 3 → 최종 점수 1.0."""
        r = self._make_reranker()
        scores = {k: 3 for k, _, _ in _SCORE_FIELDS}
        self.assertAlmostEqual(r._calculate_weighted_score(scores), 1.0, places=5)

    def test_weighted_score_all_zero(self):
        """모든 기준 0 → 최종 점수 0.0."""
        r = self._make_reranker()
        scores = {k: 0 for k, _, _ in _SCORE_FIELDS}
        self.assertAlmostEqual(r._calculate_weighted_score(scores), 0.0, places=5)

    def test_weighted_score_relevance_only(self):
        """relevance=3, 나머지 0 → 0.60."""
        r = self._make_reranker()
        scores = {k: 0 for k, _, _ in _SCORE_FIELDS}
        scores["relevance_score"] = 3
        self.assertAlmostEqual(r._calculate_weighted_score(scores), 0.60, places=5)

    def test_scale_boundaries(self):
        """0~3 척도 경계값이 정확히 변환되는지 확인."""
        r = self._make_reranker()
        cases = {
            (0, 0, 0, 0, 0): 0.0,
            (1, 0, 0, 0, 0): 0.60 / 3,
            (2, 0, 0, 0, 0): 0.60 * 2 / 3,
            (3, 0, 0, 0, 0): 0.60,
            (3, 3, 3, 3, 3): 1.0,
        }
        for (rv, sp, co, pr, cf), expected in cases.items():
            with self.subTest(relevance=rv):
                scores = {
                    "relevance_score": rv, "specificity_score": sp,
                    "completeness_score": co, "practicality_score": pr,
                    "constraint_fit_score": cf,
                }
                self.assertAlmostEqual(r._calculate_weighted_score(scores), expected, places=5)

    def test_score_always_in_0_to_1_range(self):
        """임의 조합에서 점수는 항상 [0.0, 1.0] 범위."""
        import random; random.seed(42)
        r = self._make_reranker()
        for _ in range(50):
            scores = {k: random.randint(0, 3) for k, _, _ in _SCORE_FIELDS}
            result = r._calculate_weighted_score(scores)
            self.assertGreaterEqual(result, 0.0)
            self.assertLessEqual(result, 1.0 + 1e-9)

    def test_no_prompt_returns_original_docs(self):
        """rerank_prompt=None 이면 원본 문서를 그대로 반환해야 함."""
        r = self._make_reranker(prompt=None)
        docs = [make_doc("문서 A"), make_doc("문서 B")]
        self.assertEqual(r.rerank(docs, user_query="테스트"), docs)

    def test_empty_docs_returns_empty(self):
        r = self._make_reranker(prompt="{user_query} {search_query} {document}")
        self.assertEqual(r.rerank([], user_query="테스트"), [])

    def test_filters_below_threshold(self):
        """threshold=0.5: RELEVANT(0.60)는 통과, 나머지(0.0)는 제거."""
        r = self._make_reranker(threshold=0.5,
                                prompt="{user_query} {search_query} {document}")

        def fake_invoke(prompt):
            return make_mock_response(3) if "RELEVANT" in prompt else make_mock_response(0)

        r.llm.invoke.side_effect = fake_invoke
        docs = [make_doc("RELEVANT: 기넥신 부작용 정보"), make_doc("무관한 학술대회 일정")]
        result = r.rerank(docs, user_query="기넥신 부작용")

        self.assertEqual(len(result), 1)
        self.assertIn("RELEVANT", result[0].page_content)

    def test_sorts_by_score_descending(self):
        """높은 점수 문서가 낮은 점수 문서보다 앞에 정렬되어야 함."""
        r = self._make_reranker(threshold=0.0,
                                prompt="{user_query} {search_query} {document}")

        def fake_invoke(prompt):
            return make_mock_response(3) if "HIGH" in prompt else make_mock_response(1)

        r.llm.invoke.side_effect = fake_invoke
        docs = [make_doc("LOW score 무관 문서"), make_doc("HIGH score 핵심 문서")]
        result = r.rerank(docs, user_query="테스트")

        self.assertEqual(result[0].page_content, "HIGH score 핵심 문서")
        self.assertEqual(result[1].page_content, "LOW score 무관 문서")

    def test_rank_and_score_metadata_added(self):
        """rerank 후 rank, overall_score 메타데이터가 순서대로 추가되어야 함."""
        r = self._make_reranker(threshold=0.0,
                                prompt="{user_query} {search_query} {document}")
        r.llm.invoke.return_value = make_mock_response(2, 1, 1, 1, 0)
        result = r.rerank([make_doc("문서 A"), make_doc("문서 B")], user_query="테스트")
        for i, doc in enumerate(result):
            self.assertIn("rank", doc.metadata)
            self.assertIn("overall_score", doc.metadata)
            self.assertEqual(doc.metadata["rank"], i + 1)

    def test_search_query_defaults_to_user_query(self):
        """search_query 미제공 시 user_query가 두 플레이스홀더에 모두 삽입되어야 함."""
        r = self._make_reranker(threshold=0.0,
                                prompt="{user_query} {search_query} {document}")
        r.llm.invoke.return_value = make_mock_response(2)
        r.rerank([make_doc("테스트 문서")], user_query="기넥신 부작용")
        called_prompt = r.llm.invoke.call_args[0][0]
        self.assertEqual(called_prompt.count("기넥신 부작용"), 2)

    def test_subscores_stored_in_metadata(self):
        """5가지 서브점수가 모두 metadata에 저장되어야 함."""
        r = self._make_reranker(threshold=0.0,
                                prompt="{user_query} {search_query} {document}")
        r.llm.invoke.return_value = make_mock_response(2, 2, 1, 1, 1)
        result = r.rerank([make_doc("문서")], user_query="테스트")
        self.assertEqual(len(result), 1)
        for field, _, _ in _SCORE_FIELDS:
            self.assertIn(field, result[0].metadata)
            self.assertGreaterEqual(result[0].metadata[field], 0)
            self.assertLessEqual(result[0].metadata[field], 3)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Suite 2: 프롬프트 파일 검증
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRerankPromptLoad(unittest.TestCase):
    """프롬프트 파일 내용 및 _load_rerank_prompt() 동작 검증."""

    PROMPT_PATH = (
        PROJECT_ROOT / "src" / "supervisors" / "v3"
        / "prompts" / "tools" / "reranker_instruction.txt"
    )

    def test_prompt_file_exists(self):
        self.assertTrue(self.PROMPT_PATH.exists(), f"파일 없음: {self.PROMPT_PATH}")

    def test_prompt_has_required_placeholders(self):
        content = self.PROMPT_PATH.read_text(encoding="utf-8")
        for ph in ["{user_query}", "{search_query}", "{document}"]:
            self.assertIn(ph, content, f"플레이스홀더 없음: {ph}")

    def test_prompt_mentions_all_criteria(self):
        content = self.PROMPT_PATH.read_text(encoding="utf-8").lower()
        for kw in ["relevance", "specificity", "completeness", "practicality", "constraint"]:
            self.assertIn(kw, content, f"평가 기준 없음: {kw}")

    def test_prompt_score_scale_is_0_to_3(self):
        content = self.PROMPT_PATH.read_text(encoding="utf-8")
        has_scale = any(s in content for s in ["0, 1, 2, or 3", "0~3", "0 to 3"])
        self.assertTrue(has_scale, "프롬프트에 0~3 척도가 명시되지 않음")

    def test_prompt_format_no_error(self):
        content = self.PROMPT_PATH.read_text(encoding="utf-8")
        try:
            formatted = content.format(
                user_query="기넥신 이상반응은?",
                search_query="기넥신 부작용",
                document="기넥신은 은행잎 추출물입니다."
            )
            self.assertGreater(len(formatted), 0)
        except KeyError as e:
            self.fail(f"format() 실패 — 알 수 없는 플레이스홀더: {e}")

    def test_load_rerank_prompt_sets_attribute(self):
        """_load_rerank_prompt() 호출 후 rerank_prompt가 비어 있지 않아야 함."""
        from supervisors.v3.tools.reranker import LLMReranker
        mock_pm = MagicMock()
        mock_pm.load_prompt_by_path.return_value = (
            self.PROMPT_PATH.read_text(encoding="utf-8")
            if self.PROMPT_PATH.exists()
            else "dummy {user_query} {search_query} {document}"
        )
        reranker = LLMReranker.__new__(LLMReranker)
        reranker.threshold = 0.3
        reranker.llm = MagicMock()
        reranker.prompt_manager = mock_pm
        reranker.rerank_prompt = None
        reranker._load_rerank_prompt()

        self.assertIsNotNone(reranker.rerank_prompt,
                             "rerank_prompt가 None — load_prompt_by_path 반환값이 할당되는지 확인")
        self.assertIsInstance(reranker.rerank_prompt, str)
        self.assertGreater(len(reranker.rerank_prompt), 0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Suite 3: 통합 테스트 — 실제 LLM API 호출
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRerankLLMIntegration(unittest.TestCase):
    """실제 gpt-4.1-mini를 사용하는 통합 테스트."""

    @classmethod
    def setUpClass(cls):
        try:
            from supervisors.v3.tools.reranker import LLMReranker
            cls.reranker = LLMReranker(threshold=0.3)
            cls.skip_reason = (
                None if cls.reranker.rerank_prompt
                else "rerank_prompt=None: 프롬프트 파일 로드 실패"
            )
        except Exception as e:
            cls.reranker = None
            cls.skip_reason = f"LLMReranker 초기화 실패: {e}"

    def setUp(self):
        if self.skip_reason:
            self.skipTest(self.skip_reason)

    def test_relevant_beats_irrelevant(self):
        """관련 문서가 무관한 문서보다 높은 점수를 받아야 함."""
        query = "기넥신 부작용 알려줘"
        docs = [
            make_doc("기넥신(EGb761) 주요 부작용: 두통, 소화불량, 드물게 알레르기 반응."),
            make_doc("오늘 서울 날씨: 맑음, 최고기온 25도, 미세먼지 보통."),
        ]
        result = self.reranker.rerank(copy.deepcopy(docs), user_query=query)

        if len(result) < 2:
            self.assertEqual(len(result), 1, "관련 문서 하나는 반드시 통과해야 함")
            self.assertNotIn("날씨", result[0].page_content)
            return

        self.assertNotIn("날씨", result[0].page_content,
                         "날씨 문서가 기넥신 문서보다 높은 순위면 안 됨")

    def test_scores_in_valid_range(self):
        docs = [
            make_doc("기넥신 처방 건수: 43건, 매출: 215만원"),
            make_doc("리바로 고지혈증 치료제 처방 정보"),
        ]
        result = self.reranker.rerank(copy.deepcopy(docs), user_query="기넥신 매출")
        for doc in result:
            score = doc.metadata.get("overall_score", 0.0)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0 + 1e-9)

    def test_parallel_5_docs_under_30s(self):
        """5개 문서 병렬 처리가 30초 내 완료되어야 함."""
        docs = [
            make_doc(f"기넥신 관련 문서 {i}: 혈액순환 개선 임상 결과.")
            for i in range(5)
        ]
        start = time.perf_counter()
        self.reranker.rerank(copy.deepcopy(docs), user_query="기넥신 효능")
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 30.0,
                        f"5건 reranking이 {elapsed:.1f}s로 30s 초과")

    def test_rank_sequential_and_subscores_valid(self):
        docs = [
            make_doc("기넥신 처방 및 매출 현황: 43건, 215만원"),
            make_doc("경쟁 제품 타나민 시장 점유율 데이터"),
            make_doc("전혀 무관한 내용: 날씨 정보"),
        ]
        result = self.reranker.rerank(copy.deepcopy(docs), user_query="기넥신 이번달 처방")
        if not result:
            self.skipTest("모든 문서가 threshold 미달")

        ranks = [doc.metadata.get("rank") for doc in result]
        self.assertEqual(ranks, list(range(1, len(result) + 1)))

        for doc in result:
            for field, _, _ in _SCORE_FIELDS:
                val = doc.metadata.get(field)
                self.assertIsNotNone(val)
                self.assertGreaterEqual(val, 0)
                self.assertLessEqual(val, 3)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Suite 4: 5문서 데모 — 실제 채팅 시나리오 상세 출력
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRerankLLMDemo(unittest.TestCase):
    """
    실제 채팅 시나리오를 재현하는 5문서 데모.
    각 문서의 LLM 평가 점수와 소요 시간을 상세 출력한다.
    """

    USER_QUERY = "스마트정형외과의원 기넥신 이번 달 처방 건수와 매출 알려줘"
    SEARCH_QUERY = "스마트정형외과의원 기넥신 2025년 4월 처방 건수 매출"

    # 실제 VDB/RDB에서 검색될 법한 5가지 문서
    DOCS = [
        Document(
            page_content=(
                "스마트정형외과의원 2025년 4월 기넥신 처방 현황\n"
                "- 처방 건수: 43건\n"
                "- 매출: 2,150,000원\n"
                "- 전월 대비: +5건 / +250,000원\n"
                "담당 MR: 김철수 | 최근 방문: 2025-04-10"
            ),
            metadata={"source": "RDB", "label": "핵심 — 직접 데이터"},
        ),
        Document(
            page_content=(
                "기넥신 제품 정보\n"
                "성분: 은행잎엑스(EGb 761) 40mg\n"
                "적응증: 말초순환장애, 뇌혈관순환장애에 의한 인지 및 신경 증상\n"
                "용법·용량: 1회 1정, 1일 3회 식후 복용\n"
                "주의: 항응고제 병용 시 출혈 위험 증가. 보험코드: A02BC01"
            ),
            metadata={"source": "VDB", "label": "부분 관련 — 제품 정보"},
        ),
        Document(
            page_content=(
                "스마트정형외과의원 2025년 4월 리바로 처방 현황\n"
                "- 처방 건수: 28건\n"
                "- 매출: 1,960,000원\n"
                "리바로(피타바스타틴)는 고지혈증 치료제로 기넥신과 별개 경로로 처방됨."
            ),
            metadata={"source": "RDB", "label": "부분 관련 — 다른 제품"},
        ),
        Document(
            page_content=(
                "2025년 1분기 국내 말초순환장애 치료제 시장 리포트\n"
                "총 시장 규모: 580억 원 (전년 대비 +3.2%)\n"
                "주요 점유율: 기넥신 22%, 타나민 18%, 징코민 15%\n"
                "처방 트렌드: 고령 인구 증가에 따라 증가세 지속\n"
                "출처: IMS Health 2025 Q1"
            ),
            metadata={"source": "VDB", "label": "간접 관련 — 시장 리포트"},
        ),
        Document(
            page_content=(
                "2025년 4월 주요 학술대회 일정\n"
                "- 4월 18일: 대한내과학회 춘계학술대회 (서울 코엑스)\n"
                "- 4월 25일: 대한정형외과학회 정기학술대회 (부산 벡스코)\n"
                "참가 신청 마감: 4월 10일 / 부스 운영: 09:00~18:00"
            ),
            metadata={"source": "VDB", "label": "무관 — 학술대회 일정"},
        ),
    ]

    @classmethod
    def setUpClass(cls):
        try:
            from supervisors.v3.tools.reranker import LLMReranker
            cls.reranker = LLMReranker(threshold=0.3)
            cls.skip_reason = (
                None if cls.reranker.rerank_prompt
                else "rerank_prompt=None: 프롬프트 파일 로드 실패"
            )
        except Exception as e:
            cls.reranker = None
            cls.skip_reason = f"LLMReranker 초기화 실패: {e}"

    def setUp(self):
        if self.skip_reason:
            self.skipTest(self.skip_reason)

    def test_5_doc_rerank_scenario(self):
        """
        실제 채팅 흐름 재현:
          사용자 쿼리 → VDB/RDB 5문서 검색 → LLMReranker 평가 → 상세 결과 출력
        """
        docs = copy.deepcopy(self.DOCS)

        print("\n" + "=" * 74)
        print("  [DEMO] 5문서 rerank 시작...")
        print(f"  쿼리: {self.USER_QUERY}")
        print("=" * 74)

        start = time.perf_counter()
        result = self.reranker.rerank(
            docs,
            user_query=self.USER_QUERY,
            search_query=self.SEARCH_QUERY,
        )
        elapsed = time.perf_counter() - start

        # 상세 출력
        print_rerank_detail(
            query=self.USER_QUERY,
            search_query=self.SEARCH_QUERY,
            all_docs=docs,
            result=result,
            elapsed=elapsed,
            threshold=self.reranker.threshold,
        )

        # 검증
        self.assertGreater(len(result), 0, "통과 문서가 0개 — threshold를 낮추거나 문서를 확인하세요")
        self.assertLess(elapsed, 30.0, f"30s 초과: {elapsed:.1f}s")

        # doc_1(처방 직접 데이터)이 1위여야 함
        self.assertEqual(
            result[0].metadata.get("label"), "핵심 — 직접 데이터",
            f"처방 직접 데이터가 1위여야 함. 실제 1위: {result[0].metadata.get('label')}"
        )

        # doc_5(학술대회)는 필터링되어야 함
        passed_labels = [d.metadata.get("label") for d in result]
        self.assertNotIn("무관 — 학술대회 일정", passed_labels,
                         "학술대회 일정이 처방/매출 쿼리를 통과하면 안 됨")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실행 진입점
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    # Windows cp949 터미널에서 한글 깨짐 방지
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if mode == "demo":
        # 데모만 실행
        suite.addTests(loader.loadTestsFromTestCase(TestRerankLLMDemo))
    elif mode == "unit":
        # 단위 + 프롬프트만 (API 불필요)
        suite.addTests(loader.loadTestsFromTestCase(TestRerankUnit))
        suite.addTests(loader.loadTestsFromTestCase(TestRerankPromptLoad))
        print("[INFO] 'unit' 모드 → Suite 3·4(LLM 호출) 건너뜀\n")
    else:
        # 전체 실행 (단위 → 프롬프트 → 통합 → 데모 순)
        suite.addTests(loader.loadTestsFromTestCase(TestRerankUnit))
        suite.addTests(loader.loadTestsFromTestCase(TestRerankPromptLoad))
        suite.addTests(loader.loadTestsFromTestCase(TestRerankLLMIntegration))
        suite.addTests(loader.loadTestsFromTestCase(TestRerankLLMDemo))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    test_result = runner.run(suite)
    sys.exit(0 if test_result.wasSuccessful() else 1)
