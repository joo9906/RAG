"""
HistoryManager 단위 테스트
외부 의존성(LLM, Redis)을 unittest.mock으로 격리한 단위 테스트.

실행:
    pytest history_test.py -v
"""

# isort:imports-stdlib
import sys
from unittest.mock import AsyncMock, MagicMock, patch

# isort:imports-thirdparty
import pytest
import tiktoken

sys.path.insert(0, "src")

# isort:imports-firstparty
from langchain_core.messages import AIMessage, HumanMessage

import supervisors.v3.history_manager as hm_module
from supervisors.v3.history_manager import HistoryManager

# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------
_enc = tiktoken.get_encoding("cl100k_base")

_CONTENT = "hello " * 10
_C_TOK = len(_enc.encode(_CONTENT))  # content 토큰 수
_M_TOK = _C_TOK + 4                  # 메시지당 총 토큰 (content + 4 오버헤드)

# 테스트 임계값 (실제 120K/80K 대신 소규모 값으로 동작 검증)
_MAX = _M_TOK * 8          # 9개 이상 시 초과
_KEEP = _M_TOK * 4         # 뒤에서 4개 메시지 보존 예산
_KEEP_CORR = _M_TOK * 3 + 1  # TC-03 전용: AI 타입에서 경계가 끊기도록 설정
_MIN = 2
_TMOUT = 1.0


def _model(text="[요약 결과]"):
    """고정된 요약 텍스트를 반환하는 Mock LLM."""
    resp = MagicMock()
    resp.content = text
    m = MagicMock()
    m.ainvoke = AsyncMock(return_value=resp)
    return m


def _msgs(n_pairs: int):
    """n_pairs 쌍의 HumanMessage + AIMessage 생성."""
    result = []
    for _ in range(n_pairs):
        result += [HumanMessage(content=_CONTENT), AIMessage(content=_CONTENT)]
    return result


@pytest.fixture
def anyio_backend():
    """trio가 설치되지 않은 환경에서 asyncio 백엔드만 사용한다."""
    return "asyncio"


def _patch(**overrides):
    """모듈 상수를 일괄 패치한다. 키는 hm_module의 변수명으로 지정한다."""
    base = dict(
        MAX_HISTORY_TOKENS=_MAX,
        KEEP_TOKENS_BUDGET=_KEEP,
        MIN_KEEP_MESSAGES=_MIN,
        SUMMARIZE_TIMEOUT=_TMOUT,
    )
    base.update(overrides)
    return patch.multiple("supervisors.v3.history_manager", **base)


# ---------------------------------------------------------------------------
# TC-01. 토큰 미초과 → 요약 미발생
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_tc01_no_summarize_when_under_budget():
    """토큰 합계가 MAX 미만이면 _summarize()가 호출되지 않는다."""
    model = _model()
    msgs = _msgs(3)  # 6개 × _M_TOK < 8 × _M_TOK = _MAX

    with _patch():
        mgr = HistoryManager(model=model)
        await mgr.build(msgs)

    model.ainvoke.assert_not_called()
    assert mgr.summary == ""
    assert mgr.get_history() == msgs


# ---------------------------------------------------------------------------
# TC-02. 토큰 초과 → 요약 발생 및 SystemMessage prepend
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_tc02_summarize_and_system_message_prepended():
    """토큰 초과 시 _summarize()가 호출되고 get_history() 첫 원소가 SystemMessage다."""
    model = _model("[요약 결과]")
    msgs = _msgs(5)  # 10개 × _M_TOK > 8 × _M_TOK = _MAX

    with _patch():
        mgr = HistoryManager(model=model)
        await mgr.build(msgs)

    model.ainvoke.assert_called_once()
    assert mgr.summary == "[요약 결과]"

    history = mgr.get_history()
    assert history[0].type == "system"
    assert "[이전 대화 요약]" in history[0].content
    assert "[요약 결과]" in history[0].content


# ---------------------------------------------------------------------------
# TC-03. user/assistant 쌍 경계 보정
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_tc03_pair_boundary_correction():
    """
    KEEP_TOKENS_BUDGET 역방향 탐색이 AI 타입 메시지에서 끊길 때
    split_idx를 앞 Human 메시지 위치로 보정한다.

    KEEP_CORR = 3×_M_TOK + 1 설정 시:
      뒤에서 3개(A4, H4, A3)를 채운 후 H3 추가 시 예산 초과 → break.
      split_idx = A3 인덱스(AI 타입) → Human 앞으로 보정 → H3 인덱스.
    """
    model = _model()
    msgs = _msgs(5)  # [H0,A0,H1,A1,H2,A2,H3,A3,H4,A4]

    with _patch(KEEP_TOKENS_BUDGET=_KEEP_CORR):
        mgr = HistoryManager(model=model)
        await mgr.build(msgs)

    # 보존된 메시지가 항상 Human 타입으로 시작해야 한다
    assert mgr.recent_messages[0].type == "human", (
        f"recent_messages[0].type = {mgr.recent_messages[0].type!r}, expected 'human'"
    )
    # 요약 대상의 마지막 메시지는 AI 타입이어야 한다 (완전한 쌍)
    assert model.ainvoke.called


# ---------------------------------------------------------------------------
# TC-04. MIN_KEEP_MESSAGES 하한 보장
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_tc04_min_keep_messages_respected():
    """
    KEEP_TOKENS_BUDGET이 메시지 1개분밖에 안 되더라도
    recent_messages에 MIN_KEEP_MESSAGES(2)개 이상 남아야 한다.
    """
    model = _model()
    msgs = _msgs(5)  # 10개 × _M_TOK > _MAX → 요약 발생

    with _patch(KEEP_TOKENS_BUDGET=_M_TOK):  # 예산 = 정확히 1개 분량
        mgr = HistoryManager(model=model)
        await mgr.build(msgs)

    assert len(mgr.recent_messages) >= _MIN, (
        f"recent_messages 수({len(mgr.recent_messages)}) < MIN_KEEP_MESSAGES({_MIN})"
    )


# ---------------------------------------------------------------------------
# TC-05. 요약 타임아웃 → 기존 summary 유지
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_tc05_timeout_keeps_existing_summary():
    """ainvoke가 TimeoutError를 발생시키면 기존 summary가 그대로 유지된다."""
    import asyncio

    model = MagicMock()
    model.ainvoke = AsyncMock(side_effect=asyncio.TimeoutError())

    msgs = _msgs(5)

    with _patch(SUMMARIZE_TIMEOUT=0.001):
        mgr = HistoryManager(model=model)
        mgr.summary = "기존 요약"
        await mgr.build(msgs)

    assert mgr.summary == "기존 요약"


# ---------------------------------------------------------------------------
# TC-06. 요약 API 실패 → 기존 summary 유지
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_tc06_api_error_keeps_existing_summary():
    """ainvoke가 RuntimeError를 발생시키면 기존 summary가 그대로 유지된다."""
    model = MagicMock()
    model.ainvoke = AsyncMock(side_effect=RuntimeError("API error"))

    msgs = _msgs(5)

    with _patch():
        mgr = HistoryManager(model=model)
        mgr.summary = "기존 요약"
        await mgr.build(msgs)

    assert mgr.summary == "기존 요약"


# ---------------------------------------------------------------------------
# TC-07. Rolling Summary → 통합 요약 헤더 포함 여부
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_tc07_rolling_summary_includes_integration_header():
    """기존 summary가 있을 때 LLM에 전달되는 프롬프트에 '[통합 요약]' 지시가 포함된다."""
    model = _model("[통합 요약] 두 번째 요약")
    msgs = _msgs(5)

    with _patch():
        mgr = HistoryManager(model=model)
        mgr.summary = "1차 요약본"
        await mgr.build(msgs)

    model.ainvoke.assert_called_once()
    prompt_content = model.ainvoke.call_args[0][0][0].content
    assert "1차 요약본" in prompt_content, "이전 요약이 프롬프트에 포함되어야 한다"
    assert "[통합 요약]" in prompt_content, "통합 요약 지시가 프롬프트에 포함되어야 한다"
    assert mgr.summary == "[통합 요약] 두 번째 요약"


# ---------------------------------------------------------------------------
# TC-08. split_idx <= 0 → 경고 후 기존 상태 유지
# ---------------------------------------------------------------------------
@pytest.mark.anyio
async def test_tc08_no_summarizable_messages_when_only_min_keep():
    """
    메시지 수가 MIN_KEEP_MESSAGES와 같으면 _summarize()가 호출되지 않는다.

    2개 메시지(1쌍), KEEP 예산이 전체보다 크면:
      역방향 탐색이 모두 성공 → split_idx = 0
      → _summarize_history()에서 split_idx <= 0 감지 → 경고 후 반환.
    """
    model = _model()
    msgs = _msgs(1)  # [H0, A0] — 2개
    total = _M_TOK * 2

    with _patch(
        MAX_HISTORY_TOKENS=total - 1,   # 2개면 초과
        KEEP_TOKENS_BUDGET=total + 100,  # 예산이 전체보다 커서 모두 fit
    ):
        mgr = HistoryManager(model=model)
        await mgr.build(msgs)

    model.ainvoke.assert_not_called()
    assert mgr.summary == ""
    assert len(mgr.recent_messages) == 2
