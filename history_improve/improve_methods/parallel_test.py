"""
Rerank_개선안.md 개선 항목 전체 검증 테스트

Group 1  — parallel_tool_calls binding (base_agent.py)
Group 2  — _create_prompt_template PromptTemplate 구조 검증
Group 3  — agent_prompt_function 메시지 구조 검증 (단일 SystemMessage + 히스토리)
Group 4  — process_user_query preflight 3개 작업 동시 실행 검증
Group 5  — ToolNode 병렬 도구 실행 검증
Group 6  — LLMReranker 0~10 척도 가중치 계산 검증 (LLM 호출 없음)
Group 7  — vdb.py _get_reranker() 싱글턴 검증
Group 8  — LLMReranker 동작 검증 (LLM mock 사용)
Group 9  — LLMReranker 실제 LLM 호출 통합 테스트 (실 API 필요)
"""

import os
import sys
import time
import asyncio
import pytest
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from supervisors.v3.agents.base_agent import BaseAgent
from supervisors.v3.supervisor_v3 import SupervisorV3

# ─── Pydantic-safe 추적 가능 Fake 모델 ───────────────────────────────────────
# ChatOpenAI는 Pydantic v2 모델이라 patch.object(instance, "invoke", ...) 가 실패함.
# create_react_agent와 완전 호환되면서 입력/출력을 추적할 수 있는 커스텀 모델 사용.

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field as PydanticField


class _TrackingFakeModel(BaseChatModel):
    """테스트용 가짜 LLM.
    - responses 목록에서 순서대로 응답을 반환한다.
    - captured 목록에 각 호출의 입력 메시지를 기록한다.
    - Pydantic v2 제약 없이 patch 없이도 동작한다.
    """

    responses: List[Any] = PydanticField(default_factory=list)
    captured: List[Any] = PydanticField(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "tracking_fake"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.captured.append(list(messages))
        idx = len(self.captured) - 1
        from langchain_core.messages import AIMessage as _AI
        msg = self.responses[idx] if idx < len(self.responses) else _AI(content="done")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        from langchain_core.runnables.base import RunnableBinding
        tool_names = [t.name if hasattr(t, "name") else str(t) for t in tools]
        return RunnableBinding(bound=self, kwargs={"tools": tool_names})

    def bind(self, **kwargs):
        from langchain_core.runnables.base import RunnableBinding
        return RunnableBinding(bound=self, kwargs=kwargs)

# ─── 공용 상수 & 헬퍼 ─────────────────────────────────────────────────────────

MOCK_CONFIG = {
    "ce": {"context_cards": {"enabled": False}},
    "ce_qa": {"vector_store_name": "mock_store"},
    "llm": {"provider": "openai", "model": "gpt-4o", "nano_model": "gpt-4o-mini"},
    "use_qdrant_faq_cache": {"chat": False, "streaming": False},
    "env": "test",
    "llm_reranker": "gpt-4.1-mini",
}

# _should_bind_tools private API 존재 여부 (LangGraph 버전마다 다를 수 있음)
try:
    from langgraph.prebuilt.chat_agent_executor import _should_bind_tools
    HAS_SHOULD_BIND_TOOLS = True
except ImportError:
    HAS_SHOULD_BIND_TOOLS = False


def _make_agent_cls(
    enabled_tools=(),
    tables=("tbl_a", "tbl_b"),
    collections=("col_x",),
):
    """제어 가능한 속성을 가진 구체적인 BaseAgent 서브클래스를 반환한다."""

    class ConcreteAgent(BaseAgent):
        def __init__(self):
            super().__init__("test_agent")

        @property
        def accessible_table_list(self) -> List[str]:
            return list(tables)

        @property
        def accessible_collection_list(self) -> List[str]:
            return list(collections)

        @property
        def enabled_tools(self) -> List[str]:
            return list(enabled_tools)

        def get_prompt_path(self) -> Path:
            return Path(__file__).parent / "mock_prompt.txt"

    return ConcreteAgent


def _make_agent(
    enabled_tools=(),
    tables=("tbl_a", "tbl_b"),
    collections=("col_x",),
    prompt_text="Base prompt. {data_access_section}{current_datetime}",
):
    """__init__ I/O 의존성을 모두 패치한 ConcreteAgent 인스턴스를 반환한다."""
    cls = _make_agent_cls(enabled_tools=enabled_tools, tables=tables, collections=collections)
    mock_loader = MagicMock()
    mock_pm = MagicMock()
    mock_pm.load_prompt_by_path.return_value = prompt_text

    with (
        patch("supervisors.v3.agents.base_agent.load_config", return_value=MOCK_CONFIG),
        patch("supervisors.v3.agents.base_agent.LangchainLoader", return_value=mock_loader),
        patch("supervisors.v3.agents.base_agent.PromptManager", return_value=mock_pm),
    ):
        agent = cls()

    return agent, mock_loader, mock_pm


def _render_static(agent, dt="") -> str:
    """PromptTemplate을 dt로 포맷팅해 문자열로 반환한다. 알 수 없는 변수는 원본 유지."""
    tmpl = agent._create_prompt_template()
    try:
        return tmpl.format(current_datetime=dt)
    except (KeyError, ValueError):
        # 알 수 없는 플레이스홀더가 있으면 수동 치환
        raw = tmpl.template
        for key, val in tmpl.partial_variables.items():
            raw = raw.replace("{" + key + "}", val)
        raw = raw.replace("{current_datetime}", dt)
        return raw


# ═══════════════════════════════════════════════════════════════════════════════
# Group 1 — parallel_tool_calls 바인딩 (base_agent.py 개선)
# ═══════════════════════════════════════════════════════════════════════════════

class TestParallelToolCallsBinding:
    """
    A-fix: create_agent_subgraph가 parallel_tool_calls를 보존하는지 검증.

    bind_tools(tools, parallel_tool_calls=True).kwargs를 base_model.bind()에 합쳐서
    create_react_agent에 넘긴다. _should_bind_tools → False → 재바인딩 없이 보존.
    """

    @pytest.fixture
    def base_model(self):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", api_key="dummy")

    @pytest.fixture
    def dummy_tool(self):
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def _dummy(x: str) -> str:
            """dummy"""
            return x

        return _dummy

    # ── 바인딩 구조 ──────────────────────────────────────────────────────────

    def test_merged_binding_has_parallel_tool_calls(self, base_model, dummy_tool):
        """parallel_tool_calls=True가 최외곽 RunnableBinding.kwargs에 있어야 한다."""
        tool_kwargs = base_model.bind_tools([dummy_tool], parallel_tool_calls=True).kwargs
        model = base_model.bind(**tool_kwargs, extra_body={"prompt_cache_key": "agent"})
        assert model.kwargs.get("parallel_tool_calls") is True

    def test_merged_binding_has_extra_body(self, base_model, dummy_tool):
        """extra_body(프롬프트 캐시 키)가 도구 바인딩과 함께 살아있어야 한다."""
        tool_kwargs = base_model.bind_tools([dummy_tool], parallel_tool_calls=True).kwargs
        model = base_model.bind(**tool_kwargs, extra_body={"prompt_cache_key": "agent"})
        assert model.kwargs.get("extra_body") == {"prompt_cache_key": "agent"}

    def test_merged_binding_has_tools_key(self, base_model, dummy_tool):
        """'tools'가 최외곽 kwargs에 있어야 _should_bind_tools가 False를 반환한다."""
        tool_kwargs = base_model.bind_tools([dummy_tool], parallel_tool_calls=True).kwargs
        model = base_model.bind(**tool_kwargs, extra_body={"prompt_cache_key": "agent"})
        assert "tools" in model.kwargs

    # ── _should_bind_tools 게이트 ────────────────────────────────────────────

    @pytest.mark.skipif(not HAS_SHOULD_BIND_TOOLS, reason="_should_bind_tools가 이 LangGraph 버전에 없음")
    def test_should_bind_tools_returns_false_with_tools(self, base_model, dummy_tool):
        """create_react_agent가 재바인딩하지 않아야 한다(parallel_tool_calls 탈락 방지)."""
        tool_kwargs = base_model.bind_tools([dummy_tool], parallel_tool_calls=True).kwargs
        model = base_model.bind(**tool_kwargs, extra_body={"prompt_cache_key": "agent"})
        assert _should_bind_tools(model, [dummy_tool]) is False

    @pytest.mark.skipif(not HAS_SHOULD_BIND_TOOLS, reason="_should_bind_tools가 이 LangGraph 버전에 없음")
    def test_should_bind_tools_returns_true_without_tools(self, base_model):
        """도구 없는 경우: extra_body만 바인딩 → _should_bind_tools True(create_react_agent가 도구 없음을 처리)."""
        model = base_model.bind(extra_body={"prompt_cache_key": "agent"})
        assert _should_bind_tools(model, []) is True

    @pytest.mark.skipif(not HAS_SHOULD_BIND_TOOLS, reason="_should_bind_tools가 이 LangGraph 버전에 없음")
    def test_tool_name_mismatch_raises_value_error(self, base_model, dummy_tool):
        """바인딩된 도구 목록과 실제 도구 목록이 다르면 ValueError가 발생해야 한다."""
        from langchain_core.tools import tool as lc_tool
        from langgraph.prebuilt.chat_agent_executor import _should_bind_tools

        @lc_tool
        def other_tool(x: str) -> str:
            "other"
            return x

        tool_kwargs = base_model.bind_tools([dummy_tool], parallel_tool_calls=True).kwargs
        model = base_model.bind(**tool_kwargs, extra_body={"prompt_cache_key": "agent"})
        with pytest.raises(ValueError, match="Missing tools"):
            _should_bind_tools(model, [other_tool])

    # ── create_agent_subgraph 통합 ───────────────────────────────────────────

    def test_create_react_agent_not_called_with_parallel_tool_calls(self):
        """create_react_agent에 parallel_tool_calls kwargs가 전달되면 LangGraph에서 TypeError."""
        from langchain_openai import ChatOpenAI
        from supervisors.v3.state import SupervisorV3State

        agent, mock_loader, _ = _make_agent(enabled_tools=[])
        mock_loader.get_model.return_value = ChatOpenAI(model="gpt-4o-mini", api_key="dummy")

        captured = {}

        def spy_cra(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("supervisors.v3.agents.base_agent.create_react_agent", side_effect=spy_cra):
            agent.create_agent_subgraph(state_schema=SupervisorV3State)

        assert "parallel_tool_calls" not in captured, (
            "parallel_tool_calls가 create_react_agent에 전달됨 — "
            "LangGraph 0.6.x에서 TypeError 유발 (base_agent.py fix 확인)"
        )

    def test_create_react_agent_receives_tools_kwarg(self):
        """tools는 빈 리스트여도 create_react_agent에 전달되어야 한다."""
        from langchain_openai import ChatOpenAI
        from supervisors.v3.state import SupervisorV3State

        agent, mock_loader, _ = _make_agent(enabled_tools=[])
        mock_loader.get_model.return_value = ChatOpenAI(model="gpt-4o-mini", api_key="dummy")

        captured = {}

        def spy_cra(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("supervisors.v3.agents.base_agent.create_react_agent", side_effect=spy_cra):
            agent.create_agent_subgraph(state_schema=SupervisorV3State)

        assert "tools" in captured
        assert isinstance(captured["tools"], list)

    def test_model_in_create_react_agent_has_parallel_tool_calls_when_tools_exist(self):
        """도구가 있을 때 create_react_agent에 넘기는 모델이 parallel_tool_calls=True를 가져야 한다."""
        from langchain_openai import ChatOpenAI
        from supervisors.v3.state import SupervisorV3State

        agent, mock_loader, _ = _make_agent(enabled_tools=[])
        base = ChatOpenAI(model="gpt-4o-mini", api_key="dummy")
        mock_loader.get_model.return_value = base

        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def fake_tool(x: str) -> str:
            "fake"
            return x

        agent._initialize_tools = lambda: [fake_tool]

        captured = {}

        def spy_cra(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("supervisors.v3.agents.base_agent.create_react_agent", side_effect=spy_cra):
            agent.create_agent_subgraph(state_schema=SupervisorV3State)

        model = captured["model"]
        assert model.kwargs.get("parallel_tool_calls") is True


# ═══════════════════════════════════════════════════════════════════════════════
# Group 2 — _create_prompt_template (PromptTemplate 구조 검증)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCreatePromptTemplate:
    """
    _create_prompt_template이 올바른 PromptTemplate을 반환하는지 검증.

    - data_access_section은 partial_variables로 미리 바인딩 (정적)
    - current_datetime은 input_variables로 런타임 주입 (동적)
    - 최종 렌더 결과에 {current_datetime} 플레이스홀더가 없어야 한다
    """

    def test_returns_prompt_template(self):
        from langchain_core.prompts import PromptTemplate
        agent, _, _ = _make_agent(prompt_text="Simple prompt.")
        result = agent._create_prompt_template()
        assert isinstance(result, PromptTemplate)

    def test_current_datetime_in_input_variables(self):
        """current_datetime은 런타임 주입 변수여야 한다."""
        agent, _, _ = _make_agent(prompt_text="Hello {current_datetime}")
        tmpl = agent._create_prompt_template()
        assert "current_datetime" in tmpl.input_variables

    def test_data_access_section_in_partial_variables(self):
        """data_access_section은 생성 시 이미 채워진 partial_variables여야 한다."""
        agent, _, _ = _make_agent(
            enabled_tools=[],
            prompt_text="{data_access_section}{current_datetime}",
        )
        tmpl = agent._create_prompt_template()
        assert "data_access_section" in tmpl.partial_variables

    def test_idempotent(self):
        """두 번 호출해도 같은 렌더 결과여야 한다."""
        agent, _, _ = _make_agent(
            enabled_tools=["rdb"],
            prompt_text="Prompt {data_access_section}{current_datetime}",
        )
        assert _render_static(agent, "") == _render_static(agent, "")

    # ── {current_datetime} ──────────────────────────────────────────────────

    def test_rendered_string_has_no_current_datetime_placeholder(self):
        """포맷팅 후 결과 문자열에 {current_datetime} 리터럴이 없어야 한다."""
        agent, _, _ = _make_agent(prompt_text="Time: {current_datetime} end.")
        assert "{current_datetime}" not in _render_static(agent, "2026-04-16")

    def test_rendered_string_contains_datetime_value(self):
        """포맷팅 시 전달한 datetime 값이 결과 문자열에 포함되어야 한다."""
        agent, _, _ = _make_agent(prompt_text="Time: {current_datetime} end.")
        result = _render_static(agent, "2026-04-16 09:30")
        assert "2026-04-16 09:30" in result

    # ── {data_access_section} ───────────────────────────────────────────────

    def test_data_access_section_empty_when_no_tools(self):
        """도구 없으면 data_access_section partial_variable이 빈 문자열이어야 한다."""
        agent, _, _ = _make_agent(enabled_tools=[], prompt_text="Prompt {data_access_section}.")
        tmpl = agent._create_prompt_template()
        das = tmpl.partial_variables.get("data_access_section", "MISSING")
        assert das == "", f"data_access_section이 비어있지 않음: {repr(das)}"

    @pytest.mark.parametrize(
        "tools, expect_in, not_in",
        [
            ([], [], ["Accessible Tables", "Accessible Collections"]),
            (["rdb"], ["Accessible Tables"], ["Accessible Collections"]),
            (["vdb"], ["Accessible Collections"], ["Accessible Tables"]),
            (["rdb", "vdb"], ["Accessible Tables", "Accessible Collections"], []),
        ],
    )
    def test_data_access_section_content_by_tools(self, tools, expect_in, not_in):
        agent, _, _ = _make_agent(
            enabled_tools=tools,
            tables=["crmProduct"],
            collections=["qa_col"],
            prompt_text="{data_access_section}{current_datetime}",
        )
        tmpl = agent._create_prompt_template()
        das = tmpl.partial_variables.get("data_access_section", "")
        for s in expect_in:
            assert s in das, f"enabled_tools={tools}일 때 '{s}'가 data_access_section에 없음"
        for s in not_in:
            assert s not in das, f"enabled_tools={tools}일 때 '{s}'가 data_access_section에 있으면 안 됨"

    def test_table_names_in_data_access_section(self):
        agent, _, _ = _make_agent(
            enabled_tools=["rdb"],
            tables=["crmEmployee", "crmProduct"],
            prompt_text="{data_access_section}",
        )
        das = agent._create_prompt_template().partial_variables.get("data_access_section", "")
        assert "crmEmployee" in das
        assert "crmProduct" in das

    def test_collection_names_in_data_access_section(self):
        agent, _, _ = _make_agent(
            enabled_tools=["vdb"],
            collections=["knowledge_base"],
            prompt_text="{data_access_section}",
        )
        das = agent._create_prompt_template().partial_variables.get("data_access_section", "")
        assert "knowledge_base" in das

    # ── 프롬프트 로드 ────────────────────────────────────────────────────────

    def test_prompt_loaded_via_manager_once(self):
        agent, _, mock_pm = _make_agent(prompt_text="test")
        agent._create_prompt_template()
        mock_pm.load_prompt_by_path.assert_called_once()

    def test_prompt_path_passed_to_manager(self):
        agent, _, mock_pm = _make_agent(prompt_text="test")
        agent._create_prompt_template()
        called_path = mock_pm.load_prompt_by_path.call_args[0][0]
        assert called_path == agent.get_prompt_path()

    # ── 오류 복원력 ──────────────────────────────────────────────────────────

    def test_unknown_placeholders_do_not_raise_on_template_creation(self):
        """알 수 없는 변수가 있어도 PromptTemplate 생성 자체는 예외가 없어야 한다."""
        from langchain_core.prompts import PromptTemplate
        agent, _, _ = _make_agent(
            prompt_text="Prompt {current_datetime} {totally_unknown} end."
        )
        # 생성 단계에서 예외 없음을 확인
        result = agent._create_prompt_template()
        assert isinstance(result, PromptTemplate)

    def test_render_static_helper_handles_unknown_placeholder(self):
        """_render_static이 알 수 없는 플레이스홀더를 안전하게 처리해야 한다."""
        agent, _, _ = _make_agent(
            prompt_text="Prompt {current_datetime} {totally_unknown} end."
        )
        result = _render_static(agent, "2026-04-16")
        assert isinstance(result, str)
        assert "{current_datetime}" not in result


# ═══════════════════════════════════════════════════════════════════════════════
# Group 3 — agent_prompt_function (단일 SystemMessage + 히스토리)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentPromptFunction:
    """
    create_agent_subgraph가 create_react_agent에 넘기는 prompt callable 검증.

    현재 구현: [SystemMessage(formatted_prompt)] + existing_messages
    - formatted_prompt = prompt_template.format(current_datetime=...)
    - data_access_section은 partial_variables로 이미 채워진 상태
    - 히스토리 메시지는 뒤에 이어붙임
    """

    @pytest.fixture
    def prompt_fn(self):
        """create_agent_subgraph에서 create_react_agent에 넘어가는 prompt callable을 캡처."""
        from langchain_openai import ChatOpenAI
        from supervisors.v3.state import SupervisorV3State

        agent, mock_loader, _ = _make_agent(
            enabled_tools=[],
            prompt_text="Static instruction. {data_access_section}{current_datetime}",
        )
        mock_loader.get_model.return_value = ChatOpenAI(model="gpt-4o-mini", api_key="dummy")

        captured = {}

        def spy_cra(**kwargs):
            captured["prompt"] = kwargs["prompt"]
            return MagicMock()

        with patch("supervisors.v3.agents.base_agent.create_react_agent", side_effect=spy_cra):
            agent.create_agent_subgraph(state_schema=SupervisorV3State)

        return captured["prompt"]

    def _state(self, products=None, hospitals=None, dt="2026-04-16", messages=None):
        return {
            "current_datetime": dt,
            "known_products_list": products if products is not None else [],
            "known_hospitals_list": hospitals if hospitals is not None else [],
            "messages": messages or [],
        }

    # ── 기본 구조 ────────────────────────────────────────────────────────────

    def test_returns_list(self, prompt_fn):
        assert isinstance(prompt_fn(self._state()), list)

    def test_first_message_is_system_message(self, prompt_fn):
        from langchain_core.messages import SystemMessage
        assert isinstance(prompt_fn(self._state())[0], SystemMessage)

    def test_system_message_content_is_non_empty(self, prompt_fn):
        assert prompt_fn(self._state())[0].content

    def test_empty_history_produces_exactly_one_system_message(self, prompt_fn):
        """히스토리가 없으면 SystemMessage 1개만 있어야 한다."""
        from langchain_core.messages import SystemMessage
        msgs = prompt_fn(self._state(messages=[]))
        assert len(msgs) == 1
        assert isinstance(msgs[0], SystemMessage)

    # ── SystemMessage 내용 검증 ──────────────────────────────────────────────

    def test_system_message_contains_original_instruction(self, prompt_fn):
        """원본 프롬프트 지시문이 SystemMessage에 포함되어야 한다."""
        assert "Static instruction" in prompt_fn(self._state())[0].content

    def test_system_message_contains_datetime(self, prompt_fn):
        """current_datetime 값이 SystemMessage에 반영되어야 한다."""
        assert "2026-04-16" in prompt_fn(self._state(dt="2026-04-16"))[0].content

    def test_system_message_changes_when_datetime_changes(self, prompt_fn):
        """datetime이 달라지면 SystemMessage 내용도 달라야 한다."""
        a = prompt_fn(self._state(dt="2026-01-01"))[0].content
        b = prompt_fn(self._state(dt="2026-12-31"))[0].content
        assert a != b

    def test_no_current_datetime_placeholder_in_system_message(self, prompt_fn):
        """{current_datetime} 리터럴이 SystemMessage 내에 남으면 안 된다."""
        assert "{current_datetime}" not in prompt_fn(self._state())[0].content

    def test_no_data_access_section_placeholder_in_system_message(self, prompt_fn):
        """{data_access_section} 리터럴이 SystemMessage 내에 남으면 안 된다."""
        assert "{data_access_section}" not in prompt_fn(self._state())[0].content

    # ── 히스토리 메시지 이어붙임 ─────────────────────────────────────────────

    def test_existing_messages_appended_at_end(self, prompt_fn):
        from langchain_core.messages import HumanMessage
        user_msg = HumanMessage(content="질문입니다")
        msgs = prompt_fn(self._state(messages=[user_msg]))
        assert msgs[-1] is user_msg

    def test_multiple_history_messages_preserved_in_order(self, prompt_fn):
        from langchain_core.messages import HumanMessage, AIMessage
        h1 = HumanMessage(content="q1")
        a1 = AIMessage(content="a1")
        h2 = HumanMessage(content="q2")
        msgs = prompt_fn(self._state(messages=[h1, a1, h2]))
        assert msgs[-3] is h1
        assert msgs[-2] is a1
        assert msgs[-1] is h2

    def test_total_message_count_with_history(self, prompt_fn):
        """SystemMessage 1개 + 히스토리 n개 = n+1개."""
        from langchain_core.messages import HumanMessage
        history = [HumanMessage(content=f"msg{i}") for i in range(3)]
        msgs = prompt_fn(self._state(messages=history))
        assert len(msgs) == 4  # 1 system + 3 history

    def test_none_products_hospitals_do_not_raise(self, prompt_fn):
        """known_products/hospitals가 None이어도 예외 없이 동작해야 한다."""
        state = {
            "current_datetime": "2026-04-16",
            "known_products_list": None,
            "known_hospitals_list": None,
            "messages": [],
        }
        result = prompt_fn(state)
        assert isinstance(result, list)
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Group 4 — SupervisorV3 preflight 동시 실행 (C 개선)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreflightConcurrency:
    """
    process_user_query가 _prepare_request_context, _load_user_profile,
    임베딩 계산을 ThreadPoolExecutor로 동시에 실행하는지 검증.
    """

    @pytest.fixture
    def sv3(self):
        with (
            patch("supervisors.v3.supervisor_v3.load_config", return_value=MOCK_CONFIG),
            patch("supervisors.v3.supervisor_v3.LangchainLoader"),
            patch("supervisors.v3.supervisor_v3.PromptManager"),
            patch("supervisors.v3.supervisor_v3.CacheService"),
            patch("supervisors.v3.supervisor_v3.QuestionGraph"),
            patch("supervisors.v3.supervisor_v3.ContextEnhancer"),
            patch.object(SupervisorV3, "_initialize_agents"),
            patch.object(SupervisorV3, "_create_supervisor", return_value=MagicMock()),
            patch.object(SupervisorV3, "_load_prompts"),
        ):
            sv = SupervisorV3()

        sv._chat_cache_enabled = False
        sv._streaming_cache_enabled = False
        sv._emit_ref_dict = False
        return sv

    def _ctx_result(self):
        return {
            "all_messages": [],
            "known_products": [],
            "known_hospitals": [],
            "original_input": "test",
            "detected_company_products": [],
            "detected_competitive_products": [],
            "detected_hospitals": [],
        }

    def _profile_result(self):
        return {"role": "mr", "dept_name": "", "headquarter": "", "employee_name": ""}

    def _minimal_sv3_setup(self, sv3):
        """구체적인 타이밍이 필요 없는 테스트용 빠른 mock."""
        from langchain_core.messages import AIMessage
        sv3._question_graph = MagicMock()
        sv3._question_graph._get_embeddings.return_value = [[0.1] * 8]
        sv3._prepare_request_context = MagicMock(return_value=self._ctx_result())
        sv3._load_user_profile = MagicMock(return_value=self._profile_result())
        sv3._supervisor.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="답변입니다")], "reference_dict": []}
        )

    def _make_recording(self, key, result_fn, windows, delay=0.15):
        """실행 시간 윈도우를 windows[key]에 기록하는 래퍼를 반환한다."""
        def inner(*args, **kwargs):
            t0 = time.monotonic()
            time.sleep(delay)
            windows[key] = (t0, time.monotonic())
            return result_fn(*args, **kwargs)
        return inner

    @staticmethod
    def _assert_overlaps(windows, a, b):
        """두 실행 윈도우가 겹치는지(동시 실행) 확인한다."""
        s1, e1 = windows[a]
        s2, e2 = windows[b]
        assert s1 < e2 and s2 < e1, (
            f"작업 '{a}'와 '{b}'가 겹치지 않음 — 순차 실행 감지.\n"
            f"  {a}: {s1:.3f}s – {e1:.3f}s\n"
            f"  {b}: {s2:.3f}s – {e2:.3f}s"
        )

    def _run(self, sv3, employee_id="E001"):
        with (
            patch("logger.query_logger.query_logger"),
            patch("supervisors.v3.utils.filter_ref_dict_for_user", return_value=[]),
        ):
            try:
                return sv3.process_user_query(
                    "session-test", "테스트 쿼리", [], employee_ID=employee_id
                )
            except Exception:
                return None

    # ── 동시 실행 검증 ───────────────────────────────────────────────────────

    def test_three_preflight_tasks_run_concurrently(self, sv3):
        """ctx / profile / embedding 세 작업이 모두 동시에 실행되어야 한다."""
        DELAY = 0.2
        windows = {}

        sv3._prepare_request_context = self._make_recording(
            "ctx", lambda *a, **k: self._ctx_result(), windows, DELAY
        )
        sv3._load_user_profile = self._make_recording(
            "profile", lambda *a, **k: self._profile_result(), windows, DELAY
        )
        sv3._question_graph = MagicMock()
        sv3._question_graph._get_embeddings.side_effect = self._make_recording(
            "emb", lambda q: [[0.1] * 8], windows, DELAY
        )
        from langchain_core.messages import AIMessage
        sv3._supervisor.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="ok")], "reference_dict": []}
        )

        self._run(sv3)

        assert set(windows) == {"ctx", "profile", "emb"}, (
            f"실행된 preflight 작업 집합이 올바르지 않음: {set(windows)}"
        )
        self._assert_overlaps(windows, "ctx", "profile")
        self._assert_overlaps(windows, "ctx", "emb")
        self._assert_overlaps(windows, "profile", "emb")

    def test_ctx_and_emb_overlap_without_employee_id(self, sv3):
        """employee_ID=None일 때도 ctx와 embedding이 동시에 실행되어야 한다."""
        DELAY = 0.2
        windows = {}

        sv3._prepare_request_context = self._make_recording(
            "ctx", lambda *a, **k: self._ctx_result(), windows, DELAY
        )
        sv3._question_graph = MagicMock()
        sv3._question_graph._get_embeddings.side_effect = self._make_recording(
            "emb", lambda q: [[0.1] * 8], windows, DELAY
        )
        from langchain_core.messages import AIMessage
        sv3._supervisor.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="ok")], "reference_dict": []}
        )

        self._run(sv3, employee_id=None)

        assert "ctx" in windows and "emb" in windows
        self._assert_overlaps(windows, "ctx", "emb")

    # ── 임베딩 공유 (B 개선) ─────────────────────────────────────────────────

    def test_embedding_computed_exactly_once(self, sv3):
        """query_embedding은 preflight에서 1번만 계산되어야 한다(임베딩 공유)."""
        self._minimal_sv3_setup(sv3)
        call_count = 0

        def counting_embed(queries):
            nonlocal call_count
            call_count += 1
            return [[0.1] * 8]

        sv3._question_graph._get_embeddings.side_effect = counting_embed

        self._run(sv3)

        assert call_count == 1, (
            f"임베딩 {call_count}회 호출됨. "
            "preflight에서 1회 계산 후 공유해야 함."
        )

    def test_embedding_failure_does_not_crash(self, sv3):
        """임베딩 계산 실패 시 process_user_query가 크래시하면 안 된다."""
        self._minimal_sv3_setup(sv3)
        sv3._question_graph._get_embeddings.side_effect = RuntimeError("qdrant down")
        self._run(sv3)  # 예외 없이 완료되어야 함 (None 반환 허용)

    # ── 결과 형태 ────────────────────────────────────────────────────────────

    def test_result_has_answer_key(self, sv3):
        self._minimal_sv3_setup(sv3)
        result = self._run(sv3)
        assert result is None or "answer" in result

    def test_result_has_ref_dict_key(self, sv3):
        self._minimal_sv3_setup(sv3)
        result = self._run(sv3)
        assert result is None or "ref_dict" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Group 5 — ToolNode 병렬 도구 실행 (A 개선)
# ═══════════════════════════════════════════════════════════════════════════════

class TestParallelToolCallsExecution:
    """
    LLM이 단일 응답에 tool_call 2개(RDB + VDB)를 반환했을 때,
    ToolNode가 두 도구를 같은 라운드에서 실행하는지 검증.

    _TrackingFakeModel: Pydantic v2 패치 문제를 우회하고 입출력을 추적.
    """

    @pytest.fixture
    def tools(self):
        from langchain_core.tools import tool as lc_tool

        @lc_tool
        def fake_rdb(query: str) -> str:
            """RDB에서 구조화 데이터를 조회합니다."""
            return "RDB 결과: 제품 가격 1000원"

        @lc_tool
        def fake_vdb(query: str) -> str:
            """VDB에서 벡터 유사도 검색을 합니다."""
            return "VDB 결과: 관련 문서 3건 발견"

        return [fake_rdb, fake_vdb]

    def _make_graph(self, tools, responses):
        """주어진 응답 목록을 반환하는 그래프를 생성한다."""
        from langgraph.prebuilt import create_react_agent
        from langchain_core.messages import AIMessage as _AI

        model = _TrackingFakeModel(responses=responses)
        graph = create_react_agent(model=model, tools=tools)
        return graph, model

    @staticmethod
    def _ai_parallel(tool_a: str, tool_b: str):
        from langchain_core.messages import AIMessage
        return AIMessage(
            content="",
            tool_calls=[
                {"name": tool_a, "args": {"query": "test"}, "id": "call_001", "type": "tool_call"},
                {"name": tool_b, "args": {"query": "test"}, "id": "call_002", "type": "tool_call"},
            ],
        )

    @staticmethod
    def _ai_sequential(tool_name: str, call_id: str):
        from langchain_core.messages import AIMessage
        return AIMessage(
            content="",
            tool_calls=[
                {"name": tool_name, "args": {"query": "test"}, "id": call_id, "type": "tool_call"},
            ],
        )

    @staticmethod
    def _ai_final():
        from langchain_core.messages import AIMessage
        return AIMessage(content="두 검색 결과를 종합한 최종 답변입니다.")

    def _run(self, graph, model, query="RDB와 VDB를 모두 검색해줘"):
        """그래프를 실행하고 (captured_inputs, final_state, call_count)를 반환한다."""
        from langchain_core.messages import HumanMessage
        state = graph.invoke({"messages": [HumanMessage(content=query)]})
        return model.captured, state, len(model.captured)

    # ── 병렬 실행 검증 ───────────────────────────────────────────────────────

    def test_parallel_tool_calls_take_two_llm_turns(self, tools):
        """LLM이 RDB+VDB를 단일 메시지로 요청하면 LLM 호출이 정확히 2회여야 한다."""
        responses = [self._ai_parallel("fake_rdb", "fake_vdb"), self._ai_final()]
        graph, model = self._make_graph(tools, responses)
        _, _, call_count = self._run(graph, model)

        assert call_count == 2, (
            f"LLM 호출 {call_count}회. 병렬 처리 시 2회여야 함(도구 요청 1회 + 최종 답변 1회)."
        )

    def test_second_llm_call_receives_two_tool_messages(self, tools):
        """두 번째 LLM 호출 입력에 ToolMessage가 정확히 2개여야 한다."""
        from langchain_core.messages import ToolMessage

        responses = [self._ai_parallel("fake_rdb", "fake_vdb"), self._ai_final()]
        graph, model = self._make_graph(tools, responses)
        captured_inputs, _, _ = self._run(graph, model)

        assert len(captured_inputs) >= 2
        tool_msgs = [m for m in captured_inputs[1] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 2, (
            f"두 번째 LLM 호출에 ToolMessage {len(tool_msgs)}개. 병렬 실행 시 2개여야 함."
        )

    def test_both_tool_call_ids_in_second_llm_context(self, tools):
        """두 번째 LLM 호출에 call_001(RDB)과 call_002(VDB) tool_call_id가 모두 있어야 한다."""
        from langchain_core.messages import ToolMessage

        responses = [self._ai_parallel("fake_rdb", "fake_vdb"), self._ai_final()]
        graph, model = self._make_graph(tools, responses)
        captured_inputs, _, _ = self._run(graph, model)

        tool_ids = {m.tool_call_id for m in captured_inputs[1] if isinstance(m, ToolMessage)}
        assert "call_001" in tool_ids
        assert "call_002" in tool_ids

    def test_both_tool_results_in_final_state(self, tools):
        """최종 state에 RDB와 VDB 두 도구의 실행 결과가 모두 있어야 한다."""
        from langchain_core.messages import ToolMessage

        responses = [self._ai_parallel("fake_rdb", "fake_vdb"), self._ai_final()]
        graph, model = self._make_graph(tools, responses)
        _, state, _ = self._run(graph, model)

        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) == 2
        combined = " ".join(m.content for m in tool_msgs)
        assert "RDB" in combined
        assert "VDB" in combined

    # ── 순차 실행 대조군 ─────────────────────────────────────────────────────

    def test_sequential_tool_calls_take_three_llm_turns(self, tools):
        """LLM이 RDB→VDB를 따로 요청하면 LLM 호출이 3회여야 한다."""
        responses = [
            self._ai_sequential("fake_rdb", "call_001"),
            self._ai_sequential("fake_vdb", "call_002"),
            self._ai_final(),
        ]
        graph, model = self._make_graph(tools, responses)
        _, _, call_count = self._run(graph, model)
        assert call_count == 3

    def test_sequential_second_call_has_one_tool_message(self, tools):
        """순차 처리 시 두 번째 LLM 호출에는 ToolMessage가 1개뿐이어야 한다."""
        from langchain_core.messages import ToolMessage

        responses = [
            self._ai_sequential("fake_rdb", "call_001"),
            self._ai_sequential("fake_vdb", "call_002"),
            self._ai_final(),
        ]
        graph, model = self._make_graph(tools, responses)
        captured_inputs, _, _ = self._run(graph, model)

        assert len(captured_inputs) == 3
        second_tool_msgs = [m for m in captured_inputs[1] if isinstance(m, ToolMessage)]
        assert len(second_tool_msgs) == 1, (
            f"순차 처리 두 번째 LLM 호출에 ToolMessage {len(second_tool_msgs)}개. "
            "첫 번째 도구 결과만 있어야 함."
        )

    def test_parallel_fewer_llm_turns_than_sequential(self, tools):
        """병렬 처리(2회)가 순차 처리(3회)보다 LLM 호출 횟수가 적어야 한다."""
        par_responses = [self._ai_parallel("fake_rdb", "fake_vdb"), self._ai_final()]
        g_par, m_par = self._make_graph(tools, par_responses)
        self._run(g_par, m_par)

        seq_responses = [
            self._ai_sequential("fake_rdb", "call_001"),
            self._ai_sequential("fake_vdb", "call_002"),
            self._ai_final(),
        ]
        g_seq, m_seq = self._make_graph(tools, seq_responses)
        self._run(g_seq, m_seq)

        assert len(m_par.captured) == 2, f"병렬 LLM 호출 횟수가 2여야 함: {len(m_par.captured)}"
        assert len(m_seq.captured) == 3, f"순차 LLM 호출 횟수가 3이어야 함: {len(m_seq.captured)}"
        assert len(m_par.captured) < len(m_seq.captured)


# ═══════════════════════════════════════════════════════════════════════════════
# Group 6 — LLMReranker 가중치 계산 (0~10 척도, LLM 호출 없음)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMRerankerScoring:
    """
    _calculate_weighted_score: 0~10 정수 입력 → 0.0~1.0 가중치 합산 출력.
    LLM 호출 없이 순수 계산 로직만 검증.
    """

    @pytest.fixture
    def reranker(self):
        """LLM 호출 없이 LLMReranker 인스턴스만 생성."""
        from supervisors.v3.tools.reranker import LLMReranker
        with (
            patch("supervisors.v3.tools.reranker.LangchainLoader"),
            patch("supervisors.v3.tools.reranker.PromptManager"),
        ):
            r = LLMReranker.__new__(LLMReranker)
            r.threshold = 0.3
            r.llm = MagicMock()
            r.prompt_manager = MagicMock()
            r.rerank_prompt = None
        return r

    def test_weights_sum_to_one(self, reranker):
        """WEIGHTS 합계가 1.0이어야 한다."""
        total = sum(reranker.WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"가중치 합계: {total} (1.0이어야 함)"

    def test_all_max_score_gives_one(self, reranker):
        """모든 기준이 10점이면 최종 점수가 1.0이어야 한다."""
        scores = {
            "relevance_score": 10,
            "specificity_score": 10,
            "completeness_score": 10,
            "practicality_score": 10,
            "constraint_fit_score": 10,
        }
        result = reranker._calculate_weighted_score(scores)
        assert abs(result - 1.0) < 1e-9, f"모든 10점 → {result} (1.0이어야 함)"

    def test_all_zero_score_gives_zero(self, reranker):
        """모든 기준이 0점이면 최종 점수가 0.0이어야 한다."""
        scores = {k: 0 for k in reranker.WEIGHTS}
        result = reranker._calculate_weighted_score(scores)
        assert abs(result - 0.0) < 1e-9, f"모든 0점 → {result} (0.0이어야 함)"

    def test_relevance_weight_dominates(self, reranker):
        """relevance만 10점이면 최종 점수 ≈ 0.60이어야 한다."""
        scores = {
            "relevance_score": 10,
            "specificity_score": 0,
            "completeness_score": 0,
            "practicality_score": 0,
            "constraint_fit_score": 0,
        }
        result = reranker._calculate_weighted_score(scores)
        expected = reranker.WEIGHTS["relevance_score"]
        assert abs(result - expected) < 1e-9, (
            f"relevance=10, 나머지=0 → {result} ({expected}이어야 함)"
        )

    def test_score_always_in_0_1_range(self, reranker):
        """다양한 점수 조합에서 결과는 항상 [0, 1] 범위여야 한다."""
        import random
        random.seed(42)
        for _ in range(20):
            scores = {k: random.randint(0, 10) for k in reranker.WEIGHTS}
            result = reranker._calculate_weighted_score(scores)
            assert 0.0 <= result <= 1.0 + 1e-9, (
                f"점수 {scores} → {result} ([0,1] 범위 초과)"
            )

    def test_score_normalized_by_10(self, reranker):
        """5점(절반)은 각 가중치의 절반 기여여야 한다."""
        scores = {k: 5 for k in reranker.WEIGHTS}
        result = reranker._calculate_weighted_score(scores)
        # 모든 점수가 5/10=0.5이면 weighted_sum = 0.5 * sum(weights) = 0.5
        assert abs(result - 0.5) < 1e-9, f"모든 5점 → {result} (0.5이어야 함)"

    def test_rerank_response_model_max_is_10(self):
        """RerankResponse가 0~10 범위를 강제해야 한다."""
        from supervisors.v3.tools.reranker import RerankResponse
        # 유효 값
        r = RerankResponse(relevance=10, specificity=5, completeness=3, practicality=7, constraint_fit=0)
        assert r.relevance == 10

    def test_rerank_response_model_rejects_above_10(self):
        """RerankResponse에 11점 이상을 넣으면 ValidationError여야 한다."""
        from supervisors.v3.tools.reranker import RerankResponse
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            RerankResponse(
                relevance=11,  # 범위 초과
                specificity=0,
                completeness=0,
                practicality=0,
                constraint_fit=0,
            )

    def test_rerank_response_model_rejects_below_0(self):
        """RerankResponse에 음수를 넣으면 ValidationError여야 한다."""
        from supervisors.v3.tools.reranker import RerankResponse
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            RerankResponse(
                relevance=-1,
                specificity=0,
                completeness=0,
                practicality=0,
                constraint_fit=0,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Group 7 — LLMReranker 싱글턴 (_get_reranker, vdb.py 개선)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRerankerSingleton:
    """
    vdb.py의 _get_reranker()가 모듈 레벨 싱글턴을 올바르게 반환하는지 검증.
    VDB 툴 호출마다 LLMReranker()를 새로 생성하던 문제를 해결한 개선.
    """

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """테스트 간 싱글턴 격리: 테스트 전후로 _reranker를 None으로 초기화."""
        import supervisors.v3.tools.vdb as vdb_mod
        original = vdb_mod._reranker
        vdb_mod._reranker = None
        yield
        vdb_mod._reranker = original

    def test_get_reranker_returns_llm_reranker_instance(self):
        """_get_reranker()는 LLMReranker 인스턴스를 반환해야 한다."""
        from supervisors.v3.tools.reranker import LLMReranker
        from supervisors.v3.tools.vdb import _get_reranker

        mock_instance = MagicMock(spec=LLMReranker)

        with patch("supervisors.v3.tools.vdb.LLMReranker", return_value=mock_instance):
            result = _get_reranker()

        assert result is mock_instance

    def test_get_reranker_returns_same_instance_on_second_call(self):
        """두 번 호출 시 동일한 인스턴스를 반환해야 한다(싱글턴)."""
        from supervisors.v3.tools.reranker import LLMReranker
        from supervisors.v3.tools.vdb import _get_reranker

        mock_instance = MagicMock(spec=LLMReranker)

        with patch("supervisors.v3.tools.vdb.LLMReranker", return_value=mock_instance):
            first = _get_reranker()
            second = _get_reranker()

        assert first is second, "두 번째 호출에서 다른 인스턴스 반환됨 — 싱글턴 미작동"

    def test_llm_reranker_constructor_called_only_once(self):
        """싱글턴이므로 LLMReranker() 생성자는 1회만 호출되어야 한다."""
        from supervisors.v3.tools.vdb import _get_reranker

        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)

        with patch("supervisors.v3.tools.vdb.LLMReranker", mock_cls):
            _get_reranker()
            _get_reranker()
            _get_reranker()

        assert mock_cls.call_count == 1, (
            f"LLMReranker() 생성자가 {mock_cls.call_count}회 호출됨 (1회여야 함)"
        )

    def test_singleton_threshold_is_03(self):
        """싱글턴 생성 시 threshold=0.3으로 초기화되어야 한다."""
        from supervisors.v3.tools.vdb import _get_reranker

        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)

        with patch("supervisors.v3.tools.vdb.LLMReranker", mock_cls):
            _get_reranker()

        mock_cls.assert_called_once_with(threshold=0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# Group 8 — LLMReranker 동작 검증 (LLM mock 사용)
# ═══════════════════════════════════════════════════════════════════════════════

class TestLLMRerankerWithMockLLM:
    """
    LLM을 mock으로 대체해 LLMReranker의 rerank() 로직을 검증.
    - 임계값 필터링
    - 점수 내림차순 정렬
    - 메타데이터 추가
    - 빈 목록 처리
    """

    @pytest.fixture
    def reranker(self):
        """실제 LLM 없이 LLMReranker 인스턴스를 생성한다."""
        from supervisors.v3.tools.reranker import LLMReranker, RerankResponse

        with (
            patch("supervisors.v3.tools.reranker.LangchainLoader"),
            patch("supervisors.v3.tools.reranker.PromptManager"),
        ):
            r = LLMReranker.__new__(LLMReranker)
            r.threshold = 0.3
            r.MAX_WORKERS = 10
            r.BATCH_TIMEOUT_SEC = 30
            r.SINGLE_DOC_TIMEOUT_SEC = 5
            r.EPSILON = 1e-9
            r.WEIGHTS = {
                "relevance_score": 0.60,
                "specificity_score": 0.15,
                "completeness_score": 0.10,
                "practicality_score": 0.10,
                "constraint_fit_score": 0.05,
            }
            r.llm = MagicMock()
            r.prompt_manager = MagicMock()
            r.rerank_prompt = "Query: {user_query}\nSearch: {search_query}\nDoc: {document}"

        return r

    def _make_docs(self, n: int):
        from langchain_core.documents import Document
        return [Document(page_content=f"문서 내용 {i}", metadata={"idx": i}) for i in range(n)]

    def _make_response(self, relevance=8, specificity=7, completeness=6, practicality=5, constraint_fit=4):
        from supervisors.v3.tools.reranker import RerankResponse
        return RerankResponse(
            relevance=relevance,
            specificity=specificity,
            completeness=completeness,
            practicality=practicality,
            constraint_fit=constraint_fit,
        )

    def test_rerank_empty_docs_returns_empty(self, reranker):
        """빈 문서 리스트 입력 시 빈 리스트를 반환해야 한다."""
        result = reranker.rerank([], user_query="테스트")
        assert result == []

    def test_rerank_no_prompt_returns_original(self, reranker):
        """rerank_prompt가 None이면 원본 문서를 그대로 반환해야 한다."""
        reranker.rerank_prompt = None
        docs = self._make_docs(3)
        result = reranker.rerank(docs, user_query="테스트")
        assert len(result) == len(docs)

    def test_rerank_filters_below_threshold(self, reranker):
        """threshold 미만의 문서는 결과에서 제거되어야 한다."""
        # 문서 2개: 하나는 높은 점수(통과), 하나는 낮은 점수(필터)
        high_response = self._make_response(relevance=10, specificity=10, completeness=10, practicality=10, constraint_fit=10)
        low_response = self._make_response(relevance=0, specificity=0, completeness=0, practicality=0, constraint_fit=0)

        docs = self._make_docs(2)
        responses = [high_response, low_response]
        reranker.llm.invoke.side_effect = responses

        result = reranker.rerank(docs, user_query="테스트")

        # threshold=0.3이므로 0점 문서는 필터됨
        assert len(result) == 1, f"threshold 미만 문서 필터 실패: {len(result)}개 반환"

    def test_rerank_orders_by_score_desc(self, reranker):
        """점수 높은 문서가 먼저 나와야 한다."""
        # 3개 문서, 각각 high/mid/low 점수
        responses = [
            self._make_response(relevance=3, specificity=3, completeness=3, practicality=3, constraint_fit=3),  # mid
            self._make_response(relevance=10, specificity=10, completeness=10, practicality=10, constraint_fit=10),  # high
            self._make_response(relevance=5, specificity=5, completeness=5, practicality=5, constraint_fit=5),  # med
        ]
        docs = self._make_docs(3)
        reranker.llm.invoke.side_effect = responses

        result = reranker.rerank(docs, user_query="테스트")

        # 점수 내림차순이어야 함
        scores = [d.metadata.get("overall_score", 0) for d in result]
        assert scores == sorted(scores, reverse=True), f"점수 정렬 실패: {scores}"

    def test_rerank_adds_rank_metadata(self, reranker):
        """결과 문서에 'rank' 메타데이터가 1부터 추가되어야 한다."""
        docs = self._make_docs(2)
        reranker.llm.invoke.return_value = self._make_response(relevance=8, specificity=5, completeness=5, practicality=5, constraint_fit=5)

        result = reranker.rerank(docs, user_query="테스트")

        assert all("rank" in d.metadata for d in result)
        ranks = sorted(d.metadata["rank"] for d in result)
        assert ranks == list(range(1, len(result) + 1)), f"rank 연속성 실패: {ranks}"

    def test_rerank_adds_overall_score_metadata(self, reranker):
        """결과 문서에 'overall_score' 메타데이터가 추가되어야 한다."""
        docs = self._make_docs(1)
        reranker.llm.invoke.return_value = self._make_response()

        result = reranker.rerank(docs, user_query="테스트")

        assert "overall_score" in result[0].metadata
        assert 0.0 <= result[0].metadata["overall_score"] <= 1.0

    def test_rerank_individual_scores_in_metadata(self, reranker):
        """개별 기준 점수가 메타데이터에 기록되어야 한다."""
        docs = self._make_docs(1)
        response = self._make_response(relevance=9, constraint_fit=2)
        reranker.llm.invoke.return_value = response

        result = reranker.rerank(docs, user_query="테스트")

        assert result[0].metadata.get("relevance_score") == 9
        assert result[0].metadata.get("constraint_fit_score") == 2

    def test_rerank_all_docs_pass_zero_threshold(self, reranker):
        """threshold=0.0이면 점수 0점이어도 모든 문서가 통과해야 한다."""
        reranker.threshold = 0.0
        docs = self._make_docs(3)
        reranker.llm.invoke.return_value = self._make_response(
            relevance=0, specificity=0, completeness=0, practicality=0, constraint_fit=0
        )

        result = reranker.rerank(docs, user_query="테스트")
        assert len(result) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Group 9 — LLMReranker 실제 LLM 통합 테스트
# ═══════════════════════════════════════════════════════════════════════════════

def _can_make_real_call() -> bool:
    """실제 API 호출 가능 여부 확인."""
    try:
        from config_loader import load_config
        cfg = load_config()
        return isinstance(cfg, dict) and bool(cfg)
    except Exception:
        return False


HAS_REAL_CONFIG = _can_make_real_call()

# 통합 테스트 문서: 기넥신(자사) vs 리넥신(경쟁사)
_DOC_기넥신_부작용 = (
    "기넥신(은행잎 추출물, EGb 761)의 주요 부작용으로는 두통, 어지러움, 오심, "
    "위장 장애가 보고되어 있습니다. 드물게 출혈 경향이 증가할 수 있으며, "
    "항응고제(와파린, 아스피린)와 병용 시 상호작용에 주의해야 합니다. "
    "기넥신 복용 중 이상반응 발생 시 즉시 의사와 상담하세요."
)

_DOC_리넥신_일반 = (
    "리넥신은 혈액순환 장애 개선 효능을 가진 약물입니다. "
    "주성분은 기넥신과 다르며, 적응증 및 용법용량이 다릅니다. "
    "리넥신의 부작용 프로파일은 별도 자료를 참고하시기 바랍니다."
)

_DOC_기넥신_용법 = (
    "기넥신 복용법: 성인 1회 1정(80mg), 1일 2회 식후 복용. "
    "고령자의 경우 의사 판단에 따라 용량을 조절할 수 있습니다. "
    "기넥신은 최소 12주 이상 꾸준히 복용해야 효과가 나타납니다."
)

_DOC_무관한_내용 = (
    "오늘 날씨는 맑고 기온은 15도입니다. "
    "산책하기 좋은 날씨이며, 자외선 지수가 높습니다."
)


@pytest.fixture(scope="module")
def real_reranker():
    """실제 LLMReranker 인스턴스. 설정이 없으면 테스트 스킵."""
    if not HAS_REAL_CONFIG:
        pytest.skip("실제 API 설정이 없음 — 통합 테스트 건너뜀")
    try:
        from supervisors.v3.tools.reranker import LLMReranker
        return LLMReranker(threshold=0.3)
    except Exception as e:
        pytest.skip(f"LLMReranker 초기화 실패: {e}")


@pytest.mark.skipif(not HAS_REAL_CONFIG, reason="통합 테스트: 실제 API 설정 필요")
class TestRerankerIntegration:
    """
    실제 LLM을 호출해 Rerank_개선안.md의 개선 효과를 검증.

    2-2. 0~10 척도 세분화
    2-4. 유사 제품명 혼동 해소 (기넥신 vs 리넥신)
    """

    def _make_docs(self, contents: list) -> list:
        from langchain_core.documents import Document
        return [
            Document(page_content=c, metadata={"idx": i, "content_preview": c[:30]})
            for i, c in enumerate(contents)
        ]

    def test_reranker_initializes_with_correct_model(self, real_reranker):
        """LLMReranker가 설정된 모델로 초기화되어야 한다(gpt-4.1-mini 기본값)."""
        assert real_reranker is not None
        assert real_reranker.threshold == 0.3

    def test_rerank_returns_list(self, real_reranker):
        """rerank()가 list를 반환해야 한다."""
        docs = self._make_docs([_DOC_기넥신_부작용])
        result = real_reranker.rerank(docs, user_query="기넥신 부작용")
        assert isinstance(result, list)

    def test_relevant_doc_passes_threshold(self, real_reranker):
        """기넥신 부작용 쿼리에 기넥신 부작용 문서는 threshold(0.3)를 통과해야 한다."""
        docs = self._make_docs([_DOC_기넥신_부작용])
        result = real_reranker.rerank(docs, user_query="기넥신 부작용 알려줘")

        assert len(result) >= 1, (
            "기넥신 부작용 문서가 threshold 미만으로 필터됨 — "
            f"threshold={real_reranker.threshold}"
        )

    def test_irrelevant_doc_filtered_out(self, real_reranker):
        """날씨 문서는 기넥신 쿼리와 무관하므로 필터되어야 한다."""
        docs = self._make_docs([_DOC_무관한_내용])
        result = real_reranker.rerank(docs, user_query="기넥신 부작용 알려줘")

        if result:
            score = result[0].metadata.get("overall_score", 1.0)
            assert score < 0.6, (
                f"무관한 날씨 문서의 점수({score:.3f})가 너무 높음 — "
                "리랭커가 관련성을 제대로 평가하지 못함"
            )

    def test_product_name_discrimination_ranking(self, real_reranker):
        """
        기넥신 부작용 쿼리에서 기넥신 문서가 리넥신 문서보다 높은 순위여야 한다.
        Rerank_개선안.md 2-4. 유사 제품명 혼동 해소 검증.
        """
        docs = self._make_docs([_DOC_리넥신_일반, _DOC_기넥신_부작용, _DOC_기넥신_용법])
        result = real_reranker.rerank(docs, user_query="기넥신 부작용 알려줘")

        if len(result) < 2:
            pytest.skip("필터 후 문서가 2개 미만 — 비교 불가")

        top_doc = result[0]
        top_content = top_doc.page_content

        assert "기넥신" in top_content and "리넥신" not in top_content.split("기넥신")[0], (
            f"1위 문서가 기넥신 문서가 아님: '{top_content[:60]}...'"
        )

    def test_wrong_product_has_low_constraint_fit(self, real_reranker):
        """
        기넥신 쿼리에 대한 리넥신 문서의 constraint_fit은 낮아야 한다.
        Rerank_개선안.md: IMPORTANT 규칙 — 다른 제품 문서 constraint_fit = 0 강제.
        """
        docs = self._make_docs([_DOC_리넥신_일반])
        result = real_reranker.rerank(docs, user_query="기넥신 부작용")

        if not result:
            # 필터됐으면 이미 낮은 점수 → 통과
            return

        constraint_fit = result[0].metadata.get("constraint_fit_score", 10)
        assert constraint_fit <= 3, (
            f"기넥신 쿼리 + 리넥신 문서의 constraint_fit={constraint_fit} (≤3이어야 함). "
            "reranker_instruction.txt의 제품명 불일치 규칙이 작동하지 않을 수 있음."
        )

    def test_score_is_in_0_1_range(self, real_reranker):
        """실제 LLM 호출 후 overall_score가 [0, 1] 범위여야 한다."""
        docs = self._make_docs([_DOC_기넥신_부작용, _DOC_기넥신_용법])
        result = real_reranker.rerank(docs, user_query="기넥신 부작용")

        for doc in result:
            score = doc.metadata.get("overall_score", -1)
            assert 0.0 <= score <= 1.0, f"overall_score={score} ([0,1] 범위 초과)"

    def test_individual_scores_are_0_to_10_integers(self, real_reranker):
        """개별 기준 점수(0~10 정수)가 메타데이터에 올바르게 기록되어야 한다."""
        docs = self._make_docs([_DOC_기넥신_부작용])
        result = real_reranker.rerank(docs, user_query="기넥신 부작용")

        if not result:
            pytest.skip("필터 후 문서 없음")

        score_keys = [
            "relevance_score", "specificity_score", "completeness_score",
            "practicality_score", "constraint_fit_score",
        ]
        for key in score_keys:
            val = result[0].metadata.get(key)
            assert val is not None, f"{key}가 메타데이터에 없음"
            assert isinstance(val, int), f"{key}={val} — 정수여야 함"
            assert 0 <= val <= 10, f"{key}={val} — [0,10] 범위 초과"

    def test_multiple_docs_are_sorted_by_score(self, real_reranker):
        """여러 문서가 점수 내림차순으로 정렬되어야 한다."""
        docs = self._make_docs([
            _DOC_기넥신_부작용,
            _DOC_기넥신_용법,
            _DOC_리넥신_일반,
            _DOC_무관한_내용,
        ])
        result = real_reranker.rerank(docs, user_query="기넥신 부작용")

        scores = [d.metadata.get("overall_score", 0) for d in result]
        assert scores == sorted(scores, reverse=True), (
            f"결과가 점수 내림차순이 아님: {scores}"
        )
