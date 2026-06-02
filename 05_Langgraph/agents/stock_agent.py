"""Stock Agent — wraps the MCP stock server via langchain-mcp-adapters."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent

_MCP_SERVER_PATH = str(
    Path(__file__).parent.parent / "mcp_server" / "stock_mcp_server.py"
)

_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

_SYSTEM_PROMPT = (
    "당신은 금융 주식 전문 AI 어시스턴트입니다. "
    "주식 관련 도구를 사용하여 사용자가 요청한 주식 정보를 조회하고, "
    "명확하고 이해하기 쉽게 설명해 주세요. "
    "티커 심볼을 모를 경우 올바른 심볼을 추정하여 조회하세요. "
    "(한국 주식: 005930.KS=삼성전자, 000660.KS=SK하이닉스 등)"
)


async def stock_node(state: MessagesState) -> dict[str, Any]:
    """LangGraph node: connects to MCP stock server and answers stock queries."""
    async with MultiServerMCPClient(
        {
            "stock": {
                "command": sys.executable,
                "args": [_MCP_SERVER_PATH],
                "transport": "stdio",
            }
        }
    ) as client:
        tools = client.get_tools()

        agent = create_react_agent(
            model=_llm,
            tools=tools,
            state_modifier=_SYSTEM_PROMPT,
        )

        result = await agent.ainvoke({"messages": state["messages"]})

    from langchain_core.messages import AIMessage

    last_ai = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    reply = last_ai.content if last_ai else "주식 정보를 가져오지 못했습니다."

    return {"messages": [AIMessage(content=f"## 📈 주식 정보\n\n{reply}")]}
