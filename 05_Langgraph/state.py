import operator
from typing import TypedDict
from typing_extensions import Annotated
from langchain_core.messages import BaseMessage

class userMessage(TypedDict):
    user_message: str

class agentMessage(TypedDict):
    agent_message: str

class messageHistory(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]