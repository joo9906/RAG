from dotenv import load_dotenv
import os
import weaviate
import weaviate.classes as wvc

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY", "")
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))

client = weaviate.connect_to_custom(
    http_host=WEAVIATE_HOST,
    http_port=WEAVIATE_HTTP_PORT,
    grpc_host=WEAVIATE_HOST,
    grpc_port=WEAVIATE_GRPC_PORT,
    http_secure=False,
    grpc_secure=False,
    )

print(client.is_ready())

first_collection = client.collections.get("FirstCollection")

from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 1단계: 데이터 저장소 만들기 (State)
class MyState(TypedDict):
    message: str

# 2단계: 작업 함수 만들기 (Node)
def say_hello(state):
    return {"message": "Hello, LangGraph!"}

# 3단계: 그래프 만들기
graph = StateGraph(MyState)
graph.add_node("hello", say_hello)
graph.add_edge(START, "hello")
graph.add_edge("hello", END)

# 4단계: 실행하기
app = graph.compile()
result = app.invoke({"message": ""})
print(result)
