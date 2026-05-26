from langgraph.graph import StateGraph, START, END
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.chat_models import init_chat_model
from state import messageHistory

# 환경 변수 로드 (.env 파일이 상위 폴더에 위치하므로 주 경로 설정 고려)
load_dotenv()

# API 키 설정 확인
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("⚠️ 경고: OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
# init_chat_model을 사용하여 OpenAI GPT-4o-mini 모델 초기화
gpt_4o_mini = init_chat_model(
    "gpt-4o-mini",
    model_provider="openai",
    api_key=OPENAI_API_KEY,
    temperature=0.7,
    max_tokens=300,
)

# 1. 노드 정의 (Chatbot Node)
def chatbot_node(state: messageHistory):
    # state["messages"]에 축적된 대화 목록을 GPT-4o-mini에 전달하여 답변 생성
    response = gpt_4o_mini.invoke(state["messages"])
    # 생성된 답변 메시지를 리스트 형태로 반환 (state.py의 operator.add에 의해 자동 병합됨)
    return {"messages": [response]}

# 2. 그래프 빌더 생성 및 노드 추가
workflow = StateGraph(messageHistory)
workflow.add_node("chatbot", chatbot_node)

# 3. 관계 설정 (시작 지점 -> chatbot 노드 -> 종료 지점)
workflow.add_edge(START, "chatbot")
workflow.add_edge("chatbot", END)

# 4. 그래프 컴파일
app = workflow.compile()

# 5. 콘솔 대화 인터페이스 실행부
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 LangGraph & GPT-4o-mini 기반 챗봇 시작")
    print(" - 종료하려면 'exit' 또는 'quit'을 입력하세요.")
    print("="*60)
    
    # 챗봇 상태 초기화 (messages 리스트 시작)
    state = {"messages": []}
    
    while True:
        try:
            # 사용자 입력 받기
            user_input = input("\n👤 User: ").strip()
            if not user_input:
                continue
            
            # 종료 조건 확인
            if user_input.lower() in ["exit", "quit"]:
                print("\n👋 챗봇 대화를 종료합니다. 감사합니다!")
                print("="*60 + "\n")
                break
            
            # 사용자 메시지를 대화 상태에 추가
            state["messages"].append(HumanMessage(content=user_input))
            
            # LangGraph 실행 및 응답 가져오기
            print("🤖 Assistant: ", end="", flush=True)
            result = app.invoke(state)
            
            # 최신 상태 업데이트 (이전 대화 내역이 유지됨)
            state = result
            
            # 마지막으로 생성된 AI 응답 메시지 출력
            last_message = result["messages"][-1]
            print(last_message.content)
            
        except KeyboardInterrupt:
            print("\n👋 인터럽트로 인해 챗봇을 종료합니다. 감사합니다!")
            print("="*60 + "\n")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ 에러 발생: {e}")
