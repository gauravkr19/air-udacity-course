
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
api_key=str(os.getenv("API_KEY"))
base_url=str(os.getenv("BASE_URL"))

class State(TypedDict):
    question: str
    answer: str

llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    api_key=api_key,
    base_url=base_url,
    temperature=0.7,
)

def call_model(state: State):
    resp = llm.invoke(state["question"])
    return {"answer": resp.content}

graph = StateGraph(State)
graph.add_node("call_model", call_model)
graph.add_edge(START, "call_model")
graph.add_edge("call_model", END)

app = graph.compile()
result = app.invoke({"question": "Explain LangChain in one paragraph", "answer": ""})
print(result["answer"])
