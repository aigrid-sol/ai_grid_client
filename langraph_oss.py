"""
LangGraph example using OSS (OpenAI-compatible) backend.

Flow: user message -> generate (LLM) -> [conditional] -> refine (LLM) or end -> result.
Reads BASE_URL, AI_GRID_KEY, OSS_MODEL from .env.
"""
import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()

AI_GRID_KEY = os.getenv("AI_GRID_KEY")
BASE_URL = os.getenv("BASE_URL")
OSS_MODEL = os.getenv("OSS_MODEL", "gpt-oss-120b")

REFINE_THRESHOLD = 200


class State(TypedDict):
    """
    Graph state.
    messages: full conversation (user + assistant).
    step: number of LLM calls so far.
    """
    messages: Annotated[list, add_messages]
    step: int


def get_llm(temperature: float = 0.7) -> ChatOpenAI:
    """Build ChatOpenAI pointing at the OSS endpoint."""
    return ChatOpenAI(
        model=OSS_MODEL,
        openai_api_key=AI_GRID_KEY,
        openai_api_base=BASE_URL,
        temperature=temperature,
    )


def generate(state: State) -> dict:
    """
    First node: call the LLM with the current messages and append the assistant reply.
    """
    llm = get_llm(temperature=0.7)
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "step": state.get("step", 0) + 1,
    }


def should_refine(state: State) -> Literal["refine", "end"]:
    """
    After generate: refine only if the last assistant message is longer than REFINE_THRESHOLD chars.
    """
    messages = state["messages"]
    if not messages:
        return "end"
    last = messages[-1]
    if isinstance(last, AIMessage) and last.content and len(last.content) > REFINE_THRESHOLD:
        return "refine"
    return "end"


def refine(state: State) -> dict:
    """
    Second node: ask the LLM to summarize the last long reply in one short paragraph.
    """
    messages = state["messages"]
    last_content = messages[-1].content if messages else ""
    system = (
        "You are a summarizer. Given the assistant reply below, "
        "output only a short paragraph (2-4 sentences) that captures the main points. "
        "No preamble."
    )
    llm = get_llm(temperature=0.3)
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Assistant reply to summarize:\n\n{last_content}"),
    ])
    return {
        "messages": [AIMessage(content=f"[Summary] {response.content}")],
        "step": state.get("step", 0) + 1,
    }


def build_graph() -> StateGraph:
    """
    Build the graph: START -> generate -> (refine or end) -> END.
    """
    graph = StateGraph(State)
    graph.add_node("generate", generate)
    graph.add_node("refine", refine)
    graph.add_edge(START, "generate")
    graph.add_conditional_edges("generate", should_refine, {"refine": "refine", "end": END})
    graph.add_edge("refine", END)
    return graph.compile()


def run(user_message: str) -> State:
    """Run the graph with one user message and return the final state."""
    app = build_graph()
    initial: State = {"messages": [HumanMessage(content=user_message)], "step": 0}
    return app.invoke(initial)


def run_stream(user_message: str):
    """
    Run the graph and yield (node_name, state) after each node.
    Useful for logging or UI progress.
    """
    app = build_graph()
    initial: State = {"messages": [HumanMessage(content=user_message)], "step": 0}
    for event in app.stream(initial):
        for node_name, node_state in event.items():
            yield node_name, node_state


def main():
    """Run the graph and print the final assistant message(s)."""
    user_input = (
        "Explain in a few sentences how a compiler works, "
        "then give a one-sentence summary at the end."
    )
    print("User:", user_input)
    print()
    result = run(user_input)
    print("Steps:", result.get("step", 0))
    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print("Assistant:", msg.content)
            print()
    if not any(isinstance(m, AIMessage) for m in result["messages"]):
        print("Result:", result)


if __name__ == "__main__":
    main()
