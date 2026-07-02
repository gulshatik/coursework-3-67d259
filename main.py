#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo of a LangGraph agent with memory, interrupt-before-tool-call,
and human-in-the-loop confirmation.  The script runs without
interactive input – all user messages and confirmations are
predefined.
"""

import os
import sys
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console

# --------------------------------------------------------------------------- #
# 1. Load environment and create LLM
# --------------------------------------------------------------------------- #
load_dotenv()
llm = ChatOpenAI(
    base_url="https://llm.brojs.ru/v1",
    api_key=os.getenv("BROJS_PAT_TOKEN"),
    model="openai/gpt-oss-20b",
    temperature=0.1,
)

# --------------------------------------------------------------------------- #
# 2. Define a simple tool
# --------------------------------------------------------------------------- #
@tool
def get_price(city: str, date: str) -> str:
    """
    Return a fake price for a city on a given date.
    """
    return f"Цена в {city} на {date} составляет 100₽"

# --------------------------------------------------------------------------- #
# 3. Build the agent graph
# --------------------------------------------------------------------------- #
class State(dict):
    """
    State type for the graph.  The `messages` field is a list of BaseMessage
    objects and is automatically updated by the `add_messages` transformer.
    """
    messages: List[BaseMessage]

def call_model(state: State) -> Dict[str, List[BaseMessage]]:
    """
    Call the LLM with the current messages and return a new AIMessage.
    """
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build the graph
builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools=[get_price]))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

# --------------------------------------------------------------------------- #
# 4. Memory and interrupt-before configuration
# --------------------------------------------------------------------------- #
memory = MemorySaver()
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["tools"],  # pause before each tool call
)

# --------------------------------------------------------------------------- #
# 5. Console for pretty output
# --------------------------------------------------------------------------- #
console = Console()

# --------------------------------------------------------------------------- #
# 6. Global configuration and simulated user data
# --------------------------------------------------------------------------- #
config = {"configurable": {"thread_id": "conversation-1"}}

# Predefined user messages (no interactive input)
user_messages: List[str] = [
    "Какая погода в Казани сегодня и завтра?",
    "А там же через неделю?",
]

# Predefined confirmations for each tool call (True = allow, False = deny)
confirmations: List[bool] = [True, False, True]

# --------------------------------------------------------------------------- #
# 7. Core function that streams the agent and handles pauses
# --------------------------------------------------------------------------- #
def ask_and_run(user_input: Optional[str], cfg: Dict[str, Any]) -> None:
    """
    Stream the agent's response to `user_input`.  If the agent is about to
    call a tool, pause and ask for confirmation.  The function is
    recursive – when the user allows the tool call, it resumes from the
    paused state.
    """
    # Prepare the input payload
    if user_input is not None:
        payload = {"messages": [HumanMessage(content=user_input)]}
    else:
        payload = {}

    # Stream the agent
    for chunk_type, chunk_data in graph.stream(
        payload, config=cfg, stream_mode=["messages", "updates"]
    ):
        # Current state of the graph (needed for pause detection)
        state = graph.get_state(cfg)

        # 1. Normal token streaming
        if chunk_type == "messages":
            for msg in chunk_data:
                if isinstance(msg, BaseMessage) and hasattr(msg, "content"):
                    # Write raw UTF‑8 bytes to avoid console encoding issues
                    sys.stdout.buffer.write(msg.content.encode("utf-8"))
                    sys.stdout.buffer.flush()

        # 2. Detect pause before tool call
        if "__interrupt__" in chunk_data and state.next == ("tools",):
            # The last AI message contains the pending tool call
            last_msg = state.values["messages"][-1]
            if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                tool_call = last_msg.tool_calls[0]
                name = tool_call["name"]
                args = tool_call["args"]
                console.print("\n")
                console.print(f"[bold cyan]Агент хочет вызвать утилиту {name}({args})[/bold cyan]")
                # Get confirmation from the predefined list
                if confirmations:
                    allow = confirmations.pop(0)
                else:
                    allow = True  # default to allow if list exhausted
                console.print(f"[bold yellow]Разрешить? (Y/n): {allow}[/bold yellow]")
                if allow:
                    console.print("[bold green]Разрешено. Продолжаем...[/bold green]\n")
                    # Recursively resume from the paused state
                    ask_and_run(None, cfg)
                else:
                    console.print("[bold red]Отменено[/bold red]\n")
                # Stop processing the current stream after handling the pause
                break

# --------------------------------------------------------------------------- #
# 8. Run the simulated chat loop
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    console.print("[bold magenta]=== Demo: Agent with Memory & Confirmation ===[/bold magenta]\n")
    for msg in user_messages:
        console.print(f"[bold]Вы:[/bold] {msg}")
        ask_and_run(msg, config)
        console.print("\n" + "-" * 40 + "\n")
