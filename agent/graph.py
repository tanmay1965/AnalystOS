"""LangGraph workflow with clarification interrupt and bounded retry path."""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    clarification_node,
    critic_node,
    execute_tool_node,
    failure_node,
    planner_node,
    response_node,
    route_after_critic,
)
from agent.state import AnalystState

builder = StateGraph(AnalystState)
builder.add_node("clarification", clarification_node)
builder.add_node("planner", planner_node)
builder.add_node("execute_tool", execute_tool_node)
builder.add_node("critic", critic_node)
builder.add_node("respond", response_node)
builder.add_node("failure", failure_node)

builder.add_edge(START, "clarification")
builder.add_edge("clarification", "planner")
builder.add_edge("planner", "execute_tool")
builder.add_edge("execute_tool", "critic")
builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {"respond": "respond", "retry": "planner", "failure": "failure"},
)
builder.add_edge("respond", END)
builder.add_edge("failure", END)

# In-memory checkpoints support interrupt/resume during this running app process.
# SQLite investigation history is stored separately by agent.memory.
checkpointer = MemorySaver()
analyst_graph = builder.compile(checkpointer=checkpointer)
