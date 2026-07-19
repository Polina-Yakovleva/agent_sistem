"""Сборка графа мультиагентной системы.

Топология:

    START → memory_read → summarize_session → planner ──(dispatch)──▶ субагенты
                       │                              │
                       └──────────────────────────────▶ aggregator → critic
                                                              ├─ finalize → memory_write → END
                                                              └─ revise → planner

Граф компилируется с checkpointer (Postgres или InMemorySaver) для цикла
доработки, STM между репликами и Human-in-the-loop.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.critic import (
    critic_node,
    critic_router,
    finalize_node,
    revise_node,
)
from app.agents.orchestrator import aggregator_node, dispatch, planner_node
from app.agents.state import AgentState
from app.agents.subagents import SUBAGENT_NODES
from app.memory.nodes import (
    memory_read_node,
    memory_write_node,
    summarize_session_node,
)
from app.observability.instrumentation import wrap_graph_node


def analysis_done_node(state: AgentState) -> dict:
    """Конец подсистемы анализа запросов (без aggregator/critic)."""
    _ = state
    return {}


def build_analysis_graph(checkpointer=None):
    """Граф только подсистемы анализа: до subagent_results, без генерации ответа."""
    builder = StateGraph(AgentState)

    builder.add_node("memory_read", wrap_graph_node("memory_read", memory_read_node))
    builder.add_node(
        "summarize_session",
        wrap_graph_node("summarize_session", summarize_session_node),
    )
    builder.add_node("planner", wrap_graph_node("planner", planner_node))
    for name, node in SUBAGENT_NODES.items():
        builder.add_node(name, wrap_graph_node(name, node))
    builder.add_node(
        "analysis_done",
        wrap_graph_node("analysis_done", analysis_done_node),
    )

    builder.add_edge(START, "memory_read")
    builder.add_edge("memory_read", "summarize_session")
    builder.add_edge("summarize_session", "planner")

    builder.add_conditional_edges(
        "planner",
        lambda state: dispatch(state, empty_target="analysis_done"),
        ["analysis_done", *SUBAGENT_NODES.keys()],
    )

    for name in SUBAGENT_NODES:
        builder.add_edge(name, "analysis_done")

    builder.add_edge("analysis_done", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def build_graph(checkpointer=None):
    """Собрать и скомпилировать полный граф."""
    builder = StateGraph(AgentState)

    builder.add_node("memory_read", wrap_graph_node("memory_read", memory_read_node))
    builder.add_node(
        "summarize_session",
        wrap_graph_node("summarize_session", summarize_session_node),
    )
    builder.add_node("planner", wrap_graph_node("planner", planner_node))
    for name, node in SUBAGENT_NODES.items():
        builder.add_node(name, wrap_graph_node(name, node))
    builder.add_node("aggregator", wrap_graph_node("aggregator", aggregator_node))
    builder.add_node("critic", wrap_graph_node("critic", critic_node))
    builder.add_node("finalize", wrap_graph_node("finalize", finalize_node))
    builder.add_node("revise", wrap_graph_node("revise", revise_node))
    builder.add_node("memory_write", wrap_graph_node("memory_write", memory_write_node))

    builder.add_edge(START, "memory_read")
    builder.add_edge("memory_read", "summarize_session")
    builder.add_edge("summarize_session", "planner")

    builder.add_conditional_edges(
        "planner",
        dispatch,
        ["aggregator", *SUBAGENT_NODES.keys()],
    )

    for name in SUBAGENT_NODES:
        builder.add_edge(name, "aggregator")

    builder.add_edge("aggregator", "critic")

    builder.add_conditional_edges(
        "critic",
        critic_router,
        {"finalize": "finalize", "revise": "revise"},
    )
    builder.add_edge("revise", "planner")
    builder.add_edge("finalize", "memory_write")
    builder.add_edge("memory_write", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())
