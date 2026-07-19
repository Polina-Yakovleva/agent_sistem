"""Сбор данных одного хода для turn_summary (золотой датасет, оценка RAG)."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter

_collector: ContextVar["TurnCollector | None"] = ContextVar("turn_collector", default=None)


@dataclass
class TurnCollector:
    """Аккумулятор событий хода graph.invoke."""

    trace_id: str = ""
    user_query: str = ""
    started_at: float = field(default_factory=perf_counter)
    tools_called: list[str] = field(default_factory=list)
    agents_invoked: list[str] = field(default_factory=list)
    rag_chunks_retrieved: int = 0

    def record_tool(self, name: str) -> None:
        tool = (name or "unknown").strip()
        if tool and tool not in self.tools_called:
            self.tools_called.append(tool)

    def record_agent(self, name: str) -> None:
        agent = (name or "").strip()
        if agent and agent not in self.agents_invoked:
            self.agents_invoked.append(agent)

    def add_rag_chunks(self, count: int) -> None:
        if count > 0:
            self.rag_chunks_retrieved += count

    @property
    def latency_ms(self) -> int:
        return int((perf_counter() - self.started_at) * 1000)


def start_turn_collector(*, trace_id: str, user_query: str) -> TurnCollector:
    col = TurnCollector(trace_id=trace_id, user_query=user_query)
    _collector.set(col)
    return col


def get_turn_collector() -> TurnCollector | None:
    return _collector.get()


def clear_turn_collector() -> None:
    _collector.set(None)


# Имена узлов графа → имена для отчёта (как в черновике ВКР).
_NODE_AGENT_NAMES: dict[str, str] = {
    "planner": "orchestrator",
    "aggregator": "orchestrator",
    "critic": "orchestrator",
    "revise": "orchestrator",
    "finalize": "orchestrator",
    "flight": "flight_agent",
    "booking": "booking_agent",
    "compliance": "compliance_agent",
    "external": "external_agent",
    "memory_read": "memory",
    "memory_write": "memory",
    "summarize_session": "memory",
}


def agent_name_for_node(node: str) -> str:
    return _NODE_AGENT_NAMES.get(node, node)


def agents_from_plan(plan: list | None) -> list[str]:
    """Субагенты из плана оркестратора."""
    names: list[str] = []
    for item in plan or []:
        agent = item.get("agent") if isinstance(item, dict) else getattr(item, "agent", None)
        if agent:
            names.append(f"{agent}_agent")
    return names


def merge_agents_invoked(collector: TurnCollector | None, plan: list | None) -> list[str]:
    """Объединить агентов из коллектора, плана и обязательного orchestrator."""
    ordered: list[str] = ["orchestrator"]
    for name in agents_from_plan(plan):
        if name not in ordered:
            ordered.append(name)
    if collector:
        for name in collector.agents_invoked:
            if name not in ordered:
                ordered.append(name)
    return ordered
