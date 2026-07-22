"""Прогон кейса через граф агента со сбором телеметрии."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from langchain_core.messages import HumanMessage
from langgraph.types import Command


@dataclass
class RunOutcome:
    """Результат одного хода/кейса для скореров."""

    case_id: str
    query: str
    status: str = "completed"  # completed | interrupted | error
    answer: str = ""
    plan: list[dict] = field(default_factory=list)
    initial_plan: list[dict] = field(default_factory=list)
    subagent_results: dict[str, str] = field(default_factory=dict)
    tools_called: list[str] = field(default_factory=list)
    agents_used: list[str] = field(default_factory=list)
    rag_chunks: int = 0
    revision_count: int = 0
    critic_ok: bool = False
    interrupts: int = 0
    latency_ms: int = 0
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_answer(self) -> bool:
        text = (self.answer or "").strip()
        return bool(text) and text != "(пустой ответ)"


def _plan_agents(plan: list[dict] | None) -> set[str]:
    out: set[str] = set()
    for item in plan or []:
        agent = item.get("agent") if isinstance(item, dict) else getattr(item, "agent", None)
        if agent:
            out.add(str(agent))
    return out


def _agents_used(result: dict[str, Any]) -> list[str]:
    """Субагенты, реально задействованные в ходе (план + полученные результаты)."""
    agents = _plan_agents(result.get("plan"))
    agents |= set((result.get("subagent_results") or {}).keys())
    return sorted(f"{a}_agent" for a in agents)


class AgentRunner:
    """Долгоживущий граф для батч-оценки (один инстанс на прогон)."""

    def __init__(
        self,
        *,
        enable_langfuse: bool = False,
        graph_builder: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._graph = None
        self._enable_langfuse = enable_langfuse
        self._graph_builder = graph_builder

    def __enter__(self) -> "AgentRunner":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def start(self) -> None:
        from app.agents.graph import build_graph
        from app.config import settings

        # Изоляция прогона: без LTM и без внешней телеметрии Langfuse.
        settings.eval_mode = True
        if not self._enable_langfuse:
            settings.langfuse_enabled = False
        builder = self._graph_builder or build_graph
        self._graph = builder()  # InMemorySaver внутри

    def stop(self) -> None:
        self._graph = None

    def run_turn(
        self,
        query: str,
        *,
        case_id: str = "",
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        hitl_decision: str = "да",
        max_resumes: int = 3,
    ) -> RunOutcome:
        """Выполнить один ход и (при необходимости) автоматически пройти HITL."""
        if self._graph is None:
            raise RuntimeError("AgentRunner is not started")

        from app.config import settings
        from app.observability import build_invoke_config, start_turn_context
        from app.observability.turn_collector import (
            clear_turn_collector,
            get_turn_collector,
        )
        from app.runtime import set_agent_user_id

        session_id = session_id or str(uuid.uuid4())
        user_id = user_id or settings.agent_user_id
        set_agent_user_id(user_id)
        base_config = {"configurable": {"thread_id": session_id}}

        interrupts = 0
        with start_turn_context(
            session_id=session_id, user_id=user_id, user_query=query
        ) as trace_id:
            config = build_invoke_config(
                session_id=session_id,
                user_id=user_id,
                trace_id=trace_id,
                base_config=base_config,
            )
            try:
                result = self._graph.invoke(
                    {
                        "messages": [HumanMessage(content=query)],
                        "user_query": query,
                        "session_id": session_id,
                        "user_id": user_id,
                    },
                    config,
                )
                resumes = 0
                while result.get("__interrupt__") and resumes < max_resumes:
                    interrupts += 1
                    resumes += 1
                    result = self._graph.invoke(Command(resume=hitl_decision), config)
            except Exception as exc:  # noqa: BLE001 — фиксируем как красный кейс
                clear_turn_collector()
                return RunOutcome(
                    case_id=case_id,
                    query=query,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )

            collector = get_turn_collector()
            tools = list(collector.tools_called) if collector else []
            rag_chunks = int(collector.rag_chunks_retrieved) if collector else 0
            latency = collector.latency_ms if collector else 0
            clear_turn_collector()

        status = "interrupted" if result.get("__interrupt__") else "completed"
        answer = result.get("final_answer") or result.get("draft_answer") or ""
        return RunOutcome(
            case_id=case_id,
            query=query,
            status=status,
            answer=answer,
            plan=list(result.get("plan") or []),
            initial_plan=list(result.get("initial_plan") or []),
            subagent_results=dict(result.get("subagent_results") or {}),
            tools_called=tools,
            agents_used=_agents_used(result),
            rag_chunks=rag_chunks,
            revision_count=int(result.get("revision_count") or 0),
            critic_ok=bool(result.get("critic_ok")),
            interrupts=interrupts,
            latency_ms=latency,
            raw=result,
        )
