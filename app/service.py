"""Общий сервисный слой диалога для CLI и HTTP API."""

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.agents.graph import build_graph
from app.config import settings
from app.memory.checkpoint import get_checkpointer
from app.observability import (
    build_invoke_config,
    finalize_turn_summary,
    start_turn_context,
)
from app.observability.bootstrap import run_turn_observed
from app.observability.metrics import record_hitl_interrupt
from app.runtime import set_agent_user_id


@dataclass
class InterruptInfo:
    type: str = "booking"
    summary: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnResult:
    session_id: str
    user_id: str
    status: str  # completed | interrupted
    answer: Optional[str] = None
    interrupt: Optional[InterruptInfo] = None
    raw: Optional[dict[str, Any]] = None


def _extract_answer(result: dict[str, Any]) -> str:
    return result.get("final_answer") or result.get("draft_answer") or "(пустой ответ)"


def _interrupt_info(result: dict[str, Any]) -> InterruptInfo:
    interrupt = result["__interrupt__"][0]
    payload = interrupt.value if isinstance(interrupt.value, dict) else {}
    summary = payload.get("summary", str(interrupt.value))
    return InterruptInfo(
        type=payload.get("type", "booking"),
        summary=summary,
        payload=payload if isinstance(payload, dict) else {"value": interrupt.value},
    )


class AgentService:
    """Долгоживущий граф + checkpointer (для FastAPI lifespan / CLI-сессии)."""

    def __init__(self) -> None:
        self._cm = None
        self._checkpointer = None
        self.graph = None

    def start(self) -> None:
        self._cm = get_checkpointer()
        self._checkpointer = self._cm.__enter__()
        self.graph = build_graph(checkpointer=self._checkpointer)

    def stop(self) -> None:
        if self._cm is not None:
            self._cm.__exit__(None, None, None)
            self._cm = None
            self._checkpointer = None
            self.graph = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    def run_turn(
        self,
        query: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> TurnResult:
        if self.graph is None:
            raise RuntimeError("AgentService is not started")

        session_id = (session_id or "").strip() or str(uuid.uuid4())
        user_id = (user_id or "").strip() or settings.agent_user_id
        set_agent_user_id(user_id)
        base_config = {"configurable": {"thread_id": session_id}}

        with start_turn_context(
            session_id=session_id,
            user_id=user_id,
            user_query=query,
        ) as trace_id:
            config = build_invoke_config(
                session_id=session_id,
                user_id=user_id,
                trace_id=trace_id,
                base_config=base_config,
            )
            with run_turn_observed():
                result = self.graph.invoke(
                    {
                        "messages": [HumanMessage(content=query)],
                        "user_query": query,
                        "session_id": session_id,
                        "user_id": user_id,
                    },
                    config,
                )

            if result.get("__interrupt__"):
                record_hitl_interrupt("booking")
                info = _interrupt_info(result)
                return TurnResult(
                    session_id=session_id,
                    user_id=user_id,
                    status="interrupted",
                    answer=None,
                    interrupt=info,
                    raw=result,
                )

            finalize_turn_summary(user_query=query, trace_id=trace_id, result=result)
            return TurnResult(
                session_id=session_id,
                user_id=user_id,
                status="completed",
                answer=_extract_answer(result),
                interrupt=None,
                raw=result,
            )

    def resume_turn(
        self,
        session_id: str,
        decision: str,
        user_id: Optional[str] = None,
        user_query: str = "",
    ) -> TurnResult:
        if self.graph is None:
            raise RuntimeError("AgentService is not started")

        session_id = session_id.strip()
        user_id = (user_id or "").strip() or settings.agent_user_id
        set_agent_user_id(user_id)
        base_config = {"configurable": {"thread_id": session_id}}

        with start_turn_context(
            session_id=session_id,
            user_id=user_id,
            user_query=user_query or "(resume)",
        ) as trace_id:
            config = build_invoke_config(
                session_id=session_id,
                user_id=user_id,
                trace_id=trace_id,
                base_config=base_config,
            )
            with run_turn_observed():
                result = self.graph.invoke(Command(resume=decision), config)

            if result.get("__interrupt__"):
                record_hitl_interrupt("booking")
                info = _interrupt_info(result)
                return TurnResult(
                    session_id=session_id,
                    user_id=user_id,
                    status="interrupted",
                    answer=None,
                    interrupt=info,
                    raw=result,
                )

            finalize_turn_summary(
                user_query=user_query or "(resume)",
                trace_id=trace_id,
                result=result,
            )
            return TurnResult(
                session_id=session_id,
                user_id=user_id,
                status="completed",
                answer=_extract_answer(result),
                interrupt=None,
                raw=result,
            )


def run_turn_blocking_hitl(
    service: AgentService,
    query: str,
    session_id: str,
    user_id: str,
) -> TurnResult:
    """CLI-хелпер: крутит HITL через input(), пока статус не completed."""
    result = service.run_turn(query=query, session_id=session_id, user_id=user_id)
    while result.status == "interrupted":
        assert result.interrupt is not None
        print("\n[Требуется подтверждение]")
        print(result.interrupt.summary)
        decision = input("Подтвердите операцию (да/нет): ").strip()
        result = service.resume_turn(
            session_id=result.session_id,
            user_id=result.user_id,
            decision=decision,
            user_query=query,
        )
    return result
