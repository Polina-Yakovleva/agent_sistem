"""Контекст текущего хода (AGENT_USER_ID для инструментов)."""

from contextvars import ContextVar

_agent_user_id: ContextVar[str] = ContextVar("agent_user_id", default="default")


def set_agent_user_id(user_id: str) -> None:
    _agent_user_id.set((user_id or "").strip() or "default")


def get_agent_user_id() -> str:
    return _agent_user_id.get()
