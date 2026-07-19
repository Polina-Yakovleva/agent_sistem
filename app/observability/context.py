"""Контекст корреляции для логов и метрик (contextvars)."""

from __future__ import annotations

import hashlib
from contextvars import ContextVar

_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_session_id: ContextVar[str] = ContextVar("session_id", default="")
_user_id_hash: ContextVar[str] = ContextVar("user_id_hash", default="")
_node: ContextVar[str] = ContextVar("node", default="")


def hash_user_id(user_id: str) -> str:
    """Стабильный короткий хэш идентификатора пользователя (без ПДн в логах)."""
    raw = (user_id or "default").strip() or "default"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def set_turn_context(
    *,
    trace_id: str,
    session_id: str,
    user_id: str,
) -> None:
    _trace_id.set(trace_id)
    _session_id.set(session_id)
    _user_id_hash.set(hash_user_id(user_id))


def set_node_context(node: str) -> None:
    _node.set(node)


def clear_node_context() -> None:
    _node.set("")


def get_trace_id() -> str:
    return _trace_id.get()


def get_session_id() -> str:
    return _session_id.get()


def get_user_id_hash() -> str:
    return _user_id_hash.get()


def get_node() -> str:
    return _node.get()


def context_dict() -> dict[str, str]:
    """Поля для structlog contextvars."""
    out: dict[str, str] = {}
    if get_trace_id():
        out["trace_id"] = get_trace_id()
    if get_session_id():
        out["session_id"] = get_session_id()
    if get_user_id_hash():
        out["user_id_hash"] = get_user_id_hash()
    node = get_node()
    if node:
        out["node"] = node
    return out


def bind_log_context(**extra: str) -> None:
    """Дополнительные поля в текущем контексте логирования (опционально)."""
    _ = extra  # structlog merge через processors
