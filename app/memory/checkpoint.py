"""Фабрика checkpointer LangGraph (Postgres или in-memory)."""

from contextlib import contextmanager
from typing import Iterator

from langgraph.checkpoint.memory import InMemorySaver

from app.config import settings
from app.observability.logging import get_logger

_log = get_logger("memory.checkpoint")


@contextmanager
def get_checkpointer() -> Iterator:
    """Контекстный менеджер checkpointer на всю сессию CLI/API.

    При недоступности Postgres тихо откатывается на InMemorySaver (без
    персистентности между процессами) — но пишет warning, чтобы это не
    осталось незамеченным в проде.
    """
    backend = (settings.checkpoint_backend or "postgres").strip().lower()
    pg_ctx = None
    checkpointer = None

    if backend == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            pg_ctx = PostgresSaver.from_conn_string(settings.pg_url)
            checkpointer = pg_ctx.__enter__()
            checkpointer.setup()
        except Exception as exc:
            if pg_ctx is not None:
                pg_ctx.__exit__(None, None, None)
            pg_ctx = None
            checkpointer = None
            _log.warning(
                "checkpointer_postgres_unavailable",
                error=str(exc),
                fallback="memory",
            )

    if checkpointer is None:
        checkpointer = InMemorySaver()

    try:
        yield checkpointer
    finally:
        if pg_ctx is not None:
            pg_ctx.__exit__(None, None, None)
