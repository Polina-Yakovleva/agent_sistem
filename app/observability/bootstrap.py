"""Инициализация observability и сборка config для graph.invoke."""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from app.config import settings
from app.observability.callbacks import MetricsCallbackHandler
from app.observability.context import set_turn_context
from app.observability.langfuse import (
    flush_langfuse,
    get_langfuse_handler,
    init_langfuse,
    langfuse_metadata,
    langfuse_status,
)
from app.observability.llm_metrics import start_llm_metrics_collector
from app.observability.logging import configure_logging, get_logger
from app.observability.metrics import observe_turn, start_metrics_server
from app.observability.turn_collector import clear_turn_collector, start_turn_collector
from app.observability.turn_summary import build_turn_summary, log_turn_summary


def init_observability() -> None:
    """Вызвать один раз при старте приложения."""
    if not settings.observability_enabled:
        return
    configure_logging()
    start_metrics_server()
    start_llm_metrics_collector()
    init_langfuse()
    get_logger("agent").info(
        "observability_initialized",
        metrics_port=settings.metrics_port if settings.metrics_enabled else None,
        langfuse=langfuse_status(),
    )


def flush_observability() -> None:
    flush_langfuse()


@contextmanager
def start_turn_context(
    *,
    session_id: str,
    user_id: str,
    user_query: str = "",
    trace_id: str | None = None,
) -> Iterator[str]:
    """Контекст одного хода диалога; возвращает trace_id."""
    tid = trace_id or str(uuid.uuid4())
    if settings.observability_enabled:
        set_turn_context(trace_id=tid, session_id=session_id, user_id=user_id)
        start_turn_collector(trace_id=tid, user_query=user_query)
        get_logger("agent").info(
            "turn_start",
            query_length=len(user_query),
        )
    try:
        yield tid
    finally:
        if settings.observability_enabled:
            get_logger("agent").info("turn_end")


def finalize_turn_summary(
    *,
    user_query: str,
    trace_id: str,
    result: dict,
) -> dict[str, Any]:
    """Записать итоговый JSON хода (turn_summary) после graph.invoke.

    Возвращает собранный summary до очистки collector (для batch eval).
    """
    if not settings.observability_enabled:
        return {}
    from app.observability.turn_collector import get_turn_collector

    summary = build_turn_summary(
        collector=get_turn_collector(),
        result=result,
        user_query=user_query,
        trace_id=trace_id,
    )
    log_turn_summary(summary)
    clear_turn_collector()
    flush_observability()
    return summary


def build_invoke_config(
    *,
    session_id: str,
    user_id: str,
    trace_id: str,
    base_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Собрать config для graph.invoke с callbacks и metadata."""
    config: dict[str, Any] = dict(base_config or {})
    configurable = dict(config.get("configurable") or {})
    configurable.setdefault("thread_id", session_id)
    config["configurable"] = configurable

    if not settings.observability_enabled:
        return config

    callbacks: list[Any] = list(config.get("callbacks") or [])
    callbacks.append(MetricsCallbackHandler(default_caller="graph"))
    lf = get_langfuse_handler(
        session_id=session_id,
        user_id=user_id,
        trace_id=trace_id,
    )
    if lf is not None:
        callbacks.append(lf)

    config["callbacks"] = callbacks
    metadata = dict(config.get("metadata") or {})
    metadata.update(langfuse_metadata(session_id, user_id, trace_id))
    config["metadata"] = metadata
    return config


@contextmanager
def run_turn_observed() -> Iterator[None]:
    """Обёртка полного invoke: histogram agent_turn_duration_seconds."""
    with observe_turn():
        yield
