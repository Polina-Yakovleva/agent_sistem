"""Декораторы инструментации узлов графа."""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from app.config import settings
from app.observability.context import clear_node_context, set_node_context
from app.observability.logging import get_logger
from app.observability.metrics import observe_node
from app.observability.turn_collector import agent_name_for_node, get_turn_collector

F = TypeVar("F", bound=Callable[..., Any])


def observe_node_fn(node_name: str) -> Callable[[F], F]:
    """Обёртка узла LangGraph: метрики, логи, контекст node."""

    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not settings.observability_enabled:
                return fn(*args, **kwargs)

            logger = get_logger("agent.node")
            set_node_context(node_name)
            col = get_turn_collector()
            if col:
                col.record_agent(agent_name_for_node(node_name))
            logger.info("node_start", node=node_name)
            try:
                with observe_node(node_name):
                    return fn(*args, **kwargs)
            except Exception:
                logger.exception("node_error", node=node_name)
                raise
            finally:
                logger.info("node_end", node=node_name)
                clear_node_context()

        return wrapper  # type: ignore[return-value]

    return decorator


def wrap_graph_node(node_name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Применить observe_node_fn к функции узла."""
    if not settings.observability_enabled:
        return fn
    return observe_node_fn(node_name)(fn)
