"""Prometheus-метрики агентной системы."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from time import perf_counter
from typing import Iterator

from prometheus_client import Counter, Histogram, start_http_server

from app.config import settings

TURN_DURATION = Histogram(
    "agent_turn_duration_seconds",
    "Длительность полного хода graph.invoke",
    buckets=(0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600),
)
NODE_DURATION = Histogram(
    "agent_node_duration_seconds",
    "Длительность узла LangGraph",
    ["node"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)
LLM_CALLS = Counter(
    "agent_llm_calls_total",
    "Вызовы LLM",
    ["caller"],
)
TOOL_CALLS = Counter(
    "agent_tool_calls_total",
    "Вызовы инструментов",
    ["tool", "status"],
)
CRITIC_REVISIONS = Counter(
    "agent_critic_revisions_total",
    "Циклы доработки после критика",
)
HITL_INTERRUPTS = Counter(
    "agent_hitl_interrupts_total",
    "Прерывания Human-in-the-loop",
    ["tool"],
)
ERRORS = Counter(
    "agent_errors_total",
    "Ошибки по компонентам",
    ["component"],
)
DB_QUERY_DURATION = Histogram(
    "db_query_duration_seconds",
    "Длительность SQL-запроса",
    ["operation"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5),
)
EXTERNAL_HTTP_DURATION = Histogram(
    "external_http_duration_seconds",
    "Длительность HTTP к внешним API",
    ["endpoint"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 15, 30),
)

_metrics_server_started = False
_metrics_lock = threading.Lock()


def start_metrics_server() -> None:
    global _metrics_server_started
    if not settings.metrics_enabled or not settings.observability_enabled:
        return
    with _metrics_lock:
        if _metrics_server_started:
            return
        start_http_server(settings.metrics_port)
        _metrics_server_started = True


@contextmanager
def observe_turn() -> Iterator[None]:
    if not settings.metrics_enabled:
        yield
        return
    start = perf_counter()
    try:
        yield
    finally:
        TURN_DURATION.observe(perf_counter() - start)


@contextmanager
def observe_node(node: str) -> Iterator[None]:
    if not settings.metrics_enabled:
        yield
        return
    start = perf_counter()
    try:
        yield
    finally:
        NODE_DURATION.labels(node=node).observe(perf_counter() - start)


@contextmanager
def observe_db(operation: str) -> Iterator[None]:
    if not settings.metrics_enabled:
        yield
        return
    start = perf_counter()
    try:
        yield
    except Exception:
        record_error("db")
        raise
    finally:
        DB_QUERY_DURATION.labels(operation=operation).observe(perf_counter() - start)


@contextmanager
def observe_external(endpoint: str) -> Iterator[None]:
    if not settings.metrics_enabled:
        yield
        return
    start = perf_counter()
    try:
        yield
    except Exception:
        record_error("external")
        raise
    finally:
        EXTERNAL_HTTP_DURATION.labels(endpoint=endpoint).observe(perf_counter() - start)


def record_llm_call(caller: str) -> None:
    if settings.metrics_enabled:
        LLM_CALLS.labels(caller=caller).inc()


def record_tool_call(tool: str, *, success: bool = True) -> None:
    if settings.metrics_enabled:
        status = "success" if success else "error"
        TOOL_CALLS.labels(tool=tool, status=status).inc()


def record_critic_revision() -> None:
    if settings.metrics_enabled:
        CRITIC_REVISIONS.inc()


def record_hitl_interrupt(tool: str) -> None:
    if settings.metrics_enabled:
        HITL_INTERRUPTS.labels(tool=tool).inc()


def record_error(component: str) -> None:
    if settings.metrics_enabled:
        ERRORS.labels(component=component).inc()
