"""Структурированное логирование (structlog → stdout для Loki/Promtail)."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.config import settings
from app.memory.privacy import redact_pii_value
from app.observability.context import context_dict

_configured = False


def configure_logging() -> None:
    global _configured
    if _configured or not settings.observability_enabled:
        return

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        _inject_obs_context,
    ]
    if settings.log_redact_pii:
        shared_processors.append(_redact_pii_processor)

    if settings.log_format == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(level=log_level, handlers=[])
    _configured = True


def _inject_obs_context(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    for key, value in context_dict().items():
        event_dict.setdefault(key, value)
    return event_dict


def _redact_pii_processor(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Маскировать ПДн во всех string-полях structlog-события."""
    for key, value in list(event_dict.items()):
        if key in ("timestamp", "level", "event"):
            continue
        event_dict[key] = redact_pii_value(value)
    return event_dict


def get_logger(name: str = "agent") -> structlog.stdlib.BoundLogger:
    if not settings.observability_enabled:
        return structlog.get_logger(name)
    configure_logging()
    return structlog.get_logger(name)
