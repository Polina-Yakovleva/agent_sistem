"""Интеграция Langfuse для трассировки LangChain / LangGraph."""

from __future__ import annotations

import os
from typing import Any, Optional

from app.config import settings
from app.observability.logging import get_logger

_log = get_logger("langfuse")

_handler: Any = None
_client: Any = None
_initialized = False
_init_error: str | None = None


def _configure_env() -> None:
    if settings.langfuse_public_key:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    if settings.langfuse_secret_key:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    if settings.langfuse_host:
        os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host.rstrip("/"))
        os.environ.setdefault("LANGFUSE_BASE_URL", settings.langfuse_host.rstrip("/"))
    if settings.langfuse_debug:
        os.environ.setdefault("LANGFUSE_DEBUG", "true")


def init_langfuse() -> None:
    """Инициализировать клиент Langfuse (singleton)."""
    global _initialized, _client, _init_error
    if _initialized or not settings.langfuse_enabled:
        return
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        _init_error = "missing LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY"
        _log.warning("langfuse_init_skipped", reason=_init_error)
        return

    _configure_env()
    try:
        from langfuse import Langfuse, get_client

        mask = None
        if not settings.langfuse_capture_content:

            def _mask(data: Any, **kwargs: Any) -> Any:
                if isinstance(data, dict):
                    return {k: "[masked]" for k in data}
                if isinstance(data, str):
                    return "[masked]"
                return "[masked]"

            mask = _mask

        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            mask=mask,
        )
        _client = get_client()
        _initialized = True
        _init_error = None
        try:
            if not _client.auth_check():
                _log.warning("langfuse_auth_check_failed", host=settings.langfuse_host)
        except Exception as exc:
            _log.warning("langfuse_auth_check_error", error=str(exc))
    except Exception as exc:
        _initialized = False
        _client = None
        _init_error = str(exc)
        _log.warning("langfuse_init_failed", error=_init_error, host=settings.langfuse_host)


def get_langfuse_handler(
    *,
    session_id: str,
    user_id: str,
    trace_id: str,
) -> Optional[Any]:
    """CallbackHandler для передачи в config[\"callbacks\"]."""
    if not settings.langfuse_enabled:
        return None
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None

    init_langfuse()
    if not _initialized:
        if _init_error:
            _log.warning("langfuse_handler_unavailable", error=_init_error)
        return None
    try:
        from langfuse.langchain import CallbackHandler

        global _handler
        if _handler is None:
            _handler = CallbackHandler()
        return _handler
    except Exception as exc:
        _log.warning("langfuse_handler_failed", error=str(exc))
        return None


def langfuse_metadata(session_id: str, user_id: str, trace_id: str) -> dict[str, Any]:
    """Метаданные для config LangGraph (группировка в Langfuse UI)."""
    return {
        "langfuse_session_id": session_id,
        "langfuse_user_id": user_id,
        "trace_id": trace_id,
    }


def flush_langfuse() -> None:
    if not settings.langfuse_enabled:
        return
    try:
        from langfuse import get_client

        client = get_client()
        if hasattr(client, "flush"):
            client.flush()
        if hasattr(client, "shutdown"):
            client.shutdown()
    except Exception as exc:
        _log.warning("langfuse_flush_failed", error=str(exc))


def langfuse_status() -> dict[str, Any]:
    """Краткий статус для логов при старте."""
    return {
        "enabled": settings.langfuse_enabled,
        "initialized": _initialized,
        "handler_ready": _handler is not None,
        "host": settings.langfuse_host,
        "error": _init_error,
    }
