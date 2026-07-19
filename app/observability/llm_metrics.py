"""Метрики инференса Ollama / vLLM для Prometheus (слой LLM из раздела 4.2)."""

from __future__ import annotations

import threading
import time
from typing import Optional
from urllib.parse import urlparse

import httpx
from prometheus_client import Gauge

from app.config import settings
from app.observability.metrics import record_error

_ollama_loaded = Gauge(
    "llm_ollama_loaded_models",
    "Число моделей, загруженных в Ollama (/api/ps)",
)
_vllm_up = Gauge(
    "llm_vllm_up",
    "Доступность эндпоинта метрик vLLM (1=up)",
)

_collector_thread: threading.Thread | None = None


def _ollama_base_url() -> Optional[str]:
    if settings.ollama_metrics_url:
        return settings.ollama_metrics_url.rstrip("/")
    parsed = urlparse(settings.llm_base_url)
    if "11434" in (parsed.netloc or settings.llm_base_url):
        return f"{parsed.scheme or 'http'}://{parsed.netloc or 'localhost:11434'}"
    return None


def _poll_ollama() -> None:
    base = _ollama_base_url()
    if not base:
        return
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{base}/api/ps")
            resp.raise_for_status()
            data = resp.json()
        models = data.get("models") if isinstance(data, dict) else data
        count = len(models) if isinstance(models, list) else 0
        _ollama_loaded.set(count)
    except Exception:
        record_error("llm")
        _ollama_loaded.set(0)


def _poll_vllm() -> None:
    url = (settings.vllm_metrics_url or "").strip()
    if not url:
        return
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(url)
            resp.raise_for_status()
        _vllm_up.set(1)
    except Exception:
        record_error("llm")
        _vllm_up.set(0)


def _poll_loop() -> None:
    while True:
        if settings.llm_metrics_enabled and settings.metrics_enabled:
            _poll_ollama()
            _poll_vllm()
        time.sleep(settings.llm_metrics_interval_sec)


def start_llm_metrics_collector() -> None:
    """Фоновый опрос Ollama / vLLM."""
    global _collector_thread
    if not settings.llm_metrics_enabled or not settings.observability_enabled:
        return
    if _collector_thread and _collector_thread.is_alive():
        return
    _collector_thread = threading.Thread(target=_poll_loop, daemon=True, name="llm-metrics")
    _collector_thread.start()
