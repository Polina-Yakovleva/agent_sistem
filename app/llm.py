"""Фабрика LLM.

Модель A-vibe обслуживается vLLM по OpenAI-совместимому API, поэтому
используется ChatOpenAI с произвольным base_url. Параметры берутся из конфига
(переопределяются через .env). Любой другой OpenAI-совместимый эндпоинт также
подойдёт — достаточно сменить LLM_BASE_URL / LLM_MODEL / LLM_API_KEY.
"""

from functools import lru_cache
from typing import Optional
from urllib.parse import urlparse

import httpx
from langchain_openai import ChatOpenAI

from app.config import settings


def _llm_request_headers() -> dict[str, str] | None:
    host = urlparse(settings.llm_base_url).netloc.lower()
    if "ngrok" in host:
        return {"ngrok-skip-browser-warning": "true"}
    return None


def check_llm_reachable() -> None:
    """Проверить, что LLM_BASE_URL отвечает (до долгого graph.invoke)."""
    base = settings.llm_base_url.rstrip("/")
    url = f"{base}/models"
    headers = _llm_request_headers()
    try:
        with httpx.Client(timeout=min(settings.llm_timeout, 15)) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
    except Exception as exc:
        raise ConnectionError(
            f"LLM недоступен: {settings.llm_base_url}\n"
            f"Причина: {exc}\n"
            "Обновите LLM_BASE_URL в .env — для Colab возьмите новый URL из cloudflared/ngrok "
            "или укажите локальный Ollama: http://localhost:11434/v1"
        ) from exc


@lru_cache(maxsize=8)
def get_llm(temperature: Optional[float] = None) -> ChatOpenAI:
    """Вернуть настроенный экземпляр LLM (кэшируется по значению temperature).

    В системе используется лишь несколько разных значений temperature
    (planner/critic/summarize — 0; аггрегатор/субагенты — settings.llm_temperature),
    поэтому maxsize=8 держит все варианты закэшированными без вытеснения.
    """
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        default_headers=_llm_request_headers(),
    )
