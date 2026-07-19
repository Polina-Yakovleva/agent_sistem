"""Инструменты External Agent.

Две функции поверх внешних API без ключей (см. app/external_api.py):
- search_nearby_hotels — поиск отелей рядом с локацией (OpenStreetMap);
- get_weather          — текущая погода и краткий прогноз (Open-Meteo).

Реализованы как прямые LangChain @tool (в процессе), единообразно с
инструментами Flight/Booking/Compliance. Возвращают строку-контекст для LLM.
"""

from typing import Optional

from langchain_core.tools import tool

from app.external_api import fetch_nearby_hotels, fetch_weather


@tool
def get_weather(city: str, days: Optional[int] = None) -> str:
    """Получить текущую погоду и краткий прогноз для города.

    Args:
        city: Название города (например, "Стамбул").
        days: Горизонт прогноза в днях (1–16). По умолчанию берётся из конфига.

    Returns:
        Текстовая сводка: текущая погода и прогноз по дням.
    """
    return fetch_weather(city=city, days=days)


@tool
def search_nearby_hotels(
    location: str,
    radius_km: Optional[float] = None,
    limit: Optional[int] = None,
) -> str:
    """Найти отели рядом с указанной локацией.

    Args:
        location: Город или адрес (например, "Москва" или "Москва, Тверская").
        radius_km: Радиус поиска в километрах. По умолчанию из конфига.
        limit: Максимальное число отелей в ответе. По умолчанию из конфига.

    Returns:
        Текстовый список ближайших отелей с расстоянием и адресом.
    """
    return fetch_nearby_hotels(location=location, radius_km=radius_km, limit=limit)


EXTERNAL_TOOLS = [search_nearby_hotels, get_weather]
