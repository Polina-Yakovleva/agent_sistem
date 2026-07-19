"""Обращение к внешним API для External Agent.

Источники (без API-ключей):
- Open-Meteo — прогноз погоды и геокодинг города;
- Photon (komoot, данные OpenStreetMap) — геокодинг произвольной локации
  и POI-поиск отелей (osm_tag=tourism:hotel) рядом с точкой.

Функции возвращают готовую к подстановке в контекст LLM строку и не
пробрасывают наружу сетевые исключения (вместо этого — понятное сообщение).
По аналогии с app/rag.py этот модуль — «конвейер», а инструменты-обёртки
живут в app/tools/external.py.
"""

import math
from typing import Any, Optional

import httpx

from app.config import settings
from app.observability.metrics import observe_external

# Коды погоды WMO → краткое описание на русском (Open-Meteo weather_code).
WMO_CODES: dict[int, str] = {
    0: "ясно",
    1: "преимущественно ясно",
    2: "переменная облачность",
    3: "пасмурно",
    45: "туман",
    48: "изморозь",
    51: "слабая морось",
    53: "морось",
    55: "сильная морось",
    56: "ледяная морось",
    57: "сильная ледяная морось",
    61: "небольшой дождь",
    63: "дождь",
    65: "сильный дождь",
    66: "ледяной дождь",
    67: "сильный ледяной дождь",
    71: "небольшой снег",
    73: "снег",
    75: "сильный снег",
    77: "снежная крупа",
    80: "слабый ливень",
    81: "ливень",
    82: "сильный ливень",
    85: "слабый снегопад",
    86: "сильный снегопад",
    95: "гроза",
    96: "гроза с небольшим градом",
    99: "гроза с градом",
}


def _describe_code(code: Any) -> str:
    try:
        return WMO_CODES.get(int(code), "нет данных")
    except (TypeError, ValueError):
        return "нет данных"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние между двумя точками по большому кругу (км)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- #
# Геокодинг
# --------------------------------------------------------------------------- #
def _geocode_open_meteo(client: httpx.Client, city: str) -> Optional[dict]:
    """Геокодинг города через Open-Meteo. Возвращает dict с lat/lon/именем."""
    resp = client.get(
        settings.open_meteo_geocoding_url,
        params={"name": city, "count": 1, "language": "ru", "format": "json"},
    )
    resp.raise_for_status()
    results = (resp.json() or {}).get("results") or []
    if not results:
        return None
    r = results[0]
    name_parts = [r.get("name"), r.get("admin1"), r.get("country")]
    return {
        "lat": r["latitude"],
        "lon": r["longitude"],
        "label": ", ".join(p for p in name_parts if p),
        "timezone": r.get("timezone"),
    }


def _geocode_photon(client: httpx.Client, location: str) -> Optional[dict]:
    """Геокодинг произвольной локации через Photon (данные OSM)."""
    resp = client.get(
        settings.photon_url,
        params={"q": location, "limit": 1},
        headers={"User-Agent": settings.external_user_agent},
    )
    resp.raise_for_status()
    features = (resp.json() or {}).get("features") or []
    if not features:
        return None
    f = features[0]
    lon, lat = (f.get("geometry") or {}).get("coordinates", [None, None])
    if lat is None or lon is None:
        return None
    p = f.get("properties") or {}
    label_parts = [p.get("name"), p.get("city"), p.get("country")]
    return {
        "lat": float(lat),
        "lon": float(lon),
        "label": ", ".join(x for x in label_parts if x) or location,
    }


def _geocode_location(client: httpx.Client, location: str) -> Optional[dict]:
    """Геокодинг локации: сперва Open-Meteo (город), затем Photon (произвольный адрес)."""
    place = _geocode_open_meteo(client, location)
    if place:
        return place
    return _geocode_photon(client, location)


# --------------------------------------------------------------------------- #
# Погода
# --------------------------------------------------------------------------- #
def fetch_weather(city: str, days: Optional[int] = None) -> str:
    """Получить текущую погоду и краткий прогноз для города.

    Возвращает строку-контекст для LLM или понятное сообщение об ошибке.
    """
    days = max(1, min(days or settings.weather_forecast_days, 16))
    timeout = settings.external_timeout

    try:
        with httpx.Client(timeout=timeout) as client:
            with observe_external("open_meteo_geocode"):
                place = _geocode_open_meteo(client, city)
            if not place:
                return f"Не удалось определить координаты города «{city}». Уточните название."

            with observe_external("open_meteo_forecast"):
                resp = client.get(
                    settings.open_meteo_forecast_url,
                    params={
                        "latitude": place["lat"],
                        "longitude": place["lon"],
                        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,"
                        "wind_speed_10m,weather_code",
                        "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
                        "precipitation_probability_max",
                        "timezone": "auto",
                        "forecast_days": days,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
    except httpx.HTTPError as exc:
        return f"Сервис погоды временно недоступен ({type(exc).__name__}). Попробуйте позже."

    cur = data.get("current") or {}
    lines = [
        f"Погода: {place['label']}.",
        f"Сейчас: {cur.get('temperature_2m')}°C "
        f"(ощущается как {cur.get('apparent_temperature')}°C), "
        f"{_describe_code(cur.get('weather_code'))}, "
        f"влажность {cur.get('relative_humidity_2m')}%, "
        f"ветер {cur.get('wind_speed_10m')} км/ч.",
    ]

    daily = data.get("daily") or {}
    dates = daily.get("time") or []
    if dates:
        lines.append("Прогноз по дням:")
        for i, date in enumerate(dates):
            t_max = daily.get("temperature_2m_max", [None] * len(dates))[i]
            t_min = daily.get("temperature_2m_min", [None] * len(dates))[i]
            code = daily.get("weather_code", [None] * len(dates))[i]
            precip = daily.get("precipitation_probability_max", [None] * len(dates))[i]
            precip_text = f", вероятность осадков {precip}%" if precip is not None else ""
            lines.append(
                f"  {date}: от {t_min}°C до {t_max}°C, {_describe_code(code)}{precip_text}."
            )

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Отели
# --------------------------------------------------------------------------- #
def fetch_nearby_hotels(
    location: str,
    radius_km: Optional[float] = None,
    limit: Optional[int] = None,
) -> str:
    """Найти отели рядом с заданной локацией (OSM tourism=hotel).

    Возвращает строку-контекст для LLM или понятное сообщение об ошибке.
    """
    radius_km = radius_km or settings.hotel_radius_km
    limit = limit or settings.hotel_limit
    timeout = settings.external_timeout

    try:
        with httpx.Client(timeout=timeout) as client:
            with observe_external("geocode_location"):
                place = _geocode_location(client, location)
            if not place:
                return f"Не удалось определить координаты локации «{location}». Уточните запрос."

            lat, lon = place["lat"], place["lon"]
            # Photon ранжирует POI по близости к (lat, lon); берём с запасом,
            # затем фильтруем по радиусу через haversine. lang не задаём —
            # Photon поддерживает только default/de/en/fr.
            with observe_external("photon_hotels"):
                resp = client.get(
                    settings.photon_url,
                    params={
                        "q": "hotel",
                        "lat": lat,
                        "lon": lon,
                        "limit": max(limit * 4, 20),
                        "osm_tag": "tourism:hotel",
                    },
                    headers={"User-Agent": settings.external_user_agent},
                )
                resp.raise_for_status()
                features = (resp.json() or {}).get("features") or []
    except httpx.HTTPError as exc:
        return f"Сервис поиска отелей временно недоступен ({type(exc).__name__}). Попробуйте позже."

    hotels = []
    for f in features:
        props = f.get("properties") or {}
        name = props.get("name")
        coords = (f.get("geometry") or {}).get("coordinates")
        if not name or not coords:
            continue
        h_lon, h_lat = float(coords[0]), float(coords[1])
        distance = _haversine_km(lat, lon, h_lat, h_lon)
        if distance > radius_km:
            continue
        hotels.append({"name": name, "distance": distance, "props": props})

    if not hotels:
        return (
            f"В радиусе {radius_km:g} км от «{place['label']}» отели не найдены. "
            f"Попробуйте увеличить радиус."
        )

    hotels.sort(key=lambda h: h["distance"])
    hotels = hotels[:limit]

    lines = [f"Отели рядом с «{place['label']}» (радиус {radius_km:g} км, найдено {len(hotels)}):"]
    for i, h in enumerate(hotels, 1):
        props = h["props"]
        addr_parts = [props.get("street"), props.get("housenumber"), props.get("city")]
        address = " ".join(p for p in addr_parts if p)
        addr_text = f", {address}" if address else ""
        lines.append(f"  {i}. {h['name']} — {h['distance']:.1f} км{addr_text}.")

    return "\n".join(lines)
