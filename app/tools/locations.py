"""Сопоставление городов/стран в SQL-фильтрах рейсов."""

from typing import Any

# Частые формы в запросах пользователя → подстрока для ILIKE по City/Country.
_LOCATION_ALIASES: dict[str, str] = {
    "турцию": "турция",
    "турции": "турция",
    "turkey": "турция",
    "москву": "москва",
    "москвы": "москва",
    "стамбулу": "стамбул",
    "пекину": "пекин",
    "минску": "минск",
    "екатеринбургу": "екатеринбург",
    "санкт-петербургу": "санкт-петербург",
    "спб": "санкт-петербург",
    "питер": "санкт-петербург",
    "moscow": "москва",
    "istanbul": "стамбул",
}


def normalize_location(value: str) -> str:
    key = value.strip().lower().replace("ё", "е")
    return _LOCATION_ALIASES.get(key, value.strip())


def location_pattern(value: str) -> str:
    return f"%{normalize_location(value)}%"


def append_origin_filter(
    conditions: list[str], params: dict[str, Any], origin: str, *, city_col: str, country_col: str
) -> None:
    conditions.append(f"({city_col} ILIKE %(origin)s OR {country_col} ILIKE %(origin)s)")
    params["origin"] = location_pattern(origin)


def append_destination_filter(
    conditions: list[str],
    params: dict[str, Any],
    destination: str,
    *,
    city_col: str,
    country_col: str,
) -> None:
    conditions.append(f"({city_col} ILIKE %(destination)s OR {country_col} ILIKE %(destination)s)")
    params["destination"] = location_pattern(destination)
