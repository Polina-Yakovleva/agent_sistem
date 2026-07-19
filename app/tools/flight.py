"""Инструменты Flight Agent.

Три функции поверх PostgreSQL:
- get_flights        — список доступных рейсов (фильтр по направлению/дате);
- get_flight_details — детальная информация по номеру рейса;
- search_flights     — гибкий поиск: диапазон дат, цен, времени вылета.

Все функции возвращают человекочитаемую строку, готовую для подстановки
в контекст LLM.
"""

from typing import Any, Optional

from langchain_core.tools import tool

from app.config import settings
from app.db import fetch_all, fetch_one
from app.tools.flight_resolve import resolve_flight_id
from app.tools.locations import append_destination_filter, append_origin_filter

# Идентификатор статуса отменённого билета (см. booking.py).
CANCELLED_STATUS_ID = 2

# Базовый SELECT: рейс + авиакомпания + города вылета/прибытия + число занятых
# мест (по активным билетам) и число свободных мест относительно вместимости.
_BASE_SELECT = """
SELECT
    f."Flight_ID"                         AS flight_id,
    f."Flight_Number"                     AS flight_number,
    al."Airline_Name"                     AS airline,
    dep_city."City_Name"                  AS origin_city,
    dep_ap."Airport_Name"                 AS origin_airport,
    arr_city."City_Name"                  AS destination_city,
    arr_ap."Airport_Name"                 AS destination_airport,
    f."Departure_Date"                    AS departure_date,
    f."Departure_Time"                    AS departure_time,
    f."Arrival_Date"                      AS arrival_date,
    f."Arrival_Time"                       AS arrival_time,
    f."Ticket_Price"                      AS price,
    COALESCE(occ.occupied, 0)             AS occupied_seats
FROM public."Flight" f
JOIN public."Airline" al        ON al."Airline_ID" = f."Airline_ID"
JOIN public."Airport" dep_ap    ON dep_ap."Airport_ID" = f."Departure_Airport_ID"
JOIN public."City"    dep_city  ON dep_city."City_ID" = dep_ap."City_ID"
JOIN public."Airport" arr_ap    ON arr_ap."Airport_ID" = f."Arrival_Airport_ID"
JOIN public."City"    arr_city  ON arr_city."City_ID" = arr_ap."City_ID"
JOIN public."Country" dep_country ON dep_country."Country_ID" = dep_city."Country_ID"
JOIN public."Country" arr_country ON arr_country."Country_ID" = arr_city."Country_ID"
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS occupied
    FROM public."Seat" s
    JOIN public."Ticket" t ON t."Ticket_ID" = s."Ticket_ID"
    WHERE t."Flight_ID" = f."Flight_ID"
      AND t."Ticket_Status_ID" <> %(cancelled)s
) occ ON TRUE
"""


def _format_flight(row: dict[str, Any], detailed: bool = False) -> str:
    """Сформировать читаемое описание рейса."""
    capacity = settings.flight_seat_capacity
    free_seats = max(capacity - int(row["occupied_seats"]), 0)

    head = (
        f"Рейс {row['flight_number']} (ID {row['flight_id']}, {row['airline']}): "
        f"{row['origin_city']} ({row['origin_airport']}) → "
        f"{row['destination_city']} ({row['destination_airport']})"
    )
    schedule = (
        f"вылет {row['departure_date']} в {str(row['departure_time'])[:5]}, "
        f"прибытие {row['arrival_date']} в {str(row['arrival_time'])[:5]}"
    )
    money = f"цена билета {row['price']} ₽, свободно мест {free_seats} из {capacity}"

    if not detailed:
        return f"{head}; {schedule}; {money}."

    return (
        f"{head}\n"
        f"  Маршрут: {row['origin_city']} ({row['origin_airport']}) → "
        f"{row['destination_city']} ({row['destination_airport']})\n"
        f"  Вылет: {row['departure_date']} {str(row['departure_time'])[:5]}\n"
        f"  Прибытие: {row['arrival_date']} {str(row['arrival_time'])[:5]}\n"
        f"  Номер рейса: {row['flight_number']}\n"
        f"  Авиакомпания: {row['airline']}\n"
        f"  Цена билета: {row['price']} ₽\n"
        f"  Свободно мест: {free_seats} из {capacity}"
    )


@tool
def get_flights(
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    date: Optional[str] = None,
) -> str:
    """Вернуть список доступных рейсов с возможностью фильтра по направлению и дате.

    Args:
        origin: Город вылета (например, "Москва"). Необязательно.
        destination: Город или страна прибытия (например, "Стамбул", "Турция"). Необязательно.
        date: Дата вылета в формате ГГГГ-ММ-ДД. Необязательно.

    Returns:
        Текстовый список подходящих рейсов или сообщение, что рейсы не найдены.
    """
    conditions: list[str] = []
    params: dict[str, Any] = {"cancelled": CANCELLED_STATUS_ID}

    if origin:
        append_origin_filter(
            conditions,
            params,
            origin,
            city_col='dep_city."City_Name"',
            country_col='dep_country."Country_Name"',
        )
    if destination:
        append_destination_filter(
            conditions,
            params,
            destination,
            city_col='arr_city."City_Name"',
            country_col='arr_country."Country_Name"',
        )
    if date:
        conditions.append('f."Departure_Date" = %(date)s')
        params["date"] = date.strip()

    query = _BASE_SELECT
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += ' ORDER BY f."Departure_Date", f."Departure_Time" LIMIT 50'

    rows = fetch_all(query, params)
    if not rows:
        return "По заданным условиям рейсов не найдено."

    lines = [f"Найдено рейсов: {len(rows)}."]
    lines += [f"{i}. {_format_flight(r)}" for i, r in enumerate(rows, 1)]
    return "\n".join(lines)


@tool
def get_flight_details(flight_id: int | str) -> str:
    """Вернуть детальную информацию по конкретному рейсу.

    Args:
        flight_id: Код рейса (например, SU2140) или внутренний Flight_ID (число).

    Returns:
        Подробное описание рейса или сообщение, что рейс не найден.
    """
    resolved = resolve_flight_id(flight_id)
    if resolved is None:
        return f"Рейс {flight_id} не найден."
    query = _BASE_SELECT + ' WHERE f."Flight_ID" = %(flight_id)s'
    row = fetch_one(query, {"flight_id": resolved, "cancelled": CANCELLED_STATUS_ID})
    if not row:
        return f"Рейс {flight_id} не найден."
    return _format_flight(row, detailed=True)


@tool
def search_flights(
    origin: Optional[str] = None,
    destination: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    price_min: Optional[int] = None,
    price_max: Optional[int] = None,
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
) -> str:
    """Гибкий поиск рейсов по диапазону дат, цен и времени вылета.

    Все параметры необязательны и комбинируются через логическое И.

    Args:
        origin: Город вылета.
        destination: Город или страна прибытия (например, "Турция").
        date_from: Нижняя граница даты вылета (ГГГГ-ММ-ДД).
        date_to: Верхняя граница даты вылета (ГГГГ-ММ-ДД).
        price_min: Минимальная цена билета (₽).
        price_max: Максимальная цена билета (₽).
        time_from: Самое раннее время вылета (ЧЧ:ММ).
        time_to: Самое позднее время вылета (ЧЧ:ММ).

    Returns:
        Текстовый список подходящих рейсов или сообщение об отсутствии результатов.
    """
    conditions: list[str] = []
    params: dict[str, Any] = {"cancelled": CANCELLED_STATUS_ID}

    if origin:
        append_origin_filter(
            conditions,
            params,
            origin,
            city_col='dep_city."City_Name"',
            country_col='dep_country."Country_Name"',
        )
    if destination:
        append_destination_filter(
            conditions,
            params,
            destination,
            city_col='arr_city."City_Name"',
            country_col='arr_country."Country_Name"',
        )
    if date_from:
        conditions.append('f."Departure_Date" >= %(date_from)s')
        params["date_from"] = date_from.strip()
    if date_to:
        conditions.append('f."Departure_Date" <= %(date_to)s')
        params["date_to"] = date_to.strip()
    if price_min is not None:
        conditions.append('f."Ticket_Price" >= %(price_min)s')
        params["price_min"] = price_min
    if price_max is not None:
        conditions.append('f."Ticket_Price" <= %(price_max)s')
        params["price_max"] = price_max
    if time_from:
        conditions.append('f."Departure_Time" >= %(time_from)s')
        params["time_from"] = time_from.strip()
    if time_to:
        conditions.append('f."Departure_Time" <= %(time_to)s')
        params["time_to"] = time_to.strip()

    query = _BASE_SELECT
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += ' ORDER BY f."Ticket_Price", f."Departure_Date", f."Departure_Time" LIMIT 50'

    rows = fetch_all(query, params)
    if not rows:
        return "По заданным критериям поиска рейсов не найдено."

    lines = [f"Найдено рейсов: {len(rows)}."]
    lines += [f"{i}. {_format_flight(r)}" for i, r in enumerate(rows, 1)]
    return "\n".join(lines)


FLIGHT_TOOLS = [get_flights, get_flight_details, search_flights]
