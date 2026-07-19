"""Разрешение ссылки на рейс: Flight_ID или код (SU2140)."""

from typing import Any

from app.db import fetch_one


def resolve_flight_id(flight_ref: int | str) -> int | None:
    if isinstance(flight_ref, int):
        return flight_ref
    text = str(flight_ref).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    row = fetch_one(
        'SELECT "Flight_ID" FROM public."Flight" WHERE UPPER("Flight_Number") = UPPER(%(n)s)',
        {"n": text},
    )
    return int(row["Flight_ID"]) if row else None


def fetch_flight_row(flight_ref: int | str) -> dict[str, Any] | None:
    flight_id = resolve_flight_id(flight_ref)
    if flight_id is None:
        return None
    return fetch_one(
        """
        SELECT f."Flight_ID", f."Flight_Number", f."Departure_Date", f."Departure_Time",
               f."Ticket_Price", al."Airline_Name" AS airline,
               dep."City_Name" AS origin, arr."City_Name" AS destination
        FROM public."Flight" f
        JOIN public."Airline" al     ON al."Airline_ID" = f."Airline_ID"
        JOIN public."Airport" dep_ap ON dep_ap."Airport_ID" = f."Departure_Airport_ID"
        JOIN public."City"    dep    ON dep."City_ID" = dep_ap."City_ID"
        JOIN public."Airport" arr_ap ON arr_ap."Airport_ID" = f."Arrival_Airport_ID"
        JOIN public."City"    arr    ON arr."City_ID" = arr_ap."City_ID"
        WHERE f."Flight_ID" = %(fid)s
        """,
        {"fid": flight_id},
    )
