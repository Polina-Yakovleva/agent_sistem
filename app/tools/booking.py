"""Инструменты Booking Agent.

Три функции с транзакционной логикой:
- add_passenger      — валидация и сохранение данных пассажира;
- reserve_ticket     — резервирование билета с проверкой доступности мест;
- cancel_reservation — отмена брони с освобождением места.

Необратимые операции (reserve_ticket, cancel_reservation) требуют ЯВНОГО
подтверждения пользователя. Подтверждение реализовано через Human-in-the-loop
в LangGraph: вызывается interrupt(), граф приостанавливается, и операция
выполняется только после получения положительного ответа (Command(resume=...)).
"""

import re
from datetime import date, datetime
from typing import Any, Optional

from langchain_core.tools import tool
from langgraph.types import interrupt

from app.config import settings
from app.db import fetch_one, transaction
from app.memory.passenger_profile import (
    clear_booking_reconfirm_required,
    is_booking_reconfirm_required,
    load_passenger_profile,
    mark_booking_reconfirm_required,
    save_passenger_profile,
)
from app.runtime import get_agent_user_id
from app.tools.flight_resolve import fetch_flight_row, resolve_flight_id

# Статусы билета (см. таблицу Status).
STATUS_ISSUED = 0  # Оформлен
STATUS_PROCESSING = 1  # Обработка
STATUS_CANCELLED = 2  # Отменён (создаётся при первой отмене, если отсутствует)

# Значения по умолчанию, если пользователь указал только ФИО и номер паспорта.
_DEFAULT_PASSPORT_ISSUED_BY = "не указано пользователем"
_DEFAULT_PASSPORT_ISSUE_DATE = "2000-01-01"


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #
def _is_confirmed(decision: Any) -> bool:
    """Интерпретировать ответ пользователя из Human-in-the-loop как «да/нет»."""
    if isinstance(decision, bool):
        return decision
    if isinstance(decision, dict):
        for key in ("confirm", "confirmed", "approve", "approved"):
            if key in decision:
                return bool(decision[key])
        decision = decision.get("response") or decision.get("answer") or ""
    if isinstance(decision, str):
        return decision.strip().lower() in {
            "yes",
            "y",
            "true",
            "ok",
            "confirm",
            "approve",
            "да",
            "ага",
            "подтверждаю",
            "подтвердить",
            "согласен",
        }
    return bool(decision)


def _parse_date(value: str, field: str) -> date:
    try:
        return datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        raise ValueError(
            f"Поле «{field}» должно быть датой в формате ГГГГ-ММ-ДД, получено: {value!r}"
        )


def _normalize_passport(passport_id: str) -> str:
    """Проверить и нормализовать номер паспорта (только цифры, 10 знаков для РФ)."""
    digits = re.sub(r"\D", "", passport_id or "")
    if len(digits) != 10:
        raise ValueError(
            f"Номер паспорта должен содержать 10 цифр (серия + номер), получено: {passport_id!r}"
        )
    return digits


def _free_seat_number(cur, flight_id: int) -> Optional[int]:
    """Вернуть наименьший свободный номер места для рейса или None, если мест нет."""
    cur.execute(
        """
        SELECT s."Seat_Number"
        FROM public."Seat" s
        JOIN public."Ticket" t ON t."Ticket_ID" = s."Ticket_ID"
        WHERE t."Flight_ID" = %(flight_id)s
          AND t."Ticket_Status_ID" <> %(cancelled)s
        """,
        {"flight_id": flight_id, "cancelled": STATUS_CANCELLED},
    )
    taken = {row["Seat_Number"] for row in cur.fetchall()}
    for number in range(1, settings.flight_seat_capacity + 1):
        if number not in taken:
            return number
    return None


def _next_id(cur, table: str, pk: str) -> int:
    """Сгенерировать следующий ID (в схеме нет sequence)."""
    cur.execute(f'SELECT COALESCE(MAX("{pk}"), 0) + 1 AS next_id FROM public."{table}"')
    return cur.fetchone()["next_id"]


def _validate_passenger_input(
    surname: str,
    name: str,
    passport_id: str,
    passport_issued_by: str,
    passport_issue_date: str,
    *,
    allow_partial: bool = False,
) -> tuple[str, date] | str:
    if not surname or not surname.strip():
        return "Ошибка валидации: не указана фамилия пассажира."
    if not name or not name.strip():
        return "Ошибка валидации: не указано имя пассажира."
    issued_by = (passport_issued_by or "").strip()
    issue_raw = (passport_issue_date or "").strip()
    if allow_partial:
        if not issued_by:
            issued_by = _DEFAULT_PASSPORT_ISSUED_BY
        if not issue_raw:
            issue_raw = _DEFAULT_PASSPORT_ISSUE_DATE
    if not issued_by:
        return "Ошибка валидации: не указано, кем выдан паспорт."
    try:
        passport = _normalize_passport(passport_id)
        issue_date = _parse_date(issue_raw, "дата выдачи паспорта")
    except ValueError as exc:
        return f"Ошибка валидации: {exc}"
    if issue_date > date.today():
        return "Ошибка валидации: дата выдачи паспорта не может быть в будущем."
    return passport, issue_date


def _resolve_passenger_id(
    cur,
    *,
    surname: str,
    name: str,
    patronymic: Optional[str],
    passport: str,
    passport_issued_by: str,
    issue_date: date,
    create_if_missing: bool,
) -> tuple[int, str] | str:
    """Найти пассажира по паспорту или создать запись. Вернуть (user_id, fio) либо текст ошибки."""
    cur.execute(
        """
        SELECT "User_ID", "User_surname", "User_name", "User_patronymic"
        FROM public."User"
        WHERE passport_id = %(passport)s
        """,
        {"passport": passport},
    )
    existing = cur.fetchone()
    fio = " ".join(p for p in [surname.strip(), name.strip(), (patronymic or "").strip()] if p)

    if existing:
        if existing["User_surname"].strip().lower() != surname.strip().lower():
            return (
                "Паспорт уже зарегистрирован на другую фамилию. "
                "Проверьте данные или уточните у пользователя."
            )
        if existing["User_name"].strip().lower() != name.strip().lower():
            return (
                "Паспорт уже зарегистрирован на другое имя. "
                "Проверьте данные или уточните у пользователя."
            )
        return int(existing["User_ID"]), fio

    if not create_if_missing:
        return "Пассажир с таким паспортом не найден. Сначала сохраните данные через add_passenger."

    user_id = _next_id(cur, "User", "User_ID")
    cur.execute(
        """
        INSERT INTO public."User" (
            "User_ID", "User_surname", "User_name", "User_patronymic",
            "Passport_issued_by", "Passport_issue_date", passport_id
        ) VALUES (
            %(user_id)s, %(surname)s, %(name)s, %(patronymic)s,
            %(issued_by)s, %(issue_date)s, %(passport)s
        )
        """,
        {
            "user_id": user_id,
            "surname": surname.strip(),
            "name": name.strip(),
            "patronymic": (patronymic or "").strip() or None,
            "issued_by": passport_issued_by.strip(),
            "issue_date": issue_date,
            "passport": passport,
        },
    )
    return user_id, fio


def _has_minimal_passenger_identity(
    surname: str,
    name: str,
    passport_id: str,
) -> bool:
    return all((value or "").strip() for value in (surname, name, passport_id))


def _fill_passenger_fields(
    surname: str,
    name: str,
    passport_id: str,
    passport_issued_by: str,
    passport_issue_date: str,
    patronymic: Optional[str],
) -> tuple[str, str, str, str, str, Optional[str]] | str:
    """Подставить сохранённый профиль AGENT_USER_ID для незаполненных полей."""
    agent_uid = get_agent_user_id()
    reconfirm = is_booking_reconfirm_required(agent_uid)

    if reconfirm:
        if not _has_minimal_passenger_identity(surname, name, passport_id):
            return (
                "Предыдущее бронирование было отменено. Укажите заново в запросе: "
                "фамилию, имя и номер паспорта (10 цифр). "
                "Сохранённый профиль до нового подтверждения не используется."
            )
        clear_booking_reconfirm_required(agent_uid)
        return (
            surname.strip(),
            name.strip(),
            passport_id.strip(),
            passport_issued_by.strip(),
            passport_issue_date.strip(),
            (patronymic or "").strip() or None,
        )

    profile = load_passenger_profile(agent_uid)

    def pick(value: str, profile_key: str) -> str:
        text = (value or "").strip()
        if text:
            return text
        if profile and profile.get(profile_key):
            raw = profile[profile_key]
            if profile_key == "passport_issue_date" and hasattr(raw, "isoformat"):
                return raw.isoformat()
            return str(raw).strip()
        return ""

    merged = (
        pick(surname, "surname"),
        pick(name, "name"),
        pick(passport_id, "passport_id"),
        pick(passport_issued_by, "passport_issued_by"),
        pick(passport_issue_date, "passport_issue_date"),
        pick(patronymic or "", "patronymic") or None,
    )
    if not all(merged[:3]):
        return (
            "Не указаны данные пассажира. Сообщите ФИО и паспорт в запросе "
            "или сохраните их через add_passenger (они запомнятся для вашего AGENT_USER_ID)."
        )
    if not merged[3]:
        merged = (
            merged[0],
            merged[1],
            merged[2],
            _DEFAULT_PASSPORT_ISSUED_BY,
            merged[4] or _DEFAULT_PASSPORT_ISSUE_DATE,
            merged[5],
        )
    elif not merged[4]:
        merged = (
            merged[0],
            merged[1],
            merged[2],
            merged[3],
            _DEFAULT_PASSPORT_ISSUE_DATE,
            merged[5],
        )
    return merged


def _remember_passenger_profile(
    surname: str,
    name: str,
    passport: str,
    passport_issued_by: str,
    issue_date: date,
    patronymic: Optional[str],
) -> None:
    save_passenger_profile(
        get_agent_user_id(),
        surname=surname,
        name=name,
        passport_id=passport,
        passport_issued_by=passport_issued_by,
        passport_issue_date=issue_date,
        patronymic=patronymic,
    )


# --------------------------------------------------------------------------- #
# add_passenger
# --------------------------------------------------------------------------- #
@tool
def add_passenger(
    surname: str,
    name: str,
    passport_id: str,
    passport_issued_by: str = "",
    passport_issue_date: str = "",
    patronymic: Optional[str] = None,
) -> str:
    """Проверить и сохранить данные пассажира в БД.

    Выполняет валидацию ФИО и паспортных данных, затем сохраняет запись.
    Если пассажир с таким номером паспорта уже существует, возвращает его ID
    без создания дубликата.

    Args:
        surname: Фамилия пассажира.
        name: Имя пассажира.
        passport_id: Номер паспорта (10 цифр, серия + номер).
        passport_issued_by: Кем выдан паспорт (необязательно).
        passport_issue_date: Дата выдачи (ГГГГ-ММ-ДД, необязательно).
        patronymic: Отчество (необязательно).

    Returns:
        Сообщение с присвоенным идентификатором пассажира (User_ID).
    """
    filled = _fill_passenger_fields(
        surname, name, passport_id, passport_issued_by, passport_issue_date, patronymic
    )
    if isinstance(filled, str):
        return filled
    surname, name, passport_id, passport_issued_by, passport_issue_date, patronymic = filled

    validated = _validate_passenger_input(
        surname,
        name,
        passport_id,
        passport_issued_by,
        passport_issue_date,
        allow_partial=_has_minimal_passenger_identity(surname, name, passport_id),
    )
    if isinstance(validated, str):
        return validated
    passport, issue_date = validated

    with transaction() as cur:
        cur.execute(
            'SELECT "User_ID" FROM public."User" WHERE passport_id = %(passport)s',
            {"passport": passport},
        )
        if cur.fetchone():
            _remember_passenger_profile(
                surname, name, passport, passport_issued_by, issue_date, patronymic
            )
            return (
                f"Пассажир уже зарегистрирован (паспорт {passport}). "
                "Новая запись не создавалась. Данные сохранены для повторного бронирования."
            )
        resolved = _resolve_passenger_id(
            cur,
            surname=surname,
            name=name,
            patronymic=patronymic,
            passport=passport,
            passport_issued_by=passport_issued_by,
            issue_date=issue_date,
            create_if_missing=True,
        )
        if isinstance(resolved, str):
            return resolved
        user_id, fio = resolved

    clear_booking_reconfirm_required(get_agent_user_id())
    _remember_passenger_profile(surname, name, passport, passport_issued_by, issue_date, patronymic)
    return f"Пассажир сохранён: {fio}, паспорт {passport}. Присвоен User_ID={user_id}."


# --------------------------------------------------------------------------- #
# reserve_ticket
# --------------------------------------------------------------------------- #
@tool
def reserve_ticket(
    flight_id: int | str,
    surname: str = "",
    name: str = "",
    passport_id: str = "",
    passport_issued_by: str = "",
    passport_issue_date: str = "",
    patronymic: Optional[str] = None,
    seat_number: Optional[int] = None,
) -> str:
    """Зарезервировать билет на рейс для пассажира (НЕОБРАТИМАЯ операция).

    Перед записью в БД запрашивает явное подтверждение пользователя через
    Human-in-the-loop (interrupt). Пустые поля пассажира подставляются из
    сохранённого профиля текущего AGENT_USER_ID (если пользователь уже вводил данные).

    Args:
        flight_id: Код рейса (SU2140, DP200) или внутренний Flight_ID (12).
        surname: Фамилия (необязательно, если есть сохранённый профиль).
        name: Имя.
        passport_id: Номер паспорта (10 цифр).
        passport_issued_by: Кем выдан паспорт.
        passport_issue_date: Дата выдачи (ГГГГ-ММ-ДД).
        patronymic: Отчество (необязательно).
        seat_number: Желаемый номер места (необязательно).

    Returns:
        Подтверждение брони с номером билета и места либо причину отказа.
    """
    filled = _fill_passenger_fields(
        surname, name, passport_id, passport_issued_by, passport_issue_date, patronymic
    )
    if isinstance(filled, str):
        return filled
    surname, name, passport_id, passport_issued_by, passport_issue_date, patronymic = filled

    validated = _validate_passenger_input(
        surname,
        name,
        passport_id,
        passport_issued_by,
        passport_issue_date,
        allow_partial=_has_minimal_passenger_identity(surname, name, passport_id),
    )
    if isinstance(validated, str):
        return validated
    passport, issue_date = validated

    resolved_flight_id = resolve_flight_id(flight_id)
    if resolved_flight_id is None:
        return f"Рейс {flight_id} не найден. Укажите код рейса (например, SU2140)."

    flight = fetch_flight_row(flight_id)
    if not flight:
        return f"Рейс {flight_id} не найден."
    flight_id = resolved_flight_id

    with transaction() as cur:
        resolved_passenger = _resolve_passenger_id(
            cur,
            surname=surname,
            name=name,
            patronymic=patronymic,
            passport=passport,
            passport_issued_by=passport_issued_by,
            issue_date=issue_date,
            create_if_missing=True,
        )
    if isinstance(resolved_passenger, str):
        return resolved_passenger
    user_id, fio = resolved_passenger

    # Предварительный расчёт места (для текста подтверждения), окончательная
    # проверка повторяется внутри транзакции, чтобы избежать гонок.
    with transaction() as cur:
        proposed_seat = (
            seat_number if seat_number is not None else _free_seat_number(cur, flight_id)
        )
    if proposed_seat is None:
        return f"На рейс №{flight_id} нет свободных мест. Бронирование невозможно."

    # --- Human-in-the-loop: явное подтверждение перед необратимой операцией ---
    decision = interrupt(
        {
            "action": "reserve_ticket",
            "requires_confirmation": True,
            "summary": (
                f"Подтвердите бронирование: пассажир {fio} (User_ID={user_id}), "
                f"рейс {flight['Flight_Number']} (ID {flight_id}) "
                f"{flight['origin']} → {flight['destination']} "
                f"({flight['airline']}), вылет {flight['Departure_Date']} "
                f"{str(flight['Departure_Time'])[:5]}, место {proposed_seat}, "
                f"стоимость {flight['Ticket_Price']} ₽."
            ),
            "details": {
                "user_id": user_id,
                "flight_id": flight_id,
                "seat_number": proposed_seat,
                "price": flight["Ticket_Price"],
            },
        }
    )
    if not _is_confirmed(decision):
        mark_booking_reconfirm_required(get_agent_user_id())
        return (
            "Бронирование отменено: пользователь не подтвердил операцию. "
            "При следующей попытке заново запросите ФИО и паспорт — "
            "сохранённый профиль не подставится автоматически."
        )

    # --- Транзакционная запись ---
    with transaction() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%(fid)s)", {"fid": flight_id})

        # Повторная проверка занятости места уже под блокировкой.
        cur.execute(
            """
            SELECT 1
            FROM public."Seat" s
            JOIN public."Ticket" t ON t."Ticket_ID" = s."Ticket_ID"
            WHERE t."Flight_ID" = %(fid)s
              AND t."Ticket_Status_ID" <> %(cancelled)s
              AND s."Seat_Number" = %(seat)s
            """,
            {"fid": flight_id, "cancelled": STATUS_CANCELLED, "seat": proposed_seat},
        )
        if cur.fetchone():
            # Желаемое место заняли — берём следующее свободное.
            proposed_seat = _free_seat_number(cur, flight_id)
            if proposed_seat is None:
                return f"На рейс №{flight_id} закончились свободные места. Бронирование отменено."

        ticket_id = _next_id(cur, "Ticket", "Ticket_ID")
        cur.execute(
            """
            INSERT INTO public."Ticket" (
                "Ticket_ID", "User_ID", "Flight_ID", "Ticket_Status_ID", "Purchase_Date"
            ) VALUES (%(tid)s, %(uid)s, %(fid)s, %(status)s, %(purchase)s)
            """,
            {
                "tid": ticket_id,
                "uid": user_id,
                "fid": flight_id,
                "status": STATUS_ISSUED,
                "purchase": date.today(),
            },
        )

        seat_id = _next_id(cur, "Seat", "Seat_ID")
        cur.execute(
            """
            INSERT INTO public."Seat" ("Seat_ID", "Ticket_ID", "Seat_Number")
            VALUES (%(sid)s, %(tid)s, %(seat)s)
            """,
            {"sid": seat_id, "tid": ticket_id, "seat": proposed_seat},
        )

    clear_booking_reconfirm_required(get_agent_user_id())
    _remember_passenger_profile(surname, name, passport, passport_issued_by, issue_date, patronymic)

    return (
        f"Бронирование подтверждено. Билет №{ticket_id} для пассажира {fio} "
        f"на рейс {flight['Flight_Number']} ({flight['origin']} → {flight['destination']}), "
        f"место {proposed_seat}, статус «Оформлен». Стоимость {flight['Ticket_Price']} ₽."
    )


# --------------------------------------------------------------------------- #
# cancel_reservation
# --------------------------------------------------------------------------- #
@tool
def cancel_reservation(ticket_id: int) -> str:
    """Отменить бронь и освободить место (НЕОБРАТИМАЯ операция).

    Перед изменением БД запрашивает явное подтверждение пользователя через
    Human-in-the-loop (interrupt). Помечает билет статусом «Отменён» и удаляет
    закреплённое место, возвращая его в продажу.

    Args:
        ticket_id: Номер билета (Ticket_ID).

    Returns:
        Подтверждение отмены либо причину, по которой отмена невозможна.
    """
    ticket = fetch_one(
        """
        SELECT t."Ticket_ID", t."Flight_ID", t."Ticket_Status_ID",
               u."User_surname", u."User_name", u."User_patronymic",
               s."Seat_Number" AS seat_number
        FROM public."Ticket" t
        JOIN public."User" u ON u."User_ID" = t."User_ID"
        LEFT JOIN public."Seat" s ON s."Ticket_ID" = t."Ticket_ID"
        WHERE t."Ticket_ID" = %(tid)s
        """,
        {"tid": ticket_id},
    )
    if not ticket:
        return f"Билет №{ticket_id} не найден."
    if ticket["Ticket_Status_ID"] == STATUS_CANCELLED:
        return f"Билет №{ticket_id} уже отменён."

    fio = " ".join(
        p for p in [ticket["User_surname"], ticket["User_name"], ticket["User_patronymic"]] if p
    )
    seat_text = (
        f"место {ticket['seat_number']}"
        if ticket["seat_number"] is not None
        else "место не закреплено"
    )

    # --- Human-in-the-loop: явное подтверждение перед необратимой операцией ---
    decision = interrupt(
        {
            "action": "cancel_reservation",
            "requires_confirmation": True,
            "summary": (
                f"Подтвердите отмену брони: билет №{ticket_id}, пассажир {fio}, "
                f"рейс №{ticket['Flight_ID']}, {seat_text}. "
                f"Место будет освобождено и возвращено в продажу."
            ),
            "details": {
                "ticket_id": ticket_id,
                "flight_id": ticket["Flight_ID"],
                "seat_number": ticket["seat_number"],
            },
        }
    )
    if not _is_confirmed(decision):
        return "Отмена не выполнена: пользователь не подтвердил операцию."

    # --- Транзакционная запись ---
    with transaction() as cur:
        # Гарантируем наличие статуса «Отменён» (в исходной схеме его нет).
        cur.execute(
            """
            INSERT INTO public."Status" ("Ticket_Status_ID", "Status_Name")
            VALUES (%(sid)s, %(name)s)
            ON CONFLICT ("Ticket_Status_ID") DO NOTHING
            """,
            {"sid": STATUS_CANCELLED, "name": "Отменён"},
        )
        # Освобождаем место.
        cur.execute(
            'DELETE FROM public."Seat" WHERE "Ticket_ID" = %(tid)s',
            {"tid": ticket_id},
        )
        # Помечаем билет отменённым.
        cur.execute(
            'UPDATE public."Ticket" SET "Ticket_Status_ID" = %(cancelled)s WHERE "Ticket_ID" = %(tid)s',
            {"cancelled": STATUS_CANCELLED, "tid": ticket_id},
        )

    return (
        f"Бронь отменена. Билет №{ticket_id} (пассажир {fio}, рейс №{ticket['Flight_ID']}) "
        f"переведён в статус «Отменён», {seat_text} освобождено."
    )


BOOKING_TOOLS = [add_passenger, reserve_ticket, cancel_reservation]
