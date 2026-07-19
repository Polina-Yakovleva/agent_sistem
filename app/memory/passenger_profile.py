"""Сохранённый пассажир по AGENT_USER_ID для повторного бронирования."""

from datetime import date
from typing import Any, Optional

from app.db import fetch_one, transaction

# Служебный ключ в agent_user_memory (не через LLM-extraction).
_RECONFIRM_KEY = "__booking_reconfirm__"


def load_passenger_profile(agent_user_id: str) -> Optional[dict[str, Any]]:
    if not agent_user_id:
        return None
    row = fetch_one(
        """
        SELECT surname, name, patronymic, passport_id,
               passport_issued_by, passport_issue_date
        FROM public.agent_passenger_profile
        WHERE agent_user_id = %(uid)s
        """,
        {"uid": agent_user_id},
    )
    return dict(row) if row else None


def save_passenger_profile(
    agent_user_id: str,
    *,
    surname: str,
    name: str,
    passport_id: str,
    passport_issued_by: str,
    passport_issue_date: date,
    patronymic: Optional[str] = None,
) -> None:
    if not agent_user_id:
        return
    with transaction() as cur:
        cur.execute(
            """
            INSERT INTO public.agent_passenger_profile (
                agent_user_id, surname, name, patronymic,
                passport_id, passport_issued_by, passport_issue_date
            ) VALUES (
                %(uid)s, %(surname)s, %(name)s, %(patronymic)s,
                %(passport)s, %(issued_by)s, %(issue_date)s
            )
            ON CONFLICT (agent_user_id) DO UPDATE SET
                surname = EXCLUDED.surname,
                name = EXCLUDED.name,
                patronymic = EXCLUDED.patronymic,
                passport_id = EXCLUDED.passport_id,
                passport_issued_by = EXCLUDED.passport_issued_by,
                passport_issue_date = EXCLUDED.passport_issue_date,
                updated_at = now()
            """,
            {
                "uid": agent_user_id,
                "surname": surname.strip(),
                "name": name.strip(),
                "patronymic": (patronymic or "").strip() or None,
                "passport": passport_id,
                "issued_by": passport_issued_by.strip(),
                "issue_date": passport_issue_date,
            },
        )


def mark_booking_reconfirm_required(agent_user_id: str) -> None:
    """После отмены HITL (нет) — следующее бронирование только с явными данными."""
    if not agent_user_id:
        return
    with transaction() as cur:
        cur.execute(
            """
            INSERT INTO public.agent_user_memory (user_id, memory_key, memory_value)
            VALUES (%(uid)s, %(key)s, '1')
            ON CONFLICT (user_id, memory_key)
            DO UPDATE SET memory_value = '1', updated_at = now()
            """,
            {"uid": agent_user_id, "key": _RECONFIRM_KEY},
        )


def clear_booking_reconfirm_required(agent_user_id: str) -> None:
    if not agent_user_id:
        return
    with transaction() as cur:
        cur.execute(
            """
            DELETE FROM public.agent_user_memory
            WHERE user_id = %(uid)s AND memory_key = %(key)s
            """,
            {"uid": agent_user_id, "key": _RECONFIRM_KEY},
        )


def is_booking_reconfirm_required(agent_user_id: str) -> bool:
    if not agent_user_id:
        return False
    row = fetch_one(
        """
        SELECT 1 FROM public.agent_user_memory
        WHERE user_id = %(uid)s AND memory_key = %(key)s AND memory_value = '1'
        """,
        {"uid": agent_user_id, "key": _RECONFIRM_KEY},
    )
    return row is not None


def format_passenger_profile_for_prompt(profile: dict[str, Any]) -> str:
    fio = " ".join(
        p
        for p in [
            profile.get("surname"),
            profile.get("name"),
            profile.get("patronymic"),
        ]
        if p
    )
    issue = profile.get("passport_issue_date")
    issue_s = issue.isoformat() if hasattr(issue, "isoformat") else str(issue)
    return (
        "Сохранённые данные пассажира для этого пользователя "
        "(используй при бронировании, если в запросе не указаны другие):\n"
        f"- ФИО: {fio}\n"
        f"- Паспорт: {profile.get('passport_id')}\n"
        f"- Кем выдан: {profile.get('passport_issued_by')}\n"
        f"- Дата выдачи: {issue_s}"
    )
