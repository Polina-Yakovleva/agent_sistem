"""Programmatic security checks (ПДн, HITL, memory-sourcing) перед LLM-критиком."""

from __future__ import annotations

import re

from app.agents.plan_guards import has_visa_intent
from app.memory.privacy import contains_unmasked_passport

_MEMORY_SOURCING_RE = re.compile(
    r"(предыдущ\w*\s+обращен|из\s+памят|ранее\s+(сообщал|упоминал)|"
    r"в\s+прошлых\s+сесс|по\s+данным\s+памят)",
    re.IGNORECASE,
)
_BOOKING_SUCCESS_RE = re.compile(
    r"\b(забронирован|билет\s+оформлен|бронирование\s+успеш|бронь\s+подтвержден)\b",
    re.IGNORECASE,
)
_CANCEL_SUCCESS_RE = re.compile(
    r"\b(бронирование\s+отменен|бронирование\s+отменён|бронь\s+отменен|бронь\s+отменён)\b",
    re.IGNORECASE,
)
_BOOKING_DENIAL_RE = re.compile(
    r"\b(не\s+удалось|ошибк|не\s+забронирован|отказ|требуется\s+подтвержден)\b",
    re.IGNORECASE,
)


def check_no_full_passport_in_answer(*, draft_answer: str) -> tuple[bool, str]:
    """Не раскрывать полный паспортный номер в ответе пользователю."""
    if contains_unmasked_passport(draft_answer or ""):
        return False, (
            "В ответе обнаружен полный паспортный номер. "
            "Маскируй (последние 4 цифры) или убери номер."
        )
    return True, ""


def check_no_memory_sourcing_compliance(
    *,
    user_query: str,
    draft_answer: str,
) -> tuple[bool, str]:
    """Compliance-факты не должны ссылаться на память / прошлые обращения."""
    if not has_visa_intent(user_query or ""):
        return True, ""
    if _MEMORY_SOURCING_RE.search(draft_answer or ""):
        return False, (
            "Визовый ответ не должен ссылаться на память или предыдущие обращения — "
            "только на результаты compliance_agent и RAG."
        )
    return True, ""


def check_booking_claims_grounded(
    *,
    draft_answer: str,
    tools_called: list[str] | None,
) -> tuple[bool, str]:
    """Утверждения об успешном бронировании/отмене должны следовать из инструментов."""
    answer = draft_answer or ""
    tools = set(tools_called or [])

    if _BOOKING_SUCCESS_RE.search(answer) and not _BOOKING_DENIAL_RE.search(answer):
        if "reserve_ticket" not in tools:
            return False, (
                "Утверждение об успешном бронировании без вызова reserve_ticket. "
                "Не подтверждай бронь, если booking_agent не выполнил операцию."
            )

    if _CANCEL_SUCCESS_RE.search(answer) and not _BOOKING_DENIAL_RE.search(answer):
        if "cancel_reservation" not in tools:
            return False, (
                "Утверждение об отмене брони без вызова cancel_reservation. "
                "Не подтверждай отмену без результата booking_agent."
            )

    return True, ""


def run_security_guards(
    *,
    user_query: str,
    draft_answer: str,
    tools_called: list[str] | None = None,
) -> tuple[bool, str]:
    """Сводная programmatic-проверка безопасности перед LLM-критиком."""
    checks = [
        check_no_full_passport_in_answer(draft_answer=draft_answer),
        check_no_memory_sourcing_compliance(user_query=user_query, draft_answer=draft_answer),
        check_booking_claims_grounded(draft_answer=draft_answer, tools_called=tools_called),
    ]
    for ok, msg in checks:
        if not ok:
            return False, msg
    return True, ""
