"""Извлечение ФИО и паспорта из текста запроса пользователя."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

_PASSPORT_RE = re.compile(r"\b(\d{10})\b")
_FLIGHT_CODE_RE = re.compile(r"\b([A-Z]{2}\d{3,4})\b", re.IGNORECASE)

# «для Иванова Ивана», «Иванов Иван», «фамилия Тестов»
_FOR_NAME_RE = re.compile(
    r"(?:для|пассажир[а]?|фамилия)\s+"
    r"([А-ЯA-ZЁ][а-яa-zё]+)\s+([А-ЯA-ZЁ][а-яa-zё]+)"
    r"(?:\s+([А-ЯA-ZЁ][а-яa-zё]+))?",
    re.IGNORECASE,
)
_NAME_PAIR_RE = re.compile(r"\b([А-ЯA-ZЁ][а-яa-zё]{1,30})\s+([А-ЯA-ZЁ][a-zа-яё]{1,30})\b")


@dataclass(frozen=True)
class ParsedPassenger:
    surname: str
    name: str
    patronymic: Optional[str] = None
    passport_id: Optional[str] = None
    flight_code: Optional[str] = None


def _normalize_declension(surname: str, name: str) -> tuple[str, str]:
    """Привести «Иванова Ивана» к именительному падежу для booking."""
    s = surname.strip()
    n = name.strip()
    if s.endswith("ова") and len(s) > 3:
        s = s[:-1]  # Иванова -> Иванов
    if n.endswith("а") and len(n) > 3 and s.endswith(("ов", "ев", "ин", "ын")):
        n = n[:-1]
    return s, n


def parse_passenger_from_text(text: str) -> ParsedPassenger | None:
    """Попытаться извлечь ФИО и паспорт из user_query."""
    if not text or not text.strip():
        return None

    passport_match = _PASSPORT_RE.search(text)
    passport = passport_match.group(1) if passport_match else None

    flight_match = _FLIGHT_CODE_RE.search(text)
    flight_code = flight_match.group(1).upper() if flight_match else None

    for pattern in (_FOR_NAME_RE, _NAME_PAIR_RE):
        match = pattern.search(text)
        if match:
            surname, name = match.group(1), match.group(2)
            patronymic = match.group(3) if match.lastindex and match.lastindex >= 3 else None
            if surname.lower() in {"паспорт", "рейс", "билет", "номер"}:
                continue
            surname, name = _normalize_declension(surname, name)
            if passport or (surname and name):
                return ParsedPassenger(
                    surname=surname.strip(),
                    name=name.strip(),
                    patronymic=patronymic.strip() if patronymic else None,
                    passport_id=passport,
                    flight_code=flight_code,
                )

    if passport:
        return ParsedPassenger(surname="", name="", passport_id=passport, flight_code=flight_code)
    return None


def format_passenger_hint(parsed: ParsedPassenger) -> str:
    """Текстовая подсказка для booking-agent."""
    parts = ["Из запроса пользователя извлечено:"]
    if parsed.surname and parsed.name:
        fio = f"{parsed.surname} {parsed.name}"
        if parsed.patronymic:
            fio += f" {parsed.patronymic}"
        parts.append(f"- ФИО: {fio}")
    if parsed.passport_id:
        parts.append(f"- Паспорт: {parsed.passport_id}")
    if parsed.flight_code:
        parts.append(f"- Рейс: {parsed.flight_code}")
    parts.append(
        "Действия: сначала add_passenger (орган выдачи и дата выдачи можно опустить), "
        "затем reserve_ticket с кодом рейса."
    )
    return "\n".join(parts)
