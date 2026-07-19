"""Фильтрация ПДн перед записью в долгосрочную память."""

import re

# Ключи профиля, которые нельзя сохранять в LTM.
_BLOCKED_KEYS = frozenset(
    {
        "passport",
        "passport_id",
        "passport_number",
        "user_surname",
        "user_name",
        "user_patronymic",
        "full_name",
        "fio",
        "birth_date",
        "date_of_birth",
        "passport_issued_by",
        "passport_issue_date",
        "user_id",
        "userid",
        "passenger_id",
        "ticket_id",
    }
)

_PASSPORT_RE = re.compile(r"\b\d{10}\b")
_PASSPORT_SERIES_RE = re.compile(r"\b\d{4}\s?\d{6}\b")
_PASSPORT_INLINE_RE = re.compile(
    r"(паспорт|passport)\s*[:\s№#]*\s*\d{4,10}",
    re.IGNORECASE,
)
_SENSITIVE_SUBSTR = (
    "паспорт",
    "passport",
    "серия",
    "номер паспорта",
)


def is_blocked_profile_key(key: str) -> bool:
    normalized = (key or "").strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in _BLOCKED_KEYS:
        return True
    return any(
        token in normalized
        for token in ("passport", "паспорт", "birth", "user_id", "userid", "ticket_id")
    )


def contains_unmasked_passport(text: str) -> bool:
    """Есть ли в тексте полный паспортный номер (10 цифр или серия+номер)."""
    if not text:
        return False
    return bool(_PASSPORT_RE.search(text) or _PASSPORT_SERIES_RE.search(text))


def redact_pii_text(text: str) -> str:
    """Маскировать паспортные номера и явные ПДn в строках логов."""
    if not text:
        return ""
    out = _PASSPORT_SERIES_RE.sub("[passport]", text)
    out = _PASSPORT_RE.sub("[passport]", out)
    out = _PASSPORT_INLINE_RE.sub(r"\1 [passport]", out)
    return out


def redact_pii_value(value: object) -> object:
    """Рекурсивно маскировать строки в dict/list для structlog и JSONL."""
    if isinstance(value, str):
        return redact_pii_text(value)
    if isinstance(value, dict):
        return {k: redact_pii_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_pii_value(item) for item in value]
    return value


def sanitize_profile_facts(facts: dict[str, str]) -> dict[str, str]:
    """Оставить только разрешённые пары ключ–значение."""
    clean: dict[str, str] = {}
    for key, value in (facts or {}).items():
        if not key or not value:
            continue
        if is_blocked_profile_key(key):
            continue
        if _contains_sensitive_text(str(value)):
            continue
        clean[key.strip()[:128]] = str(value).strip()[:2000]
    return clean


def sanitize_episode_text(text: str) -> str:
    """Убрать явные паспортные номера и пометить обрезку ПДн."""
    if not text:
        return ""
    out = _PASSPORT_SERIES_RE.sub("[удалено]", text)
    out = _PASSPORT_RE.sub("[удалено]", out)
    lower = out.lower()
    if any(s in lower for s in _SENSITIVE_SUBSTR):
        return ""
    return out.strip()[:2000]


def _contains_sensitive_text(value: str) -> bool:
    if contains_unmasked_passport(value):
        return True
    lower = value.lower()
    return any(s in lower for s in _SENSITIVE_SUBSTR)
