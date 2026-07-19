"""Профиль пользователя в PostgreSQL (LTM)."""

from app.db import fetch_all, transaction
from app.memory.privacy import sanitize_profile_facts


def load_profile(user_id: str) -> dict[str, str]:
    """Загрузить все ключи профиля для user_id."""
    if not user_id:
        return {}
    rows = fetch_all(
        """
        SELECT memory_key, memory_value
        FROM public.agent_user_memory
        WHERE user_id = %(user_id)s
        ORDER BY memory_key
        """,
        {"user_id": user_id},
    )
    return {row["memory_key"]: row["memory_value"] for row in rows}


def upsert_profile_facts(user_id: str, facts: dict[str, str]) -> int:
    """Сохранить факты профиля. Возвращает число записанных ключей."""
    clean = sanitize_profile_facts(facts)
    if not user_id or not clean:
        return 0
    with transaction() as cur:
        for key, value in clean.items():
            cur.execute(
                """
                INSERT INTO public.agent_user_memory (user_id, memory_key, memory_value)
                VALUES (%(user_id)s, %(key)s, %(value)s)
                ON CONFLICT (user_id, memory_key)
                DO UPDATE SET
                    memory_value = EXCLUDED.memory_value,
                    updated_at = now()
                """,
                {"user_id": user_id, "key": key, "value": value},
            )
    return len(clean)


def format_profile_for_prompt(profile: dict[str, str]) -> str:
    if not profile:
        return ""
    lines = [f"- {k}: {v}" for k, v in profile.items()]
    return "Известно о пользователе:\n" + "\n".join(lines)
