"""Структуры для извлечения фактов в LTM."""

from pydantic import BaseModel, Field


class MemoryExtraction(BaseModel):
    """Результат LLM-извлечения после ответа пользователю."""

    profile_facts: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Несекретные факты о пользователе (предпочтения, User_ID, город вылета). "
            "Только то, что пользователь явно сообщил. Без паспорта и ФИО."
        ),
    )
    episode_summary: str = Field(
        default="",
        description="1–3 предложения: что спрашивали и чем закончился диалоговый ход.",
    )
