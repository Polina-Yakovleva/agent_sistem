"""Сжатие длинной истории сессии (STM)."""

from langchain_core.messages import BaseMessage, HumanMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from app.config import settings
from app.llm import get_llm

SUMMARIZE_PROMPT = """Ты сжимаешь историю диалога пользователя с ассистентом авиакомпании.
Сохрани: направления, даты, номера рейсов, User_ID, решения пользователя, открытые вопросы.
Не включай паспортные данные и полные ФИО.
Ответ — краткое резюме на русском (до 15 предложений)."""


def should_summarize(messages: list[BaseMessage] | None) -> bool:
    return bool(messages) and len(messages) > settings.memory_max_turns


def summarize_messages(messages: list[BaseMessage], existing_summary: str = "") -> str:
    """LLM-резюме переданных сообщений."""
    lines: list[str] = []
    if existing_summary:
        lines.append(f"Предыдущее резюме:\n{existing_summary}\n")
    lines.append("Новые реплики:")
    for msg in messages:
        role = "Пользователь" if isinstance(msg, HumanMessage) else "Ассистент"
        content = str(getattr(msg, "content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")

    result = get_llm(temperature=0).invoke(
        [
            {"role": "system", "content": SUMMARIZE_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ]
    )
    return (result.content or "").strip()


def trim_messages_update(
    messages: list[BaseMessage],
    keep_last: int | None = None,
) -> list:
    """Вернуть обновление messages: сброс + последние keep_last реплики."""
    keep_last = keep_last or settings.memory_context_turns
    if len(messages) <= keep_last:
        return []
    kept = messages[-keep_last:]
    return [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept]
