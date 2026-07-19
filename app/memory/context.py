"""Сборка текстового контекста сессии для промптов."""

from langchain_core.messages import BaseMessage, HumanMessage

from app.config import settings


def format_messages_for_prompt(messages: list[BaseMessage] | None, limit: int | None = None) -> str:
    """Отформатировать последние реплики диалога."""
    limit = limit or settings.memory_context_turns
    if not messages:
        return ""
    tail = messages[-limit:] if limit > 0 else messages
    lines: list[str] = []
    for msg in tail:
        role = "Пользователь" if isinstance(msg, HumanMessage) else "Ассистент"
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        text = str(content).strip()
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


def filter_ltm_for_booking(memory_context: str) -> str:
    """Убрать из LTM идентификаторы, чтобы агент бронирования их не подставлял."""
    if not memory_context:
        return ""
    kept: list[str] = []
    for line in memory_context.splitlines():
        lower = line.lower()
        if "user_id" in lower or "ticket_id" in lower or "userid" in lower:
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def build_session_context_block(
    *,
    session_summary: str = "",
    messages: list[BaseMessage] | None = None,
    memory_context: str = "",
) -> str:
    """Единый блок контекста для planner / aggregator / субагентов."""
    parts: list[str] = []
    if memory_context:
        parts.append(f"Долгосрочный контекст:\n{memory_context}")
    if session_summary:
        parts.append(f"Краткое резюме сессии:\n{session_summary}")
    history = format_messages_for_prompt(messages)
    if history:
        parts.append(f"Недавний диалог:\n{history}")
    return "\n\n".join(parts)
