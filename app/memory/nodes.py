"""Узлы графа: memory_read, memory_write, summarize_session."""

from langchain_core.messages import AIMessage

from app.agents.progress import progress
from app.agents.prompts import MEMORY_EXTRACTION_PROMPT
from app.agents.state import AgentState
from app.config import settings
from app.llm import get_llm
from app.memory.context import build_session_context_block
from app.memory.episodic import (
    format_episodes_for_prompt,
    retrieve_episodes,
    store_episode,
)
from app.memory.schemas import MemoryExtraction
from app.memory.store import format_profile_for_prompt, load_profile, upsert_profile_facts
from app.memory.summarize import (
    should_summarize,
    summarize_messages,
    trim_messages_update,
)
from app.observability.logging import get_logger
from app.observability.metrics import record_error

_log = get_logger("memory.nodes")


def _turn_reset_fields() -> dict:
    """Сброс полей одного хода графа (сохраняя messages и session_*)."""
    return {
        "plan": [],
        "initial_plan": [],
        "subagent_results": {},
        "draft_answer": "",
        "critic_ok": False,
        "critic_feedback": "",
        "revision_count": 0,
        "final_answer": "",
    }


def memory_read_node(state: AgentState) -> dict:
    """Загрузить LTM и подготовить memory_context; сбросить поля текущего хода."""
    progress("память", "загружаю контекст…")
    updates = _turn_reset_fields()

    user_id = state.get("user_id") or "default"
    query = state.get("user_query") or ""

    memory_parts: list[str] = []
    if settings.memory_enabled and not settings.eval_mode:
        try:
            profile = load_profile(user_id)
            profile_text = format_profile_for_prompt(profile)
            if profile_text:
                memory_parts.append(profile_text)
        except Exception as exc:
            record_error("memory")
            _log.warning("memory_profile_load_failed", error=str(exc))

        try:
            episodes = retrieve_episodes(user_id, query)
            episode_text = format_episodes_for_prompt(episodes)
            if episode_text:
                memory_parts.append(episode_text)
        except Exception as exc:
            record_error("memory")
            _log.warning("memory_episodes_retrieve_failed", error=str(exc))

    updates["memory_context"] = "\n\n".join(memory_parts)
    return updates


def summarize_session_node(state: AgentState) -> dict:
    """Сжать историю, если сообщений слишком много."""
    messages = state.get("messages") or []
    if not should_summarize(messages):
        return {}

    progress("память", "сжимаю историю сессии…")
    existing = state.get("session_summary") or ""
    new_summary = summarize_messages(messages, existing)
    trim_ops = trim_messages_update(messages)

    out: dict = {"session_summary": new_summary}
    if trim_ops:
        out["messages"] = trim_ops
    return out


def memory_write_node(state: AgentState) -> dict:
    """Записать ход в LTM и добавить ответ ассистента в messages."""
    final = state.get("final_answer") or state.get("draft_answer") or ""
    user_id = state.get("user_id") or "default"
    session_id = state.get("session_id") or ""
    query = state.get("user_query") or ""

    out: dict = {}
    if final:
        out["messages"] = [AIMessage(content=final)]

    if not settings.memory_enabled or not final or settings.eval_mode:
        return out

    progress("память", "сохраняю долгосрочный контекст…")
    try:
        extractor = get_llm(temperature=0).with_structured_output(MemoryExtraction)
        extracted: MemoryExtraction = extractor.invoke(
            [
                {"role": "system", "content": MEMORY_EXTRACTION_PROMPT},
                {
                    "role": "user",
                    "content": (f"Запрос пользователя:\n{query}\n\nОтвет ассистента:\n{final}"),
                },
            ]
        )
        if extracted.profile_facts:
            upsert_profile_facts(user_id, extracted.profile_facts)
        if extracted.episode_summary:
            store_episode(
                user_id=user_id,
                session_id=session_id,
                text=extracted.episode_summary,
                metadata={"query": query[:500]},
            )
    except Exception as exc:
        record_error("memory")
        _log.warning("memory_write_failed", error=str(exc))

    return out


def session_context_for_subagent(state: AgentState) -> str:
    """Краткий контекст сессии для ReAct-субагентов."""
    return build_session_context_block(
        session_summary=state.get("session_summary") or "",
        messages=state.get("messages"),
        memory_context=state.get("memory_context") or "",
    )
