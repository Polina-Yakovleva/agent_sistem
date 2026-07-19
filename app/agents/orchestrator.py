"""Оркестратор: планировщик (декомпозиция + параллельный диспатч) и агрегатор."""

from langgraph.types import Send

from app.agents.plan_guards import enforce_plan_guards
from app.agents.progress import AGENT_TITLES, progress
from app.agents.prompts import (
    AGGREGATOR_PROMPT,
    MEMORY_CONTEXT_SUFFIX,
    PLANNER_PROMPT,
    PLANNER_REVISION_SUFFIX,
)
from app.agents.state import AgentState, Plan
from app.llm import get_llm
from app.memory.context import build_session_context_block


def _planner_system_prompt(state: AgentState) -> str:
    prompt = PLANNER_PROMPT
    if state.get("critic_feedback") and state.get("revision_count", 0) > 0:
        prompt += PLANNER_REVISION_SUFFIX.format(feedback=state["critic_feedback"])
    ctx = build_session_context_block(
        session_summary=state.get("session_summary") or "",
        messages=state.get("messages"),
        memory_context=state.get("memory_context") or "",
    )
    if ctx.strip():
        prompt += MEMORY_CONTEXT_SUFFIX.format(context=ctx)
    return prompt


def planner_node(state: AgentState) -> dict:
    """Разложить запрос пользователя на подзадачи для субагентов."""
    if state.get("critic_feedback") and state.get("revision_count", 0) > 0:
        progress("оркестратор", "дорабатываю план по замечанию критика…")
    else:
        progress("оркестратор", "анализирую запрос и составляю план…")

    planner = get_llm(temperature=0).with_structured_output(Plan)
    plan: Plan = planner.invoke(
        [
            {"role": "system", "content": _planner_system_prompt(state)},
            {"role": "user", "content": state["user_query"]},
        ]
    )
    subtasks = plan.subtasks
    raw_plan = [st.model_dump() for st in subtasks]
    subtasks = enforce_plan_guards(state["user_query"], raw_plan)
    if subtasks != raw_plan:
        progress("оркестратор", "план дополнен rule-based guard (compliance/compound)")
    if subtasks:
        agents = ", ".join(
            AGENT_TITLES.get(st.get("agent", ""), st.get("agent", ""))
            if isinstance(st, dict)
            else AGENT_TITLES.get(st.agent, st.agent)
            for st in subtasks
        )
        progress("оркестратор", f"задействую: {agents}")
    else:
        progress("оркестратор", "субагенты не требуются, отвечаю напрямую")
    updates: dict = {"plan": subtasks}
    if state.get("revision_count", 0) == 0:
        updates["initial_plan"] = subtasks
    return updates


def dispatch(state: AgentState, *, empty_target: str = "aggregator"):
    """Условный диспатч: параллельно отправить подзадачи субагентам.

    Возвращает список Send (по одному на подзадачу) — это создаёт параллельные
    ветки. Если план пуст, переходим в ``empty_target`` (aggregator или analysis_done).
    """
    plan = state.get("plan") or []
    if not plan:
        return empty_target
    session_ctx = build_session_context_block(
        session_summary=state.get("session_summary") or "",
        messages=state.get("messages"),
        memory_context=state.get("memory_context") or "",
    )
    return [
        Send(
            st["agent"],
            {
                "task": st["task"],
                "user_query": state["user_query"],
                "session_context": session_ctx,
                "memory_context": state.get("memory_context") or "",
                "user_id": state.get("user_id") or "default",
            },
        )
        for st in plan
    ]


def aggregator_node(state: AgentState) -> dict:
    """Собрать единый ответ из результатов субагентов."""
    progress("оркестратор", "собираю итоговый ответ…")
    results = state.get("subagent_results") or {}
    plan = state.get("plan") or []
    if results:
        evidence = "\n\n".join(f"[{name}]\n{text}" for name, text in results.items())
        user_content = (
            f"Запрос пользователя:\n{state['user_query']}\n\nРезультаты субагентов:\n{evidence}"
        )
    elif plan:
        user_content = (
            f"Запрос пользователя:\n{state['user_query']}\n\n"
            f"(Субагенты в плане ({len(plan)}), но результатов пока нет — "
            f"сообщи, что данных недостаточно; не отвечай из общих знаний по визе/рейсам.)"
        )
    else:
        user_content = (
            f"Запрос пользователя:\n{state['user_query']}\n\n"
            f"(Субагенты не привлекались — ответь самостоятельно, если это уместно.)"
        )

    msg = get_llm().invoke(
        [
            {"role": "system", "content": AGGREGATOR_PROMPT},
            {"role": "user", "content": user_content},
        ]
    )
    return {"draft_answer": msg.content or ""}
