"""Критик: верификация черновика ответа и маршрутизация цикла доработки."""

from app.agents.progress import progress
from app.agents.prompts import CRITIC_PROMPT
from app.agents.response_guards import run_response_guards
from app.agents.security_guards import run_security_guards
from app.agents.state import AgentState, CriticVerdict
from app.config import settings
from app.llm import get_llm
from app.observability.turn_collector import get_turn_collector


def critic_node(state: AgentState) -> dict:
    """Проверить черновик ответа и вынести вердикт."""
    progress("критик", "проверяю ответ…")
    draft = state.get("draft_answer", "") or ""

    guard_ok, guard_feedback = run_response_guards(
        user_query=state.get("user_query", ""),
        plan=state.get("plan"),
        subagent_results=state.get("subagent_results"),
        draft_answer=draft,
    )
    if not guard_ok:
        progress("критик", "есть замечания (traceability), отправляю на доработку")
        return {"critic_ok": False, "critic_feedback": guard_feedback}

    collector = get_turn_collector()
    tools_called = list(collector.tools_called) if collector else []
    sec_ok, sec_feedback = run_security_guards(
        user_query=state.get("user_query", ""),
        draft_answer=draft,
        tools_called=tools_called,
    )
    if not sec_ok:
        progress("критик", "есть замечания (security), отправляю на доработку")
        return {"critic_ok": False, "critic_feedback": sec_feedback}

    user_content = f"Запрос пользователя:\n{state['user_query']}\n\nЧерновик ответа:\n{draft}"
    plan = state.get("plan") or []
    results = state.get("subagent_results") or {}
    if plan:
        evidence = "\n".join(
            f"- {agent}: {'есть' if results.get(agent) else 'нет'}"
            for agent in {item.get("agent") for item in plan if item.get("agent")}
        )
        user_content += f"\n\nПлан и результаты субагентов:\n{evidence}"

    critic = get_llm(temperature=0).with_structured_output(CriticVerdict)
    verdict: CriticVerdict = critic.invoke(
        [
            {"role": "system", "content": CRITIC_PROMPT},
            {"role": "user", "content": user_content},
        ]
    )
    progress("критик", "ответ принят" if verdict.ok else "есть замечания, отправляю на доработку")
    return {"critic_ok": bool(verdict.ok), "critic_feedback": verdict.feedback or ""}


def critic_router(state: AgentState) -> str:
    """Решить: финализировать ответ или отправить на доработку.

    Доработка ограничена settings.max_revisions, чтобы избежать зацикливания.
    """
    if state.get("critic_ok") or state.get("revision_count", 0) >= settings.max_revisions:
        return "finalize"
    return "revise"


def finalize_node(state: AgentState) -> dict:
    """Зафиксировать итоговый ответ для пользователя."""
    return {"final_answer": state.get("draft_answer", "")}


def revise_node(state: AgentState) -> dict:
    """Увеличить счётчик доработок перед повторным планированием."""
    from app.observability.metrics import record_critic_revision

    record_critic_revision()
    return {"revision_count": state.get("revision_count", 0) + 1}
