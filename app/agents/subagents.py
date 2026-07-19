"""Субагенты: ReAct-агенты (TAO-цикл) и узлы-обёртки для графа.

Каждый субагент — это create_react_agent со своим набором инструментов и
системным промптом. Узел-обёртка принимает подзадачу (через Send), запускает
ReAct-агента и кладёт результат в общий стейт (subagent_results).
"""

from functools import lru_cache

from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import create_react_agent

from app.agents.progress import AGENT_TITLES, progress
from app.agents.prompts import SUBAGENT_PROMPTS
from app.agents.state import AgentState
from app.llm import get_llm
from app.memory.context import build_session_context_block, filter_ltm_for_booking
from app.memory.passenger_profile import (
    format_passenger_profile_for_prompt,
    is_booking_reconfirm_required,
    load_passenger_profile,
)
from app.runtime import set_agent_user_id
from app.tools import (
    BOOKING_TOOLS,
    COMPLIANCE_TOOLS,
    EXTERNAL_TOOLS,
    FLIGHT_TOOLS,
)
from app.tools.passenger_parse import format_passenger_hint, parse_passenger_from_text

_TOOLSETS = {
    "flight": FLIGHT_TOOLS,
    "booking": BOOKING_TOOLS,
    "compliance": COMPLIANCE_TOOLS,
    "external": EXTERNAL_TOOLS,
}


@lru_cache(maxsize=None)
def _build_react_agent(name: str):
    """Создать (и закэшировать) ReAct-агента для субагента name."""
    return create_react_agent(
        model=get_llm(),
        tools=_TOOLSETS[name],
        prompt=SUBAGENT_PROMPTS[name],
        name=f"{name}_agent",
    )


def _run_subagent(name: str, state: AgentState, config: RunnableConfig) -> dict:
    """Запустить субагента на его подзадаче и вернуть результат в общий стейт.

    config пробрасывается во вложенного агента, чтобы Human-in-the-loop
    (interrupt в инструментах booking) корректно приостанавливал весь граф.
    """
    task = state.get("task") or state.get("user_query", "")
    title = AGENT_TITLES.get(name, name)
    progress(title, f"выполняю подзадачу: {task}")
    set_agent_user_id(state.get("user_id") or "default")
    memory_context = state.get("memory_context") or ""
    if name == "booking":
        memory_context = filter_ltm_for_booking(memory_context)
        agent_uid = state.get("user_id") or "default"
        if is_booking_reconfirm_required(agent_uid):
            memory_context = (
                f"{memory_context}\n\n"
                "ВАЖНО: пользователь отменил предыдущее подтверждение бронирования (ответ «нет»). "
                "Запроси заново ФИО и паспорт; не вызывай reserve_ticket без полного набора данных "
                "и не подставляй сохранённый профиль."
            ).strip()
        else:
            profile = load_passenger_profile(agent_uid)
            if profile:
                profile_block = format_passenger_profile_for_prompt(profile)
                memory_context = (
                    f"{memory_context}\n\n{profile_block}".strip()
                    if memory_context
                    else profile_block
                )
    session_ctx = state.get("session_context") or build_session_context_block(
        session_summary=state.get("session_summary") or "",
        messages=state.get("messages"),
        memory_context=memory_context,
    )
    user_content = task
    if name == "booking":
        user_content = (
            "Важно: не подставляй чужой User_ID из эпизодов памяти. "
            "Если в запросе есть ФИО и паспорт — сначала add_passenger, затем reserve_ticket. "
            "passport_issued_by и passport_issue_date можно не указывать.\n\n"
        )
        full_query = state.get("user_query") or task
        parsed = parse_passenger_from_text(full_query)
        if parsed and (parsed.surname or parsed.passport_id):
            user_content += format_passenger_hint(parsed) + "\n\n"
        user_content += f"Текущая подзадача:\n{task}"
        if session_ctx.strip():
            user_content = f"{session_ctx}\n\n{user_content}"
    elif session_ctx.strip():
        user_content = f"{session_ctx}\n\nТекущая подзадача:\n{task}"
    agent = _build_react_agent(name)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_content}]},
        config,
    )
    answer = result["messages"][-1].content if result.get("messages") else ""
    progress(title, "готово")
    return {"subagent_results": {name: answer}}


# Узлы графа (имена совпадают с AgentName). Send передаёт сюда payload с task.
def flight_node(state: AgentState, config: RunnableConfig) -> dict:
    return _run_subagent("flight", state, config)


def booking_node(state: AgentState, config: RunnableConfig) -> dict:
    return _run_subagent("booking", state, config)


def compliance_node(state: AgentState, config: RunnableConfig) -> dict:
    return _run_subagent("compliance", state, config)


def external_node(state: AgentState, config: RunnableConfig) -> dict:
    return _run_subagent("external", state, config)


SUBAGENT_NODES = {
    "flight": flight_node,
    "booking": booking_node,
    "compliance": compliance_node,
    "external": external_node,
}
