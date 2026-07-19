"""Проверки полноты ответа и traceability перед финализацией."""

from __future__ import annotations

import re

from app.agents.plan_guards import has_flight_intent, has_visa_intent, is_compound_flight_visa

_FLIGHT_ANSWER_RE = re.compile(
    r"\b(рейс|рейсы|перел[её]т|билет|расписан|вылет|"
    r"не найден|нет рейс|вариант|перелет)\b",
    re.IGNORECASE,
)
_VISA_ANSWER_RE = re.compile(
    r"\b(виза|визу|визы|безвиз|не требуется|не нужн|обязательн|оформ)\b",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


def _planned_agents(plan: list[dict] | None) -> set[str]:
    agents: set[str] = set()
    for item in plan or []:
        agent = item.get("agent") if isinstance(item, dict) else None
        if agent:
            agents.add(str(agent))
    return agents


def check_subagent_results_present(
    *,
    user_query: str,
    plan: list[dict] | None,
    subagent_results: dict[str, str] | None,
) -> tuple[bool, str]:
    """Убедиться, что для каждого subtask в плане есть результат субагента."""
    _ = user_query
    planned = _planned_agents(plan)
    results = subagent_results or {}
    missing = [a for a in planned if a not in results or not str(results.get(a, "")).strip()]
    if missing:
        names = ", ".join(missing)
        return False, f"Нет результатов субагентов: {names}. Задействуй их через инструменты."
    return True, ""


def check_compound_answer_coverage(
    *,
    user_query: str,
    plan: list[dict] | None,
    draft_answer: str,
) -> tuple[bool, str]:
    """Compound flight+visa: ответ должен затрагивать оба домена."""
    q = user_query or ""
    if not (is_compound_flight_visa(q) or (has_flight_intent(q) and has_visa_intent(q))):
        return True, ""

    planned = _planned_agents(plan)
    answer = _norm(draft_answer)
    issues: list[str] = []

    if "flight" in planned and not _FLIGHT_ANSWER_RE.search(answer):
        issues.append("в ответе нет информации о рейсах")
    if "compliance" in planned and not _VISA_ANSWER_RE.search(answer):
        issues.append("в ответе нет информации о визе/въезде")

    if issues:
        return False, "Составной запрос: " + "; ".join(issues) + "."
    return True, ""


def check_visa_traceability(
    *,
    user_query: str,
    plan: list[dict] | None,
    subagent_results: dict[str, str] | None,
) -> tuple[bool, str]:
    """Visa / compliance в плане: ответ должен опираться на compliance_agent."""
    if not has_visa_intent(user_query or ""):
        return True, ""

    planned = _planned_agents(plan)
    results = subagent_results or {}

    if "compliance" in planned:
        if "compliance" not in results or not str(results.get("compliance", "")).strip():
            return False, (
                "Визовый запрос: нужен результат compliance_agent через "
                "check_visa_requirements, а не ответ из памяти."
            )
        return True, ""

    if not planned:
        return False, (
            "Визовый запрос нельзя отвечать без compliance_agent и check_visa_requirements."
        )
    return True, ""


def run_response_guards(
    *,
    user_query: str,
    plan: list[dict] | None,
    subagent_results: dict[str, str] | None,
    draft_answer: str,
) -> tuple[bool, str]:
    """Сводная programmatic-проверка перед critic LLM."""
    checks = [
        check_visa_traceability(
            user_query=user_query, plan=plan, subagent_results=subagent_results
        ),
        check_subagent_results_present(
            user_query=user_query, plan=plan, subagent_results=subagent_results
        ),
        check_compound_answer_coverage(user_query=user_query, plan=plan, draft_answer=draft_answer),
    ]
    for ok, msg in checks:
        if not ok:
            return False, msg
    return True, ""
