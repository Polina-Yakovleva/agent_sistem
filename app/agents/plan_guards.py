"""Rule-based guards для плана оркестратора (compliance, compound)."""

from __future__ import annotations

import re
from typing import Iterable

from app.agents.state import AgentName, SubTask

_VISA_RE = re.compile(
    r"\b(виза|визу|визы|визой|визе|visa|безвиз|въезд|въезда|"
    r"пребыван|погранич|таможн)\b",
    re.IGNORECASE,
)

_FLIGHT_RE = re.compile(
    r"\b(рейс|рейсы|перел[её]т|билет|расписан|вылет|"
    r"flight|авиабилет)\b",
    re.IGNORECASE,
)

_COMPOUND_CONJ_RE = re.compile(
    r"\b(и|а\s+также|плюс|ещ[её]|также)\b",
    re.IGNORECASE,
)

_CARRIER_POLICY_RE = re.compile(
    r"\b(?:"
    r"багаж\w*|"
    r"ручн.{0,8}клад\w*|"
    r"норм\w*\s+багаж\w*|"
    r"перевозчик\w*|"
    r"авиакомпан\w*|"
    r"животн\w*|"
    r"аэрофлот|aeroflot|"
    r"побед\w*|pobeda|"
    r"s7|"
    r"carry.?on|hand.?luggage|baggage"
    r")\b",
    re.IGNORECASE,
)

# Несуществующие имена инструментов, которые LLM иногда подставляет в task.
_HALLUCINATED_TOOL_REPLACEMENTS: dict[str, str] = {
    "check_baggage_rules": "get_carrier_policy",
    "check_baggage": "get_carrier_policy",
    "baggage_rules": "get_carrier_policy",
    "get_baggage_rules": "get_carrier_policy",
    "check_visa": "check_visa_requirements",
}


def _norm(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


def has_visa_intent(query: str) -> bool:
    return bool(_VISA_RE.search(_norm(query)))


def has_flight_intent(query: str) -> bool:
    return bool(_FLIGHT_RE.search(_norm(query)))


def has_carrier_policy_intent(query: str) -> bool:
    return bool(_CARRIER_POLICY_RE.search(_norm(query)))


def is_compound_flight_visa(query: str) -> bool:
    """Запрос явно объединяет рейсы и визу/въезд."""
    q = _norm(query)
    if not has_visa_intent(q) or not has_flight_intent(q):
        return False
    return bool(_COMPOUND_CONJ_RE.search(q)) or (
        has_flight_intent(q) and has_visa_intent(q) and len(q.split()) >= 4
    )


def _agents_in_plan(subtasks: Iterable[dict | SubTask]) -> set[str]:
    agents: set[str] = set()
    for item in subtasks:
        if isinstance(item, dict):
            agent = item.get("agent")
        else:
            agent = item.agent
        if agent:
            agents.add(str(agent))
    return agents


def _task_for_agent(subtasks: list[dict], agent: str) -> str | None:
    for item in subtasks:
        if item.get("agent") == agent:
            return item.get("task")
    return None


def _upsert_subtask(subtasks: list[dict], agent: AgentName, task: str) -> list[dict]:
    for item in subtasks:
        if item.get("agent") == agent:
            if task not in (item.get("task") or ""):
                item["task"] = f"{item['task']}. {task}".strip(". ")
            return subtasks
    subtasks.append({"agent": agent, "task": task})
    return subtasks


def _compliance_visa_task(query: str) -> str:
    return (
        "Проверить визовые требования и правила въезда через check_visa_requirements "
        f"для запроса: {query.strip()}"
    )


def _flight_task(query: str) -> str:
    return (
        "Найти или проверить рейсы по запросу пользователя через инструменты flight "
        f"(search_flights / get_flights): {query.strip()}"
    )


def _carrier_policy_task(query: str) -> str:
    return (
        "Проверить правила перевозчика и багажа через get_carrier_policy "
        f"для запроса: {query.strip()}"
    )


def _has_hallucinated_tool(task: str) -> bool:
    norm = _norm(task)
    if re.search(r"\bcheck_visa\b(?!_requirements)", task or "", re.IGNORECASE):
        return True
    for fake in _HALLUCINATED_TOOL_REPLACEMENTS:
        if fake == "check_visa":
            continue
        if fake in norm:
            return True
    return False


def _sanitize_task_tools(task: str) -> str:
    out = task or ""
    for fake, real in sorted(_HALLUCINATED_TOOL_REPLACEMENTS.items(), key=lambda x: -len(x[0])):
        if fake == "check_visa":
            out = re.sub(r"\bcheck_visa\b(?!_requirements)", real, out, flags=re.IGNORECASE)
        else:
            out = re.sub(re.escape(fake), real, out, flags=re.IGNORECASE)
    out = re.sub(
        r"check_visa_requirements_requirements",
        "check_visa_requirements",
        out,
        flags=re.IGNORECASE,
    )
    return out


def _sanitize_plan_tools(plan: list[dict]) -> list[dict]:
    for item in plan:
        item["task"] = _sanitize_task_tools(item.get("task") or "")
    return plan


def _dedupe_subtasks_by_agent(subtasks: list[dict]) -> list[dict]:
    """Слить дубли одного agent (LLM иногда даёт два compliance подряд)."""
    merged: list[dict] = []
    for item in subtasks:
        agent = item.get("agent")
        task = (item.get("task") or "").strip()
        if not agent:
            continue
        for existing in merged:
            if existing.get("agent") == agent:
                if task and task not in (existing.get("task") or ""):
                    existing["task"] = f"{existing['task']}. {task}".strip(". ")
                break
        else:
            merged.append({"agent": agent, "task": task})
    return merged


def _needs_compliance_enrichment(task: str, query: str) -> bool:
    """Task слишком короткий, без контекста запроса или с выдуманным инструментом."""
    norm_task = _norm(task)
    if _has_hallucinated_tool(task):
        return True
    if "check_visa" not in norm_task and "виз" not in norm_task:
        if has_visa_intent(query):
            return True
    if has_carrier_policy_intent(query):
        if "get_carrier_policy" not in norm_task and "багаж" not in norm_task:
            return True
    q = _norm(query)
    if q and q not in norm_task and len(task) < max(48, int(len(query) * 0.55)):
        return True
    return False


def enforce_plan_guards(query: str, subtasks: list[dict]) -> list[dict]:
    """Дополнить или исправить план LLM обязательными subtask по intent."""
    q = (query or "").strip()
    if not q:
        return subtasks

    plan = [dict(st) for st in subtasks]
    agents = _agents_in_plan(plan)

    if has_visa_intent(q):
        if "compliance" not in agents:
            plan = _upsert_subtask(plan, "compliance", _compliance_visa_task(q))
        else:
            task = _task_for_agent(plan, "compliance") or ""
            if _needs_compliance_enrichment(task, q):
                plan = _upsert_subtask(plan, "compliance", _compliance_visa_task(q))

    if has_carrier_policy_intent(q):
        if "compliance" not in _agents_in_plan(plan):
            plan = _upsert_subtask(plan, "compliance", _carrier_policy_task(q))
        else:
            task = _task_for_agent(plan, "compliance") or ""
            if _needs_compliance_enrichment(task, q):
                plan = _upsert_subtask(plan, "compliance", _carrier_policy_task(q))

    if is_compound_flight_visa(q) or (has_visa_intent(q) and has_flight_intent(q)):
        if "flight" not in _agents_in_plan(plan):
            plan = _upsert_subtask(plan, "flight", _flight_task(q))
        else:
            task = _task_for_agent(plan, "flight") or ""
            if not _FLIGHT_RE.search(_norm(task)):
                plan = _upsert_subtask(plan, "flight", _flight_task(q))

    return _sanitize_plan_tools(_dedupe_subtasks_by_agent(plan))
