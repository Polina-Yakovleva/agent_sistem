"""Детерминированные скореры под каждый ``success_criteria`` золотого датасета.

Каждый критерий возвращает :class:`CriterionResult` с флагом ``applicable``:
если критерий нельзя проверить по имеющимся эталонам (например,
``visa_stated_correctly`` без ``reference_facts.visa_required``), он помечается
неприменимым и не искажает pass-rate раздела.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from scripts.eval.runner import RunOutcome
from scripts.eval.schema import Case, Turn


@dataclass
class CriterionResult:
    name: str
    passed: bool
    detail: str = ""
    applicable: bool = True


@dataclass
class Target:
    """Ожидания, против которых проверяется исход хода."""

    expected_agents: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    expected_tools_any: list[str] = field(default_factory=list)
    reference_facts: dict[str, Any] = field(default_factory=dict)
    must_contain_any: list[str] = field(default_factory=list)


def _norm(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


def _contains_any(text: str, needles: list[str]) -> bool:
    hay = _norm(text)
    return any(_norm(n) in hay for n in needles if n)


def _plan_agents(plan: list[dict] | None) -> set[str]:
    out: set[str] = set()
    for item in plan or []:
        agent = item.get("agent") if isinstance(item, dict) else getattr(item, "agent", None)
        if agent:
            out.add(str(agent))
    return out


def _bare(agent: str) -> str:
    return agent[: -len("_agent")] if agent.endswith("_agent") else agent


# --- Отдельные критерии --------------------------------------------------- #
def _has_final_answer(t: Target, o: RunOutcome) -> CriterionResult:
    return CriterionResult("has_final_answer", o.has_answer, f"answer_len={len(o.answer or '')}")


def _used_agents(o: RunOutcome) -> set[str]:
    used = set(o.agents_used)
    if not used:
        used = {f"{a}_agent" for a in _plan_agents(o.plan)}
        used |= {f"{k}_agent" for k in (o.subagent_results or {})}
    return used


def _expected_agents_subset(t: Target, o: RunOutcome) -> CriterionResult:
    if not t.expected_agents:
        return CriterionResult(
            "expected_agents_subset", True, "no expected_agents", applicable=False
        )
    used = _used_agents(o)
    missing = [a for a in t.expected_agents if a not in used]
    return CriterionResult(
        "expected_agents_subset", not missing, f"missing={missing}; used={sorted(used)}"
    )


def _expected_tools_subset(t: Target, o: RunOutcome) -> CriterionResult:
    if not t.expected_tools:
        return CriterionResult("expected_tools_subset", True, "no expected_tools", applicable=False)
    called = set(o.tools_called)
    missing = [x for x in t.expected_tools if x not in called]
    return CriterionResult(
        "expected_tools_subset", not missing, f"missing={missing}; called={sorted(called)}"
    )


def _expected_tools_any_hit(t: Target, o: RunOutcome) -> CriterionResult:
    if not t.expected_tools_any:
        return CriterionResult(
            "expected_tools_any_hit", True, "no expected_tools_any", applicable=False
        )
    hit = set(t.expected_tools_any) & set(o.tools_called)
    return CriterionResult("expected_tools_any_hit", bool(hit), f"hit={sorted(hit)}")


def _parallel_domains(t: Target, o: RunOutcome) -> CriterionResult:
    agents = _plan_agents(o.plan) or {_bare(a) for a in o.agents_used}
    return CriterionResult(
        "parallel_domains", len(agents) >= 2, f"distinct_agents={sorted(agents)}"
    )


_VISA_NEG = ["не нужн", "не треб", "безвиз", "без визы", "виза не"]
_VISA_POS = ["нужна виз", "требуется виз", "необходима виз", "нужно оформить виз", "визу нужно"]


def _visa_stated_correctly(t: Target, o: RunOutcome) -> CriterionResult:
    if "visa_required" not in t.reference_facts:
        return CriterionResult(
            "visa_stated_correctly", True, "no reference visa_required", applicable=False
        )
    required = bool(t.reference_facts["visa_required"])
    ans = _norm(o.answer)
    says_no = any(_norm(s) in ans for s in _VISA_NEG)
    says_yes = any(_norm(s) in ans for s in _VISA_POS)
    if required:
        ok = says_yes and not says_no
    else:
        ok = says_no and not says_yes
    return CriterionResult(
        "visa_stated_correctly", ok, f"required={required}; says_yes={says_yes}; says_no={says_no}"
    )


def _plan_agents_ok(t: Target, o: RunOutcome) -> CriterionResult:
    if not t.expected_agents:
        return CriterionResult("plan_agents_ok", True, "no expected_agents", applicable=False)
    plan_agents = _plan_agents(o.plan) | _plan_agents(o.initial_plan)
    expected_bare = {_bare(a) for a in t.expected_agents}
    missing = sorted(expected_bare - plan_agents)
    return CriterionResult(
        "plan_agents_ok", not missing, f"missing={missing}; plan={sorted(plan_agents)}"
    )


# Ключевые слова параметров подзадачи по агенту.
def _plan_task_params_ok(t: Target, o: RunOutcome) -> CriterionResult:
    if not t.expected_agents:
        return CriterionResult("plan_task_params_ok", True, "no expected_agents", applicable=False)
    tasks_by_agent: dict[str, str] = {}
    for item in (o.plan or []) + (o.initial_plan or []):
        if isinstance(item, dict) and item.get("agent"):
            tasks_by_agent.setdefault(str(item["agent"]), "")
            tasks_by_agent[str(item["agent"])] += " " + _norm(item.get("task") or "")

    rf = t.reference_facts
    problems: list[str] = []
    for agent in {_bare(a) for a in t.expected_agents}:
        task = tasks_by_agent.get(agent, "")
        if not task:
            problems.append(f"{agent}:no-task")
            continue
        if agent == "compliance":
            country = _norm(str(rf.get("destination_country") or ""))
            ok = "виз" in task or "багаж" in task or (country and country[:4] in task)
            if not ok:
                problems.append(f"{agent}:no-country/topic")
        elif agent == "flight":
            if not any(k in task for k in ("рейс", "перел", "москв", "полет", "flight", "билет")):
                problems.append(f"{agent}:no-route")
    return CriterionResult("plan_task_params_ok", not problems, f"problems={problems}")


def _is_error_result(text: str) -> bool:
    n = _norm(text)
    return (not n) or ("ошибк" in n and "валидац" not in n) or "traceback" in n


def _both_subtasks_addressed(t: Target, o: RunOutcome) -> CriterionResult:
    expected = {_bare(a) for a in t.expected_agents} or _plan_agents(o.plan)
    if len(expected) < 2:
        expected = _plan_agents(o.plan)
    if len(expected) < 2:
        return CriterionResult(
            "both_subtasks_addressed", False, f"expected<2 agents: {sorted(expected)}"
        )
    results = o.subagent_results or {}
    missing = [a for a in expected if _is_error_result(results.get(a, ""))]
    ok = (not missing) and o.has_answer
    return CriterionResult(
        "both_subtasks_addressed", ok, f"missing_or_empty={missing}; has_answer={o.has_answer}"
    )


_FLIGHT_LISTED = ["рейс", "найдено рейсов", "flight_id", "id "]
_FLIGHT_EMPTY_OK = ["не найден", "не найдено", "нет рейс", "отсутств"]


def _flights_listed_or_empty_ok(t: Target, o: RunOutcome) -> CriterionResult:
    text = (o.subagent_results or {}).get("flight", "")
    if not text:
        # fallback на итоговый ответ, если субагент не разложен по ключам
        text = o.answer
    n = _norm(text)
    listed = any(_norm(s) in n for s in _FLIGHT_LISTED)
    empty_ok = any(_norm(s) in n for s in _FLIGHT_EMPTY_OK)
    ok = (listed or empty_ok) and not _is_error_result(text)
    return CriterionResult(
        "flights_listed_or_empty_ok", ok, f"listed={listed}; empty_ok={empty_ok}"
    )


def _must_contain_any(t: Target, o: RunOutcome) -> CriterionResult:
    if not t.must_contain_any:
        return CriterionResult("must_contain_any", True, "no must_contain_any", applicable=False)
    ok = _contains_any(o.answer, t.must_contain_any)
    return CriterionResult("must_contain_any", ok, f"needles={t.must_contain_any}")


_CLARIFY_MARKERS = ["уточните", "какое направление", "не понял", "переформулируйте"]


def _context_retention_turn2(t: Target, o: RunOutcome) -> CriterionResult:
    """Ответ 2-го хода удерживает контекст 1-го (нет переспроса, есть сущность)."""
    ans = _norm(o.answer)
    reclarifies = any(m in ans for m in _CLARIFY_MARKERS)
    entity_ok = True
    if t.must_contain_any:
        entity_ok = _contains_any(o.answer, t.must_contain_any)
    elif t.reference_facts:
        vals = [str(v) for v in t.reference_facts.values() if isinstance(v, (str, int))]
        entity_ok = (not vals) or _contains_any(o.answer, vals)
    ok = o.has_answer and entity_ok and not reclarifies
    return CriterionResult(
        "context_retention_turn2", ok, f"entity_ok={entity_ok}; reclarifies={reclarifies}"
    )


_SCORERS: dict[str, Callable[[Target, RunOutcome], CriterionResult]] = {
    "has_final_answer": _has_final_answer,
    "expected_agents_subset": _expected_agents_subset,
    "expected_tools_subset": _expected_tools_subset,
    "expected_tools_any_hit": _expected_tools_any_hit,
    "parallel_domains": _parallel_domains,
    "visa_stated_correctly": _visa_stated_correctly,
    "plan_agents_ok": _plan_agents_ok,
    "plan_task_params_ok": _plan_task_params_ok,
    "both_subtasks_addressed": _both_subtasks_addressed,
    "flights_listed_or_empty_ok": _flights_listed_or_empty_ok,
    "context_retention_turn2": _context_retention_turn2,
    "must_contain_any": _must_contain_any,
}


def available_criteria() -> list[str]:
    return sorted(_SCORERS)


def target_from_case(case: Case) -> Target:
    return Target(
        expected_agents=case.expected_agents,
        expected_tools=case.expected_tools,
        expected_tools_any=case.expected_tools_any,
        reference_facts=case.reference_facts,
        must_contain_any=case.must_contain_any,
    )


def target_from_turn(case: Case, turn: Turn) -> Target:
    """Ожидания хода: агенты/инструменты — с уровня кейса, факты/фразы — с хода."""
    return Target(
        expected_agents=case.expected_agents,
        expected_tools=case.expected_tools,
        expected_tools_any=case.expected_tools_any,
        reference_facts=turn.reference_facts or case.reference_facts,
        must_contain_any=turn.must_contain_any or case.must_contain_any,
    )


def score_criteria(
    criteria: list[str], target: Target, outcome: RunOutcome
) -> list[CriterionResult]:
    """Оценить список критериев. ``must_contain_any`` проверяется, если задан."""
    results: list[CriterionResult] = []
    seen: set[str] = set()
    for name in criteria:
        fn = _SCORERS.get(name)
        if fn is None:
            results.append(CriterionResult(name, False, "unknown criterion", applicable=False))
            continue
        results.append(fn(target, outcome))
        seen.add(name)
    # must_contain_any проверяем всегда, если он есть и не был в списке критериев
    if target.must_contain_any and "must_contain_any" not in seen:
        results.append(_must_contain_any(target, outcome))
    return results


def score_case(case: Case, outcome: RunOutcome) -> list[CriterionResult]:
    """Оценить одноходовый кейс по его ``success_criteria``."""
    return score_criteria(case.success_criteria, target_from_case(case), outcome)


def case_passed(results: list[CriterionResult]) -> Optional[bool]:
    """Кейс пройден, если пройдены все применимые критерии. None — если нечего мерить."""
    applicable = [r for r in results if r.applicable]
    if not applicable:
        return None
    return all(r.passed for r in applicable)
