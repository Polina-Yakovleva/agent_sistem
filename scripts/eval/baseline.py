"""Альтернативное моделирование: упрощённый baseline vs агент.

Baseline — одиночный вызов LLM без планировщика, субагентов и инструментов.
Сравнение с полным агентом на OOS проверяет тезис методики «не избыточен ли
этап планирования» и даёт светофор «улучшает / проще / не даёт выигрыша».
"""

from __future__ import annotations

from dataclasses import dataclass

from scripts.eval.runner import RunOutcome
from scripts.eval.schema import Case
from scripts.eval.scorers.deterministic import case_passed, score_case
from scripts.eval.thresholds import Light, ValidationContext

_BASELINE_PROMPT = """Ты — ассистент авиакомпании. Ответь на запрос пользователя
кратко и по делу на русском. У тебя нет доступа к базам и инструментам —
если для точного ответа нужны данные, честно сообщи об этом."""


def _baseline_answer(query: str) -> str:
    from app.llm import get_llm

    msg = get_llm().invoke(
        [
            {"role": "system", "content": _BASELINE_PROMPT},
            {"role": "user", "content": query},
        ]
    )
    return msg.content or ""


def run_baseline(cases: list[Case]) -> dict:
    """Прогнать упрощённый baseline и посчитать pass-rate по тем же критериям."""
    hits = graded = 0
    per_case: list[dict] = []
    for case in cases:
        answer = _baseline_answer(case.user_query)
        outcome = RunOutcome(case_id=case.id, query=case.user_query, answer=answer)
        verdict = case_passed(score_case(case, outcome))
        if verdict is None:
            continue
        graded += 1
        hits += int(verdict)
        per_case.append({"id": case.id, "passed": verdict})
    return {
        "pass_rate": hits / graded if graded else 0.0,
        "n": graded,
        "per_case": per_case,
    }


@dataclass
class AltVerdict:
    light: Light
    verdict: str
    agent_pass_rate: float
    baseline_pass_rate: float
    delta_pp: float


def compare_alternative(
    agent_pass_rate: float,
    baseline_pass_rate: float,
    ctx: ValidationContext,
) -> AltVerdict:
    """Светофор альтмоделирования по порогу улучшения для текущей СЗ."""
    delta_pp = (agent_pass_rate - baseline_pass_rate) * 100.0
    threshold = ctx.profile.alt_improvement_pp
    if delta_pp >= threshold:
        light, verdict = Light.GREEN, "агент значимо лучше baseline"
    elif delta_pp <= -threshold:
        light, verdict = (
            Light.RED,
            "baseline не хуже/лучше — этап планирования может быть избыточен",
        )
    else:
        light, verdict = Light.YELLOW, "разница в пределах порога — выигрыш не значим"
    return AltVerdict(
        light=light,
        verdict=verdict,
        agent_pass_rate=agent_pass_rate,
        baseline_pass_rate=baseline_pass_rate,
        delta_pp=delta_pp,
    )
