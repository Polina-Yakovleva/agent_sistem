"""Тесты детерминированных скореров (offline, без сервисов)."""

from scripts.eval.runner import RunOutcome
from scripts.eval.schema import Case
from scripts.eval.scorers.deterministic import (
    case_passed,
    score_case,
    score_criteria,
    target_from_case,
)


def _case(**raw) -> Case:
    raw.setdefault("id", "c1")
    raw.setdefault("suite", "e2e")
    raw.setdefault("user_query", "q")
    return Case.from_raw(raw, suite="e2e")


def test_has_final_answer_true_false():
    o = RunOutcome(case_id="c1", query="q", answer="Готово")
    r = score_criteria(["has_final_answer"], target_from_case(_case()), o)
    assert r[0].passed is True

    o2 = RunOutcome(case_id="c1", query="q", answer="(пустой ответ)")
    r2 = score_criteria(["has_final_answer"], target_from_case(_case()), o2)
    assert r2[0].passed is False


def test_expected_agents_and_tools_subset():
    case = _case(
        expected_agents=["flight_agent", "compliance_agent"],
        expected_tools=["check_visa_requirements"],
        success_criteria=["expected_agents_subset", "expected_tools_subset"],
    )
    o = RunOutcome(
        case_id="c1",
        query="q",
        answer="ok",
        plan=[{"agent": "flight", "task": "t"}, {"agent": "compliance", "task": "t"}],
        subagent_results={"flight": "рейсы", "compliance": "виза"},
        tools_called=["check_visa_requirements", "get_flights"],
    )
    results = score_case(case, o)
    by = {r.name: r for r in results}
    assert by["expected_agents_subset"].passed
    assert by["expected_tools_subset"].passed


def test_expected_tools_any_hit_missing():
    case = _case(
        expected_tools_any=["get_flights", "search_flights"],
        success_criteria=["expected_tools_any_hit"],
    )
    o = RunOutcome(case_id="c1", query="q", answer="ok", tools_called=["get_weather"])
    r = score_case(case, o)
    assert r[0].passed is False


def test_visa_stated_correctly_negative_fact():
    case = _case(
        reference_facts={"visa_required": False},
        success_criteria=["visa_stated_correctly"],
    )
    ok = RunOutcome(case_id="c1", query="q", answer="Виза не требуется, безвизовый въезд")
    bad = RunOutcome(case_id="c1", query="q", answer="Нужна виза для въезда")
    assert score_case(case, ok)[0].passed is True
    assert score_case(case, bad)[0].passed is False


def test_visa_not_applicable_without_reference():
    case = _case(success_criteria=["visa_stated_correctly"])
    o = RunOutcome(case_id="c1", query="q", answer="что-то")
    r = score_case(case, o)[0]
    assert r.applicable is False


def test_parallel_domains():
    case = _case(success_criteria=["parallel_domains"])
    o = RunOutcome(
        case_id="c1",
        query="q",
        answer="ok",
        plan=[{"agent": "flight", "task": "t"}, {"agent": "compliance", "task": "t"}],
    )
    assert score_case(case, o)[0].passed is True


def test_must_contain_any_auto_checked():
    case = _case(must_contain_any=["Турц"], success_criteria=["has_final_answer"])
    o = RunOutcome(case_id="c1", query="q", answer="Рейсы в Турцию найдены")
    names = {r.name for r in score_case(case, o)}
    assert "must_contain_any" in names


def test_case_passed_none_when_no_applicable():
    case = _case(success_criteria=["visa_stated_correctly"])  # неприменим без reference
    o = RunOutcome(case_id="c1", query="q", answer="x")
    assert case_passed(score_case(case, o)) is None


def test_flights_listed_or_empty_ok():
    case = _case(success_criteria=["flights_listed_or_empty_ok"])
    listed = RunOutcome(
        case_id="c1", query="q", answer="ok", subagent_results={"flight": "Найдено рейсов: 3."}
    )
    empty = RunOutcome(
        case_id="c1",
        query="q",
        answer="ok",
        subagent_results={"flight": "По заданным условиям рейсов не найдено."},
    )
    assert score_case(case, listed)[0].passed is True
    assert score_case(case, empty)[0].passed is True
