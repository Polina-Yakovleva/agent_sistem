"""Тесты для app.agents.response_guards."""

from app.agents.response_guards import (
    check_compound_answer_coverage,
    check_subagent_results_present,
    check_visa_traceability,
    run_response_guards,
)


def test_check_subagent_results_present_missing():
    ok, msg = check_subagent_results_present(
        user_query="q", plan=[{"agent": "flight", "task": "t"}], subagent_results={}
    )
    assert not ok
    assert "flight" in msg


def test_check_subagent_results_present_ok():
    ok, msg = check_subagent_results_present(
        user_query="q",
        plan=[{"agent": "flight", "task": "t"}],
        subagent_results={"flight": "рейс SU100"},
    )
    assert ok
    assert msg == ""


def test_check_compound_answer_coverage_flags_missing_visa_info():
    query = "Найди рейс в Турцию и нужна ли виза"
    plan = [{"agent": "flight", "task": "..."}, {"agent": "compliance", "task": "..."}]
    ok, msg = check_compound_answer_coverage(
        user_query=query, plan=plan, draft_answer="Рейс SU100 вылетает в 10:00."
    )
    assert not ok
    assert "виз" in msg


def test_check_compound_answer_coverage_ok_when_both_covered():
    query = "Найди рейс в Турцию и нужна ли виза"
    plan = [{"agent": "flight", "task": "..."}, {"agent": "compliance", "task": "..."}]
    answer = "Рейс SU100 вылетает в 10:00. Виза не требуется до 60 дней."
    ok, msg = check_compound_answer_coverage(user_query=query, plan=plan, draft_answer=answer)
    assert ok


def test_check_compound_answer_coverage_skips_non_compound_query():
    ok, msg = check_compound_answer_coverage(
        user_query="Найди рейс в Стамбул", plan=[], draft_answer="ничего о визе"
    )
    assert ok


def test_check_visa_traceability_requires_compliance_result():
    ok, msg = check_visa_traceability(
        user_query="Нужна ли виза в Турцию?",
        plan=[{"agent": "compliance", "task": "t"}],
        subagent_results={},
    )
    assert not ok


def test_check_visa_traceability_ok_with_result():
    ok, msg = check_visa_traceability(
        user_query="Нужна ли виза в Турцию?",
        plan=[{"agent": "compliance", "task": "t"}],
        subagent_results={"compliance": "виза не нужна"},
    )
    assert ok


def test_check_visa_traceability_fails_without_compliance_in_plan():
    ok, msg = check_visa_traceability(
        user_query="Нужна ли виза в Турцию?", plan=[], subagent_results={}
    )
    assert not ok


def test_run_response_guards_aggregates_first_failure():
    ok, msg = run_response_guards(
        user_query="Нужна ли виза в Турцию?",
        plan=[{"agent": "compliance", "task": "t"}],
        subagent_results={},
        draft_answer="ответ",
    )
    assert not ok
    assert msg


def test_run_response_guards_ok_when_all_checks_pass():
    ok, msg = run_response_guards(
        user_query="Найди рейс в Стамбул",
        plan=[{"agent": "flight", "task": "t"}],
        subagent_results={"flight": "рейс SU100"},
        draft_answer="Рейс SU100 вылетает в 10:00.",
    )
    assert ok
    assert msg == ""
