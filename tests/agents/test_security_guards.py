"""Тесты для app.agents.security_guards."""

from app.agents.security_guards import (
    check_booking_claims_grounded,
    check_no_full_passport_in_answer,
    check_no_memory_sourcing_compliance,
    run_security_guards,
)


def test_check_no_full_passport_in_answer_detects_number():
    ok, msg = check_no_full_passport_in_answer(draft_answer="Ваш паспорт 1234567890")
    assert not ok
    assert msg


def test_check_no_full_passport_in_answer_ok_when_masked():
    ok, msg = check_no_full_passport_in_answer(draft_answer="Ваш паспорт ****7890")
    assert ok


def test_check_no_memory_sourcing_compliance_flags_reference():
    ok, msg = check_no_memory_sourcing_compliance(
        user_query="Нужна ли виза в Турцию?",
        draft_answer="Ранее сообщали, что виза не нужна",
    )
    assert not ok


def test_check_no_memory_sourcing_compliance_ok_without_visa_intent():
    ok, msg = check_no_memory_sourcing_compliance(
        user_query="Найди рейс", draft_answer="Из памяти: рейс SU100"
    )
    assert ok


def test_check_booking_claims_grounded_requires_tool_call():
    ok, msg = check_booking_claims_grounded(draft_answer="Билет забронирован", tools_called=[])
    assert not ok


def test_check_booking_claims_grounded_ok_with_tool_call():
    ok, msg = check_booking_claims_grounded(
        draft_answer="Билет забронирован", tools_called=["reserve_ticket"]
    )
    assert ok


def test_check_booking_claims_grounded_ignores_denial_answers():
    ok, msg = check_booking_claims_grounded(
        draft_answer="Не удалось забронировать билет: нет мест", tools_called=[]
    )
    assert ok


def test_run_security_guards_detects_first_violation():
    ok, msg = run_security_guards(
        user_query="Забронируй билет", draft_answer="Билет забронирован", tools_called=[]
    )
    assert not ok
    assert msg


def test_run_security_guards_ok_when_all_checks_pass():
    ok, msg = run_security_guards(
        user_query="Забронируй билет",
        draft_answer="Билет забронирован",
        tools_called=["reserve_ticket"],
    )
    assert ok
    assert msg == ""
