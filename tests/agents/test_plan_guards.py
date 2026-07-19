"""Тесты для app.agents.plan_guards."""

from app.agents.plan_guards import (
    enforce_plan_guards,
    has_carrier_policy_intent,
    has_flight_intent,
    has_visa_intent,
    is_compound_flight_visa,
)


def test_has_visa_intent():
    assert has_visa_intent("Нужна ли виза в Турцию?")
    assert not has_visa_intent("Найди рейс до Стамбула")


def test_has_flight_intent():
    assert has_flight_intent("Найди рейсы из Москвы в Стамбул")
    assert not has_flight_intent("Нужна виза в Турцию")


def test_has_carrier_policy_intent():
    assert has_carrier_policy_intent("Какая норма багажа у Аэрофлота?")
    assert not has_carrier_policy_intent("Нужна виза в Турцию")


def test_is_compound_flight_visa_true_with_conjunction():
    assert is_compound_flight_visa("Найди рейс в Турцию и подскажи нужна ли виза")


def test_is_compound_flight_visa_false_single_intent():
    assert not is_compound_flight_visa("Нужна ли виза в Турцию")


def test_enforce_plan_guards_adds_compliance_for_visa():
    subtasks = enforce_plan_guards("Нужна виза в Турцию?", [])
    assert any(st["agent"] == "compliance" for st in subtasks)


def test_enforce_plan_guards_adds_compliance_for_carrier_policy():
    subtasks = enforce_plan_guards("Какая норма ручной клади у Победы?", [])
    assert any(st["agent"] == "compliance" for st in subtasks)


def test_enforce_plan_guards_adds_flight_for_compound_query():
    query = "Найди рейс в Турцию и подскажи нужна ли виза"
    subtasks = enforce_plan_guards(query, [{"agent": "compliance", "task": "проверить визу"}])
    agents = {st["agent"] for st in subtasks}
    assert "flight" in agents
    assert "compliance" in agents


def test_enforce_plan_guards_sanitizes_hallucinated_tool():
    subtasks = enforce_plan_guards(
        "Нужна виза в Турцию?",
        [{"agent": "compliance", "task": "используй check_baggage_rules для проверки"}],
    )
    task = next(st["task"] for st in subtasks if st["agent"] == "compliance")
    assert "check_baggage_rules" not in task
    assert "get_carrier_policy" in task or "check_visa_requirements" in task


def test_enforce_plan_guards_dedupes_duplicate_agents():
    subtasks = enforce_plan_guards(
        "Нужна виза в Турцию?",
        [
            {
                "agent": "compliance",
                "task": "проверить визу через check_visa_requirements для Турции",
            },
            {"agent": "compliance", "task": "дополнительно уточнить детали"},
        ],
    )
    compliance_items = [st for st in subtasks if st["agent"] == "compliance"]
    assert len(compliance_items) == 1


def test_enforce_plan_guards_returns_input_unchanged_for_empty_query():
    subtasks = [{"agent": "flight", "task": "t"}]
    assert enforce_plan_guards("", subtasks) is subtasks
