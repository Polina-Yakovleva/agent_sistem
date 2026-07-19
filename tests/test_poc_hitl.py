"""Сценарный PoC-тест: happy path + рискованный путь с эскалацией на человека.

Прогоняется в CI без внешней инфраструктуры: доступ к БД (``fetch_all``,
``fetch_one``, ``transaction``) мокается через ``monkeypatch``, LLM не нужен —
happy path и HITL-эскалация проверяются на реальных функциях инструментов.
"""

import pytest

from app.tools import booking as booking_module
from app.tools import flight as flight_module
from examples import poc_hitl


@pytest.fixture
def fake_db(monkeypatch):
    """Изолированные in-memory фейки БД для happy/risky путей."""
    monkeypatch.setattr(flight_module, "fetch_all", lambda *a, **k: list(poc_hitl.FAKE_FLIGHTS))
    monkeypatch.setattr(booking_module, "fetch_one", lambda *a, **k: dict(poc_hitl.FAKE_TICKET))
    monkeypatch.setattr(booking_module, "transaction", poc_hitl._fake_transaction)


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_happy_path_runs_without_escalation(fake_db):
    """Read-only поиск рейсов отрабатывает автономно и возвращает результат."""
    answer = poc_hitl.run_happy_path(origin="Москва", destination="Стамбул")

    assert "Найдено рейсов" in answer
    assert "SU2134" in answer
    assert "TK420" in answer


# --------------------------------------------------------------------------- #
# Рискованный путь — эскалация на человека
# --------------------------------------------------------------------------- #
def test_risky_path_escalates_to_human(fake_db):
    """Необратимая операция ставит граф на паузу и запрашивает подтверждение."""
    outcome = poc_hitl.run_risky_path(ticket_id=42, decision="да")

    assert outcome.escalated is True
    assert "Подтвердите отмену" in outcome.summary
    assert "№42" in outcome.summary


def test_risky_path_aborts_when_human_declines(fake_db):
    """Ответ «нет» отменяет операцию — запись в БД не выполняется."""
    calls = {"transaction": 0}

    def _tracking_transaction(*args, **kwargs):
        calls["transaction"] += 1
        return poc_hitl._fake_transaction()

    # transaction не должен вызываться при отказе.
    booking_module.transaction = _tracking_transaction
    try:
        outcome = poc_hitl.run_risky_path(ticket_id=42, decision="нет")
    finally:
        booking_module.transaction = poc_hitl._fake_transaction

    assert outcome.escalated is True
    assert "не подтвердил" in outcome.final_text
    assert calls["transaction"] == 0


def test_risky_path_executes_when_human_confirms(fake_db):
    """Ответ «да» выполняет необратимую операцию."""
    outcome = poc_hitl.run_risky_path(ticket_id=42, decision="да")

    assert outcome.escalated is True
    assert "отменена" in outcome.final_text.lower()
    assert "№42" in outcome.final_text
