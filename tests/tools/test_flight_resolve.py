"""Тесты для app.tools.flight_resolve — app.db.fetch_one мокается, реальная БД не используется."""

from app.tools import flight_resolve


def test_resolve_flight_id_returns_int_passthrough():
    assert flight_resolve.resolve_flight_id(42) == 42


def test_resolve_flight_id_numeric_string():
    assert flight_resolve.resolve_flight_id("42") == 42


def test_resolve_flight_id_empty_string_returns_none():
    assert flight_resolve.resolve_flight_id("   ") is None


def test_resolve_flight_id_looks_up_by_flight_number(monkeypatch):
    calls = {}

    def fake_fetch_one(query, params):
        calls["query"] = query
        calls["params"] = params
        return {"Flight_ID": 777}

    monkeypatch.setattr(flight_resolve, "fetch_one", fake_fetch_one)
    result = flight_resolve.resolve_flight_id("su2140")
    assert result == 777
    assert calls["params"] == {"n": "su2140"}


def test_resolve_flight_id_returns_none_when_not_found(monkeypatch):
    monkeypatch.setattr(flight_resolve, "fetch_one", lambda query, params: None)
    assert flight_resolve.resolve_flight_id("XX999") is None


def test_fetch_flight_row_returns_none_when_flight_not_resolved(monkeypatch):
    monkeypatch.setattr(flight_resolve, "fetch_one", lambda query, params: None)
    assert flight_resolve.fetch_flight_row("XX999") is None


def test_fetch_flight_row_queries_details_for_resolved_flight(monkeypatch):
    calls = []

    def fake_fetch_one(query, params):
        calls.append((query, params))
        if len(calls) == 1:
            return {"Flight_ID": 5}
        return {
            "Flight_ID": 5,
            "Flight_Number": "SU100",
            "origin": "Москва",
            "destination": "Стамбул",
        }

    monkeypatch.setattr(flight_resolve, "fetch_one", fake_fetch_one)
    row = flight_resolve.fetch_flight_row("SU100")
    assert row["Flight_Number"] == "SU100"
    assert len(calls) == 2
