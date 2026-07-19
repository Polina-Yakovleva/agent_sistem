"""Тесты для app.tools.passenger_parse."""

from app.tools.passenger_parse import (
    ParsedPassenger,
    format_passenger_hint,
    parse_passenger_from_text,
)


def test_parse_passenger_from_text_extracts_name_and_passport():
    parsed = parse_passenger_from_text(
        "Забронируй для Иванова Ивана паспорт 1234567890 рейс SU2140"
    )
    assert parsed is not None
    assert parsed.passport_id == "1234567890"
    assert parsed.flight_code == "SU2140"
    assert parsed.surname == "Иванов"
    assert parsed.name == "Иван"


def test_parse_passenger_from_text_passport_only():
    parsed = parse_passenger_from_text("Мой паспорт 9876543210")
    assert parsed is not None
    assert parsed.passport_id == "9876543210"
    assert parsed.surname == ""
    assert parsed.name == ""


def test_parse_passenger_from_text_returns_none_without_data():
    assert parse_passenger_from_text("Привет, как дела?") is None


def test_parse_passenger_from_text_empty_string():
    assert parse_passenger_from_text("") is None
    assert parse_passenger_from_text(None) is None


def test_format_passenger_hint_includes_extracted_fields():
    parsed = ParsedPassenger(
        surname="Иванов", name="Иван", passport_id="1234567890", flight_code="SU2140"
    )
    hint = format_passenger_hint(parsed)
    assert "Иванов Иван" in hint
    assert "1234567890" in hint
    assert "SU2140" in hint
    assert "add_passenger" in hint
