"""Тесты для app.memory.privacy."""

from app.memory.privacy import (
    contains_unmasked_passport,
    is_blocked_profile_key,
    redact_pii_text,
    redact_pii_value,
    sanitize_episode_text,
    sanitize_profile_facts,
)


def test_is_blocked_profile_key():
    assert is_blocked_profile_key("passport_id")
    assert is_blocked_profile_key("User-ID")
    assert not is_blocked_profile_key("favorite_seat")


def test_contains_unmasked_passport():
    assert contains_unmasked_passport("паспорт 1234567890")
    assert not contains_unmasked_passport("паспорт ****7890")


def test_redact_pii_text_masks_passport():
    redacted = redact_pii_text("Мой паспорт 1234567890 действителен")
    assert "1234567890" not in redacted
    assert "[passport]" in redacted


def test_redact_pii_text_empty_string():
    assert redact_pii_text("") == ""


def test_redact_pii_value_recurses_into_dict_and_list():
    value = {"a": "паспорт 1234567890", "b": ["текст 9876543210", 42]}
    redacted = redact_pii_value(value)
    assert "1234567890" not in redacted["a"]
    assert "9876543210" not in redacted["b"][0]
    assert redacted["b"][1] == 42


def test_sanitize_profile_facts_drops_blocked_keys_and_sensitive_values():
    facts = {
        "passport_id": "1234567890",
        "favorite_seat": "12A",
        "note": "паспорт 1234567890",
    }
    clean = sanitize_profile_facts(facts)
    assert "passport_id" not in clean
    assert clean.get("favorite_seat") == "12A"
    assert "note" not in clean


def test_sanitize_episode_text_removes_passport_and_blanks_sensitive():
    assert sanitize_episode_text("Уточнил паспорт 1234567890") == ""


def test_sanitize_episode_text_keeps_plain_text():
    assert sanitize_episode_text("Обычный текст без ПДн") == "Обычный текст без ПДн"
