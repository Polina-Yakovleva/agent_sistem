"""Тесты для app.tools.locations."""

from app.tools.locations import (
    append_destination_filter,
    append_origin_filter,
    location_pattern,
    normalize_location,
)


def test_normalize_location_alias():
    assert normalize_location("Москву") == "москва"
    assert normalize_location("Istanbul") == "стамбул"


def test_normalize_location_passthrough_unknown():
    assert normalize_location("Париж") == "Париж"


def test_location_pattern_wraps_with_wildcards():
    assert location_pattern("Москву") == "%москва%"


def test_append_origin_filter_builds_condition_and_param():
    conditions: list[str] = []
    params: dict = {}
    append_origin_filter(conditions, params, "Москву", city_col="c.city", country_col="c.country")
    assert conditions == ["(c.city ILIKE %(origin)s OR c.country ILIKE %(origin)s)"]
    assert params["origin"] == "%москва%"


def test_append_destination_filter_builds_condition_and_param():
    conditions: list[str] = []
    params: dict = {}
    append_destination_filter(
        conditions, params, "Стамбулу", city_col="c.city", country_col="c.country"
    )
    assert conditions == ["(c.city ILIKE %(destination)s OR c.country ILIKE %(destination)s)"]
    assert params["destination"] == "%стамбул%"
