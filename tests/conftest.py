"""Общие фикстуры для unit-тестов."""

import pytest

from app.config import Settings


@pytest.fixture
def make_settings():
    """Фабрика Settings, игнорирующая реальный .env — для изолированных тестов."""

    def _make(**overrides) -> Settings:
        return Settings(_env_file=None, **overrides)

    return _make
