"""Тесты для app.config.Settings."""


def test_defaults_when_no_env_file(make_settings):
    s = make_settings()
    assert s.pg_host == "localhost"
    assert s.pg_port == 5432
    assert s.qdrant_port == 6333
    assert s.llm_temperature == 0.1
    assert s.memory_enabled is True
    assert s.checkpoint_backend == "postgres"


def test_pg_url_encodes_credentials(make_settings):
    s = make_settings(
        pg_user="a b",
        pg_password="p@ss/word",
        pg_host="db",
        pg_port=5555,
        pg_database="mydb",
    )
    assert s.pg_url == "postgresql://a+b:p%40ss%2Fword@db:5555/mydb"


def test_pg_conninfo_contains_all_fields(make_settings):
    s = make_settings(pg_host="h", pg_port=1, pg_database="d", pg_user="u", pg_password="pw")
    conninfo = s.pg_conninfo
    assert "host=h" in conninfo
    assert "port=1" in conninfo
    assert "dbname=d" in conninfo
    assert "user=u" in conninfo
    assert "password=pw" in conninfo


def test_env_override(monkeypatch, make_settings):
    monkeypatch.setenv("PG_HOST", "custom-host")
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    s = make_settings()
    assert s.pg_host == "custom-host"
    assert s.llm_model == "custom-model"
