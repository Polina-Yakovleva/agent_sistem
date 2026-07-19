"""Тесты для app.tools.compliance — app.rag.retrieve мокается, Qdrant/эмбеддинги не используются."""

from app.tools import compliance as compliance_module


def test_check_visa_requirements_builds_visa_query(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(compliance_module, "retrieve", lambda **kw: captured.update(kw) or "ctx")

    result = compliance_module.check_visa_requirements.invoke({"country": "Турция"})

    assert result == "ctx"
    assert captured["doc_types"] == ["visa"]
    assert captured["entity"] == "КД МИД"
    assert "Турция" in captured["query"]


def test_check_visa_requirements_uses_custom_question(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(compliance_module, "retrieve", lambda **kw: captured.update(kw) or "ctx")

    compliance_module.check_visa_requirements.invoke(
        {"country": "Турция", "question": "виза по прибытии"}
    )

    assert captured["query"] == "виза по прибытии"


def test_get_carrier_policy_normalizes_airline_and_topic(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(compliance_module, "retrieve", lambda **kw: captured.update(kw) or "ctx")

    compliance_module.get_carrier_policy.invoke({"airline": "аэрофлот", "topic": "багаж"})

    assert captured["entity"] == "Aeroflot"
    assert captured["doc_types"] == ["baggage"]


def test_get_carrier_policy_defaults_to_both_topics(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(compliance_module, "retrieve", lambda **kw: captured.update(kw) or "ctx")

    compliance_module.get_carrier_policy.invoke({"airline": "S7"})

    assert set(captured["doc_types"]) == {"baggage", "animals"}
    assert captured["entity"] == "S7"
