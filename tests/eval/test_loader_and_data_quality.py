"""Тесты загрузчика датасета и offline-анализа качества данных."""

from scripts.eval import data_quality
from scripts.eval.loader import load_dataset
from scripts.eval.thresholds import Light


def test_dataset_loads_all_suites():
    ds = load_dataset()
    for suite in ("e2e", "rag", "booking", "multiturn", "out_of_scope"):
        assert suite in ds.suites, f"missing suite {suite}"
    assert "oot" not in ds.suites
    assert ds.suite("e2e").count > 0
    assert ds.suite("rag").cases[0].rag_eval is not None


def test_counts_match_target():
    ds = load_dataset()
    for row in ds.count_report():
        assert row["matches"], f"count mismatch in {row}"


def test_booking_filler_removed():
    ds = load_dataset()
    ids = {c.id for c in ds.suite("booking").cases}
    # старые наполнители DP203..DP211 удалены
    assert not any(
        cid.startswith("book_info_0") and cid[-2:].isdigit() and cid != "book_info_missing_flight"
        for cid in ids
    )
    assert "book_reserve_flow_002" in ids
    assert "book_short_passport" in ids


def test_multiturn_criteria_enriched():
    ds = load_dataset()
    by_id = {c.id: c for c in ds.suite("multiturn").cases}
    turn2 = by_id["mt_uae_trip"].multiturn[1]
    assert "context_retention_turn2" in turn2.success_criteria


def test_data_quality_no_unknown_tools():
    ds = load_dataset()
    rep = data_quality.analyze(ds)
    assert not rep.unknown_tools
    assert not rep.unknown_agents
    section = data_quality.to_section(rep)
    assert section.light in (Light.GREEN, Light.YELLOW)


def test_data_quality_covers_all_agents():
    ds = load_dataset()
    rep = data_quality.analyze(ds)
    assert {
        "flight_agent",
        "compliance_agent",
        "booking_agent",
        "external_agent",
    } <= rep.agents_covered
