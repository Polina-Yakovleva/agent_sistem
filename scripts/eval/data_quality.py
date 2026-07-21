"""Качество данных золотого датасета без прогонов LLM.

Проверяет сверку count/target_count, дубли запросов внутри суита и «утечки»
(одинаковые запросы в разных суитах), а также покрытие агентов/инструментов.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scripts.eval.loader import Dataset
from scripts.eval.metrics import SectionResult
from scripts.eval.thresholds import Light

_KNOWN_AGENTS = {"flight_agent", "booking_agent", "compliance_agent", "external_agent"}
_KNOWN_TOOLS = {
    "get_flights",
    "get_flight_details",
    "search_flights",
    "add_passenger",
    "reserve_ticket",
    "cancel_reservation",
    "check_visa_requirements",
    "get_carrier_policy",
    "get_weather",
    "search_nearby_hotels",
}


@dataclass
class DataQualityReport:
    count_rows: list[dict] = field(default_factory=list)
    intra_suite_duplicates: dict[str, list[str]] = field(default_factory=dict)
    cross_suite_leaks: list[dict] = field(default_factory=list)
    agents_covered: set[str] = field(default_factory=set)
    tools_covered: set[str] = field(default_factory=set)
    unknown_agents: set[str] = field(default_factory=set)
    unknown_tools: set[str] = field(default_factory=set)

    def as_dict(self) -> dict:
        return {
            "counts": self.count_rows,
            "intra_suite_duplicates": self.intra_suite_duplicates,
            "cross_suite_leaks": self.cross_suite_leaks,
            "agents_covered": sorted(self.agents_covered),
            "tools_covered": sorted(self.tools_covered),
            "agents_missing": sorted(_KNOWN_AGENTS - self.agents_covered),
            "tools_missing": sorted(_KNOWN_TOOLS - self.tools_covered),
            "unknown_agents": sorted(self.unknown_agents),
            "unknown_tools": sorted(self.unknown_tools),
        }


def _norm_q(text: str) -> str:
    return " ".join((text or "").lower().replace("ё", "е").split())


def analyze(dataset: Dataset) -> DataQualityReport:
    rep = DataQualityReport(count_rows=dataset.count_report())
    seen_global: dict[str, str] = {}

    for suite_name, suite in dataset.suites.items():
        seen_local: dict[str, int] = {}
        for case in suite.cases:
            rep.agents_covered.update(case.expected_agents)
            rep.unknown_agents.update(set(case.expected_agents) - _KNOWN_AGENTS)
            tools = set(case.expected_tools) | set(case.expected_tools_any)
            rep.tools_covered.update(tools)
            rep.unknown_tools.update(tools - _KNOWN_TOOLS)

            queries = [case.user_query]
            if case.is_multiturn:
                queries = [t.user_query for t in case.multiturn]
            for q in queries:
                key = _norm_q(q)
                if not key or key == "placeholder":
                    continue
                seen_local[key] = seen_local.get(key, 0) + 1
                if key in seen_global and seen_global[key] != suite_name:
                    rep.cross_suite_leaks.append(
                        {"query": q, "suites": [seen_global[key], suite_name]}
                    )
                seen_global.setdefault(key, suite_name)
        dups = [q for q, c in seen_local.items() if c > 1]
        if dups:
            rep.intra_suite_duplicates[suite_name] = dups

    return rep


def to_section(rep: DataQualityReport) -> SectionResult:
    """Светофор Раздела 1: красный при несовпадении count или неизвестных инструментах."""
    count_mismatch = [r for r in rep.count_rows if not r["matches"]]
    unknown = rep.unknown_agents or rep.unknown_tools
    dups = sum(len(v) for v in rep.intra_suite_duplicates.values())

    if unknown or count_mismatch:
        light = Light.RED
    elif dups > 0 or rep.cross_suite_leaks:
        light = Light.YELLOW
    else:
        light = Light.GREEN
    comment = (
        f"count_mismatch={len(count_mismatch)}; intra_dups={dups}; "
        f"cross_leaks={len(rep.cross_suite_leaks)}; "
        f"unknown_tools={sorted(rep.unknown_tools)}"
    )
    return SectionResult(
        name="Качество данных", light=light, comment=comment, metrics=rep.as_dict()
    )
