"""Абляция рефлексии: эффект удаления критика и границы компетенции.

- ``build_graph_no_critic`` собирает граф без узла критика (aggregator → finalize),
  что позволяет сравнить качество с полным графом (эффект удаления рефлексии);
- ``run_reflection_ablation`` сравнивает pass-rate с критиком и без него;
- ``run_out_of_scope`` проверяет «понимание своих границ»: на нерелевантных/
  «мусорных» запросах агент должен отказать/уточнить, а не выдумывать факты.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scripts.eval.runner import AgentRunner
from scripts.eval.schema import Case
from scripts.eval.scorers.deterministic import _norm, case_passed, score_case


def build_graph_no_critic():
    """Полный граф, но с маршрутом aggregator → finalize (без критика/ревизий)."""
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    from app.agents.critic import finalize_node
    from app.agents.orchestrator import aggregator_node, dispatch, planner_node
    from app.agents.state import AgentState
    from app.agents.subagents import SUBAGENT_NODES
    from app.memory.nodes import (
        memory_read_node,
        memory_write_node,
        summarize_session_node,
    )
    from app.observability.instrumentation import wrap_graph_node

    builder = StateGraph(AgentState)
    builder.add_node("memory_read", wrap_graph_node("memory_read", memory_read_node))
    builder.add_node(
        "summarize_session", wrap_graph_node("summarize_session", summarize_session_node)
    )
    builder.add_node("planner", wrap_graph_node("planner", planner_node))
    for name, node in SUBAGENT_NODES.items():
        builder.add_node(name, wrap_graph_node(name, node))
    builder.add_node("aggregator", wrap_graph_node("aggregator", aggregator_node))
    builder.add_node("finalize", wrap_graph_node("finalize", finalize_node))
    builder.add_node("memory_write", wrap_graph_node("memory_write", memory_write_node))

    builder.add_edge(START, "memory_read")
    builder.add_edge("memory_read", "summarize_session")
    builder.add_edge("summarize_session", "planner")
    builder.add_conditional_edges("planner", dispatch, ["aggregator", *SUBAGENT_NODES.keys()])
    for name in SUBAGENT_NODES:
        builder.add_edge(name, "aggregator")
    builder.add_edge("aggregator", "finalize")
    builder.add_edge("finalize", "memory_write")
    builder.add_edge("memory_write", END)
    return builder.compile(checkpointer=InMemorySaver())


def _pass_rate(runner: AgentRunner, cases: list[Case]) -> tuple[float, int]:
    hits = graded = 0
    for case in cases:
        outcome = runner.run_turn(case.user_query, case_id=case.id)
        verdict = case_passed(score_case(case, outcome))
        if verdict is None:
            continue
        graded += 1
        hits += int(verdict)
    return (hits / graded if graded else 0.0), graded


@dataclass
class ReflectionResult:
    with_critic_pass_rate: float
    no_critic_pass_rate: float
    n: int

    @property
    def critic_gain(self) -> float:
        return self.with_critic_pass_rate - self.no_critic_pass_rate


def run_reflection_ablation(cases: list[Case]) -> ReflectionResult:
    """Сравнить pass-rate полного графа и графа без критика."""
    with AgentRunner() as full:
        base, n = _pass_rate(full, cases)
    with AgentRunner(graph_builder=build_graph_no_critic) as ablated:
        no_critic, _ = _pass_rate(ablated, cases)
    return ReflectionResult(with_critic_pass_rate=base, no_critic_pass_rate=no_critic, n=n)


# Маркеры корректного отказа/уточнения на запросах вне компетенции.
_REFUSAL_MARKERS = [
    "не мог",
    "не могу",
    "не по адресу",
    "не отношусь",
    "уточните",
    "не располагаю",
    "нет данных",
    "не найден",
    "вне моей компетен",
    "помогаю с авиа",
    "не связан",
]


@dataclass
class OutOfScopeResult:
    n: int
    correct_refusals: int
    per_case: list[dict] = field(default_factory=list)

    @property
    def refusal_rate(self) -> float:
        return self.correct_refusals / self.n if self.n else 0.0


def run_out_of_scope(runner: AgentRunner, cases: list[Case]) -> OutOfScopeResult:
    """Проверить понимание границ: агент отказывает/уточняет вместо выдумывания."""
    correct = 0
    per_case: list[dict] = []
    for case in cases:
        outcome = runner.run_turn(case.user_query, case_id=case.id)
        ans = _norm(outcome.answer)
        refused = any(m in ans for m in _REFUSAL_MARKERS)
        no_tools = not outcome.tools_called
        ok = refused or no_tools
        correct += int(ok)
        per_case.append({"id": case.id, "refused": refused, "no_tools": no_tools})
    return OutOfScopeResult(n=len(cases), correct_refusals=correct, per_case=per_case)
