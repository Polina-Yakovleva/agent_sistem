"""Оценка одноходовых суитов (e2e и производные) с агрегацией pass-rate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from scripts.eval import checkpoint as ckpt
from scripts.eval.metrics import PassStat
from scripts.eval.runner import AgentRunner, RunOutcome
from scripts.eval.schema import Case
from scripts.eval.scorers.deterministic import (
    CriterionResult,
    case_passed,
    score_case,
)


@dataclass
class CaseResult:
    case_id: str
    passed: Optional[bool]
    outcome: RunOutcome
    criteria: list[CriterionResult] = field(default_factory=list)

    def failed_criteria(self) -> list[str]:
        return [c.name for c in self.criteria if c.applicable and not c.passed]


@dataclass
class SuiteEvalResult:
    suite: str
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def stat(self) -> PassStat:
        graded = [c for c in self.cases if c.passed is not None]
        passed = sum(1 for c in graded if c.passed)
        return PassStat(passed=passed, total=len(graded))

    def as_dict(self) -> dict:
        s = self.stat
        lo, hi = s.wilson_ci()
        return {
            "suite": self.suite,
            "n_graded": s.total,
            "n_passed": s.passed,
            "pass_rate": s.rate,
            "wilson_ci95": [lo, hi],
            "cases": [
                {
                    "id": c.case_id,
                    "passed": c.passed,
                    "failed_criteria": c.failed_criteria(),
                    "status": c.outcome.status,
                    "tools_called": c.outcome.tools_called,
                    "agents_used": c.outcome.agents_used,
                    "latency_ms": c.outcome.latency_ms,
                }
                for c in self.cases
            ],
        }


def evaluate_case(runner: AgentRunner, case: Case) -> CaseResult:
    outcome = runner.run_turn(case.user_query, case_id=case.id)
    criteria = score_case(case, outcome)
    return CaseResult(
        case_id=case.id,
        passed=case_passed(criteria),
        outcome=outcome,
        criteria=criteria,
    )


def evaluate_cases(
    runner: AgentRunner,
    cases: list[Case],
    *,
    suite: str = "e2e",
    checkpoint_stem: Optional[str] = None,
    skip_ids: Optional[set[str]] = None,
) -> SuiteEvalResult:
    """Прогон кейсов. При ``checkpoint_stem`` пишет JSONL и пропускает уже сделанные id."""
    result = SuiteEvalResult(suite=suite)
    path = ckpt.checkpoint_path(suite, checkpoint_stem) if checkpoint_stem else None
    done = set(skip_ids or ())
    if path is not None:
        done |= ckpt.load_done_ids(path)

    for case in cases:
        if case.id in done:
            continue
        cr = evaluate_case(runner, case)
        result.cases.append(cr)
        if path is not None:
            ckpt.append_case(
                path,
                {
                    "id": cr.case_id,
                    "passed": cr.passed,
                    "status": cr.outcome.status,
                    "tools_called": cr.outcome.tools_called,
                    "agents_used": cr.outcome.agents_used,
                    "latency_ms": cr.outcome.latency_ms,
                    "failed_criteria": cr.failed_criteria(),
                    "error": cr.outcome.error,
                },
            )
    return result


def per_tool_quality(cases: list[Case], results_by_id: dict[str, Optional[bool]]) -> dict:
    """Pass-rate кейсов, сгруппированных по expected-инструменту."""
    buckets: dict[str, list[bool]] = {}
    for case in cases:
        tools = list(case.expected_tools) + list(case.expected_tools_any)
        verdict = results_by_id.get(case.id)
        if verdict is None:
            continue
        for tool in tools:
            buckets.setdefault(tool, []).append(bool(verdict))
    return {
        tool: {"n": len(v), "pass_rate": sum(v) / len(v) if v else 0.0}
        for tool, v in sorted(buckets.items())
    }
