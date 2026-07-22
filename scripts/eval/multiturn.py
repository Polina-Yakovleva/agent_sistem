"""Оценка многоходовых сессий"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from scripts.eval.runner import AgentRunner, RunOutcome
from scripts.eval.schema import Case
from scripts.eval.scorers.deterministic import (
    CriterionResult,
    case_passed,
    score_criteria,
    target_from_turn,
)


@dataclass
class TurnResult:
    index: int
    query: str
    passed: Optional[bool]
    outcome: RunOutcome
    criteria: list[CriterionResult] = field(default_factory=list)


@dataclass
class MultiturnCaseResult:
    case_id: str
    passed: Optional[bool]
    turns: list[TurnResult] = field(default_factory=list)


def evaluate_case(runner: AgentRunner, case: Case) -> MultiturnCaseResult:
    session_id = str(uuid.uuid4()) if case.multiturn_same_session else None
    turn_results: list[TurnResult] = []
    for i, turn in enumerate(case.multiturn):
        sid = session_id if case.multiturn_same_session else str(uuid.uuid4())
        outcome = runner.run_turn(turn.user_query, case_id=f"{case.id}#t{i + 1}", session_id=sid)
        target = target_from_turn(case, turn)
        criteria = score_criteria(turn.success_criteria, target, outcome)
        turn_results.append(
            TurnResult(
                index=i + 1,
                query=turn.user_query,
                passed=case_passed(criteria),
                outcome=outcome,
                criteria=criteria,
            )
        )

    turn_verdicts = [t.passed for t in turn_results if t.passed is not None]
    passed: Optional[bool]
    if not turn_verdicts:
        passed = None
    else:
        passed = all(turn_verdicts)
    return MultiturnCaseResult(case_id=case.id, passed=passed, turns=turn_results)


def evaluate_multiturn(runner: AgentRunner, cases: list[Case]) -> list[MultiturnCaseResult]:
    return [evaluate_case(runner, c) for c in cases]
