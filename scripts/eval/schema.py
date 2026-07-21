"""Модель кейсов золотого датасета."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RagEval:
    """Блок оценки retrieval для суита ``rag``."""

    query: str
    doc_types: list[str]
    entity: Optional[str]
    relevant_match: dict[str, Any]
    expect_retrieval: bool = True
    top_k: int = 5

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "RagEval":
        return cls(
            query=raw.get("query", ""),
            doc_types=list(raw.get("doc_types") or []),
            entity=raw.get("entity"),
            relevant_match=dict(raw.get("relevant_match") or {}),
            expect_retrieval=bool(raw.get("expect_retrieval", True)),
            top_k=int(raw.get("top_k", 5)),
        )


@dataclass
class Turn:
    """Один ход многоходового кейса."""

    user_query: str
    success_criteria: list[str] = field(default_factory=list)
    must_contain_any: list[str] = field(default_factory=list)
    reference_facts: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "Turn":
        return cls(
            user_query=raw.get("user_query", ""),
            success_criteria=list(raw.get("success_criteria") or []),
            must_contain_any=list(raw.get("must_contain_any") or []),
            reference_facts=dict(raw.get("reference_facts") or {}),
        )


@dataclass
class Case:
    """Кейс золотого датасета (единый для всех суитов)."""

    id: str
    suite: str
    user_query: str
    raw: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"

    expected_agents: list[str] = field(default_factory=list)
    expected_tools: list[str] = field(default_factory=list)
    expected_tools_any: list[str] = field(default_factory=list)
    reference_facts: dict[str, Any] = field(default_factory=dict)
    must_contain_any: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)

    rag_eval: Optional[RagEval] = None
    booking_eval: dict[str, Any] = field(default_factory=dict)
    multiturn: list[Turn] = field(default_factory=list)
    multiturn_same_session: bool = True

    @classmethod
    def from_raw(cls, raw: dict[str, Any], *, suite: str) -> "Case":
        mt_block = raw.get("multiturn") or {}
        turns = [Turn.from_raw(t) for t in (mt_block.get("turns") or [])]
        rag_raw = raw.get("rag_eval")
        return cls(
            id=raw.get("id", ""),
            suite=raw.get("suite", suite),
            user_query=raw.get("user_query", ""),
            raw=raw,
            tags=list(raw.get("tags") or []),
            difficulty=raw.get("difficulty", "medium"),
            expected_agents=list(raw.get("expected_agents") or []),
            expected_tools=list(raw.get("expected_tools") or []),
            expected_tools_any=list(raw.get("expected_tools_any") or []),
            reference_facts=dict(raw.get("reference_facts") or {}),
            must_contain_any=list(raw.get("must_contain_any") or []),
            success_criteria=list(raw.get("success_criteria") or []),
            rag_eval=RagEval.from_raw(rag_raw) if rag_raw else None,
            booking_eval=dict(raw.get("booking_eval") or {}),
            multiturn=turns,
            multiturn_same_session=bool(mt_block.get("same_session", True)),
        )

    @property
    def is_multiturn(self) -> bool:
        return bool(self.multiturn)
