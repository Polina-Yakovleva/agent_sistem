"""Оценка retrieval"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from scripts.eval.schema import Case, RagEval


@dataclass
class RagCaseResult:
    case_id: str
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    first_relevant_rank: Optional[int]
    hits: int
    expect_retrieval: bool
    detail: str = ""


def _payload_is_relevant(payload: dict[str, Any], match: dict[str, Any]) -> bool:
    if not payload:
        return False
    if match.get("doc_type") and payload.get("doc_type") != match["doc_type"]:
        return False
    if match.get("entity") and payload.get("entity") != match["entity"]:
        return False
    needles = match.get("text_contains") or []
    if needles:
        text = (payload.get("text") or "").lower()
        if not any(str(n).lower() in text for n in needles):
            return False
    return True


def _query_hits(rag: RagEval):
    """Достать top_k чанков из Qdrant тем же способом, что и продовый retrieve()."""
    from app.config import settings
    from app.rag import _build_filter, _get_client, _get_embeddings

    vector = _get_embeddings().embed_query(rag.query)
    return (
        _get_client()
        .query_points(
            collection_name=settings.qdrant_collection,
            query=vector,
            query_filter=_build_filter(rag.doc_types, rag.entity),
            limit=rag.top_k,
            with_payload=True,
        )
        .points
    )


def evaluate_case(case: Case) -> RagCaseResult:
    rag = case.rag_eval
    if rag is None:
        return RagCaseResult(case.id, 0.0, 0.0, 0.0, None, 0, False, "no rag_eval block")

    hits = _query_hits(rag)
    rel_flags = [_payload_is_relevant(h.payload or {}, rag.relevant_match) for h in hits]
    n_rel = sum(rel_flags)
    first_rank = next((i + 1 for i, r in enumerate(rel_flags) if r), None)
    k = rag.top_k or len(hits) or 1

    recall = 1.0 if n_rel > 0 else 0.0
    precision = n_rel / k
    rr = (1.0 / first_rank) if first_rank else 0.0

    # Если retrieval НЕ ожидается — «успех» означает отсутствие релевантного.
    if not rag.expect_retrieval:
        recall = 1.0 if n_rel == 0 else 0.0

    return RagCaseResult(
        case_id=case.id,
        recall_at_k=recall,
        precision_at_k=precision,
        reciprocal_rank=rr,
        first_relevant_rank=first_rank,
        hits=len(hits),
        expect_retrieval=rag.expect_retrieval,
        detail=f"n_relevant={n_rel}/{len(hits)}",
    )


@dataclass
class RagReport:
    results: list[RagCaseResult] = field(default_factory=list)

    @property
    def recall_at_k(self) -> float:
        return _mean([r.recall_at_k for r in self.results])

    @property
    def precision_at_k(self) -> float:
        return _mean([r.precision_at_k for r in self.results])

    @property
    def mrr(self) -> float:
        return _mean([r.reciprocal_rank for r in self.results])

    def as_dict(self) -> dict:
        return {
            "n": len(self.results),
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "mrr": self.mrr,
            "cases": [r.__dict__ for r in self.results],
        }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def evaluate_rag(cases: list[Case]) -> RagReport:
    """Оценить весь rag-суит (кейсы без блока rag_eval пропускаются)."""
    report = RagReport()
    for case in cases:
        if case.rag_eval is None:
            continue
        report.results.append(evaluate_case(case))
    return report
