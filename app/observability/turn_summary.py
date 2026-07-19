"""Итоговый структурированный лог хода (для 4.4 golden dataset и 4.5 оценки)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import settings
from app.memory.privacy import redact_pii_text, redact_pii_value
from app.observability.context import get_user_id_hash
from app.observability.logging import get_logger
from app.observability.turn_collector import (
    TurnCollector,
    merge_agents_invoked,
)


def build_turn_summary(
    *,
    collector: TurnCollector | None,
    result: dict[str, Any],
    user_query: str,
    trace_id: str,
) -> dict[str, Any]:
    """Собрать JSON-объект хода по схеме из раздела 4.2 ВКР."""
    plan = result.get("plan")
    tools = list(collector.tools_called) if collector else []
    if not tools and result.get("subagent_results"):
        # fallback: хотя бы зафиксировать факт вызова субагентов
        pass

    latency_ms = collector.latency_ms if collector else 0
    critic_ok = bool(result.get("critic_ok"))
    if result.get("final_answer") and not result.get("critic_feedback"):
        critic_ok = critic_ok or True

    logged_query: str | None = None
    if settings.log_include_query:
        logged_query = redact_pii_text(user_query) if settings.log_redact_pii else user_query

    summary: dict[str, Any] = {
        "event": "turn_summary",
        "trace_id": trace_id,
        "user_query": logged_query,
        "agents_invoked": merge_agents_invoked(collector, plan),
        "tools_called": tools,
        "critic_approved": critic_ok,
        "revision_count": int(result.get("revision_count") or 0),
        "latency_ms": latency_ms,
        "rag_chunks_retrieved": int(collector.rag_chunks_retrieved) if collector else 0,
        "has_final_answer": bool(result.get("final_answer") or result.get("draft_answer")),
        "session_id": result.get("session_id"),
        "user_id_hash": get_user_id_hash() or None,
    }
    if settings.log_include_query:
        summary["query_length"] = len(user_query)
    return summary


def log_turn_summary(summary: dict[str, Any]) -> None:
    """Записать turn_summary в structlog и опционально в JSONL для датасета."""
    if not settings.observability_enabled or not settings.log_turn_summary:
        return

    if settings.log_redact_pii:
        summary = redact_pii_value(summary)  # type: ignore[assignment]

    logger = get_logger("agent.turn")
    logger.info("turn_summary", **{k: v for k, v in summary.items() if k != "event"})

    if settings.turn_summary_jsonl_path:
        path = Path(settings.turn_summary_jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
