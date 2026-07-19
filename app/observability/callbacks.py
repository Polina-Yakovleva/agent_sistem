"""LangChain callback: счётчики LLM и tool без записи содержимого."""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from app.observability.metrics import record_llm_call, record_tool_call
from app.observability.turn_collector import get_turn_collector


class MetricsCallbackHandler(BaseCallbackHandler):
    """Метрики вызовов LLM/tools по событиям LangChain."""

    def __init__(self, default_caller: str = "graph") -> None:
        super().__init__()
        self._default_caller = default_caller
        self._caller_stack: list[str] = []

    def _current_caller(self) -> str:
        return self._caller_stack[-1] if self._caller_stack else self._default_caller

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or (serialized or {}).get("id") or ""
        if name:
            self._caller_stack.append(str(name)[:64])

    def on_chain_end(self, *args: Any, **kwargs: Any) -> None:
        if self._caller_stack:
            self._caller_stack.pop()

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        record_llm_call(self._current_caller())

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None:
        record_llm_call(self._current_caller())

    def on_tool_end(self, output: str, *, name: str, **kwargs: Any) -> None:
        tool = name or "unknown"
        record_tool_call(tool, success=True)
        col = get_turn_collector()
        if col:
            col.record_tool(tool)

    def on_tool_error(self, error: BaseException, *, name: str, **kwargs: Any) -> None:
        tool = name or "unknown"
        record_tool_call(tool, success=False)
        col = get_turn_collector()
        if col:
            col.record_tool(tool)
