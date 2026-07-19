"""Состояние графа и структуры данных оркестрации."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# Доступные субагенты (значение поля agent в плане оркестратора).
AgentName = Literal["flight", "booking", "compliance", "external"]


class SubTask(BaseModel):
    """Одна подзадача из декомпозиции запроса оркестратором."""

    agent: AgentName = Field(description="Какой субагент выполняет подзадачу")
    task: str = Field(description="Формулировка подзадачи для субагента на русском языке")


class Plan(BaseModel):
    """Результат декомпозиции запроса пользователя."""

    subtasks: list[SubTask] = Field(
        default_factory=list,
        description="Список подзадач. Пустой, если запрос не требует инструментов.",
    )


class CriticVerdict(BaseModel):
    """Вердикт критика по черновику ответа."""

    ok: bool = Field(
        description="True, если ответ корректен, полон и можно отправлять пользователю"
    )
    feedback: str = Field(
        default="",
        description="Если ok=False — конкретные замечания и что доработать",
    )


def _merge_results(left: dict | None, right: dict | None) -> dict:
    """Редьюсер: объединяет результаты субагентов из параллельных веток."""
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class AgentState(TypedDict, total=False):
    """Общее состояние графа."""

    # --- STM: сессия ---
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    user_id: str
    session_summary: str

    # --- LTM: контекст, загруженный в memory_read ---
    memory_context: str

    # --- Текущий ход ---
    user_query: str
    plan: list[dict]
    initial_plan: list[dict]
    subagent_results: Annotated[dict[str, str], _merge_results]
    draft_answer: str
    critic_ok: bool
    critic_feedback: str
    revision_count: int
    final_answer: str
