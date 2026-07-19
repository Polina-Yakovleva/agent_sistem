"""PoC: happy path (`get_flights`) + risky path with HITL (`cancel_reservation`).

Run without Postgres/LLM: ``python -m examples.poc_hitl``.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict

from app.tools import booking as booking_module
from app.tools import flight as flight_module

# --------------------------------------------------------------------------- #
# In-memory фейки БД (чтобы PoC работал без Postgres)
# --------------------------------------------------------------------------- #
FAKE_FLIGHTS: list[dict[str, Any]] = [
    {
        "flight_id": 12,
        "flight_number": "SU2134",
        "airline": "Aeroflot",
        "origin_city": "Москва",
        "origin_airport": "Шереметьево",
        "destination_city": "Стамбул",
        "destination_airport": "IST",
        "departure_date": "2026-08-01",
        "departure_time": "10:30",
        "arrival_date": "2026-08-01",
        "arrival_time": "13:45",
        "price": 18500,
        "occupied_seats": 40,
    },
    {
        "flight_id": 15,
        "flight_number": "TK420",
        "airline": "Turkish Airlines",
        "origin_city": "Москва",
        "origin_airport": "Внуково",
        "destination_city": "Стамбул",
        "destination_airport": "SAW",
        "departure_date": "2026-08-01",
        "departure_time": "18:05",
        "arrival_date": "2026-08-01",
        "arrival_time": "21:10",
        "price": 21200,
        "occupied_seats": 120,
    },
]

FAKE_TICKET: dict[str, Any] = {
    "Ticket_ID": 42,
    "Flight_ID": 12,
    "Ticket_Status_ID": 0,  # «Оформлен» — билет активен, отмена имеет смысл
    "User_surname": "Иванов",
    "User_name": "Иван",
    "User_patronymic": "Иванович",
    "seat_number": 14,
}


class _FakeCursor:
    """Курсор-заглушка: принимает любые execute/fetch без реальной БД."""

    def execute(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - заглушка
        return None

    def fetchone(self) -> Optional[dict]:
        return None

    def fetchall(self) -> list[dict]:
        return []


@contextmanager
def _fake_transaction() -> Iterator[_FakeCursor]:
    """Транзакция-заглушка: имитирует атомарный write без Postgres."""
    yield _FakeCursor()


def install_in_memory_backends(*, tickets: Optional[dict[str, Any]] = None) -> None:
    """Подменить доступ к БД in-memory фейками (для standalone-запуска PoC).

    В тестах вместо этого используйте ``monkeypatch`` для изоляции.
    """
    ticket = tickets if tickets is not None else FAKE_TICKET

    flight_module.fetch_all = lambda *a, **k: list(FAKE_FLIGHTS)
    booking_module.fetch_one = lambda *a, **k: dict(ticket) if ticket else None
    booking_module.transaction = _fake_transaction


# --------------------------------------------------------------------------- #
# Happy path — автономная read-only операция
# --------------------------------------------------------------------------- #
def run_happy_path(origin: str = "Москва", destination: str = "Стамбул") -> str:
    """Поиск рейсов: read-only, выполняется без эскалации на человека."""
    return flight_module.get_flights.func(origin=origin, destination=destination)


# --------------------------------------------------------------------------- #
# Рискованный путь — необратимая операция с эскалацией (HITL)
# --------------------------------------------------------------------------- #
class _CancelState(TypedDict, total=False):
    ticket_id: int
    result: str


def _cancel_node(state: _CancelState) -> dict:
    """Узел графа, выполняющий необратимую операцию отмены брони.

    Внутри ``cancel_reservation`` вызывается ``interrupt()`` — здесь граф и
    приостанавливается (эскалация на человека).
    """
    result = booking_module.cancel_reservation.func(ticket_id=state["ticket_id"])
    return {"result": result}


def build_cancel_graph(checkpointer=None):
    """Минимальный граф с checkpointer — обязателен для Human-in-the-loop."""
    builder = StateGraph(_CancelState)
    builder.add_node("cancel", _cancel_node)
    builder.add_edge(START, "cancel")
    builder.add_edge("cancel", END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


@dataclass
class RiskyOutcome:
    """Итог рискованного пути."""

    escalated: bool  # была ли эскалация на человека (interrupt)
    summary: str  # текст запроса подтверждения, показанный человеку
    final_text: str  # финальный ответ инструмента после решения


def run_risky_path(ticket_id: int, decision: str, graph=None) -> RiskyOutcome:
    """Прогнать отмену брони: эскалировать на человека и применить его решение.

    Args:
        ticket_id: билет для отмены.
        decision: ответ человека («да»/«нет»), передаётся через ``Command(resume=...)``.
        graph: заранее собранный граф (по умолчанию создаётся новый).
    """
    graph = graph or build_cancel_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = graph.invoke({"ticket_id": ticket_id}, config)

    interrupts = result.get("__interrupt__")
    if not interrupts:
        # Операция не потребовала подтверждения (например, билет не найден).
        return RiskyOutcome(escalated=False, summary="", final_text=result.get("result", ""))

    payload = interrupts[0].value
    summary = payload.get("summary", "") if isinstance(payload, dict) else str(payload)

    resumed = graph.invoke(Command(resume=decision), config)
    return RiskyOutcome(escalated=True, summary=summary, final_text=resumed.get("result", ""))


# --------------------------------------------------------------------------- #
# Демонстрация
# --------------------------------------------------------------------------- #
def main() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    install_in_memory_backends()

    print("=" * 72)
    print("PoC 1/2 — HAPPY PATH: поиск рейсов (read-only, без человека)")
    print("=" * 72)
    print(run_happy_path())

    print()
    print("=" * 72)
    print("PoC 2/2 — РИСКОВАННЫЙ ПУТЬ: отмена брони (эскалация на человека)")
    print("=" * 72)

    print("\n[Сценарий A] человек отклоняет операцию → «нет»")
    outcome_no = run_risky_path(ticket_id=42, decision="нет")
    print(f"  эскалация на человека: {outcome_no.escalated}")
    print(f"  показано человеку    : {outcome_no.summary}")
    print(f"  результат            : {outcome_no.final_text}")

    print("\n[Сценарий B] человек подтверждает операцию → «да»")
    outcome_yes = run_risky_path(ticket_id=42, decision="да")
    print(f"  эскалация на человека: {outcome_yes.escalated}")
    print(f"  результат            : {outcome_yes.final_text}")


if __name__ == "__main__":
    main()
