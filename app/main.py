"""CLI-запуск мультиагентной системы.

Примеры:
    python -m app.main "Найди рейсы из Москвы в Стамбул"
    python -m app.main                    # интерактивная сессия (несколько реплик)

В интерактивном режиме один thread_id на сессию (STM). LTM — по AGENT_USER_ID.
Поддерживается Human-in-the-loop для бронирования.
"""

import os
import sys
import uuid

from app.config import settings
from app.llm import check_llm_reachable
from app.observability import flush_observability, init_observability
from app.service import AgentService, run_turn_blocking_hitl


def _print_answer(answer: str) -> None:
    print("\n=== Ответ ===")
    print(answer)


def run_session(initial_query: str | None = None) -> None:
    """Интерактивная сессия с STM (общий thread_id) и LTM (user_id)."""
    session_id = str(uuid.uuid4())
    user_id = os.environ.get("AGENT_USER_ID") or settings.agent_user_id

    print(f"Сессия: {session_id[:8]}… | Пользователь LTM: {user_id}")
    print("Команды: пустая строка или «выход» — завершить.\n")

    with AgentService() as service:
        query = initial_query
        while True:
            if not query:
                query = input("Ваш запрос: ").strip()
            if not query:
                continue
            if query.lower() in {"выход", "exit", "quit", "q"}:
                break
            result = run_turn_blocking_hitl(service, query, session_id, user_id)
            _print_answer(result.answer or "(пустой ответ)")
            query = None


def run_once(query: str) -> None:
    """Один запрос без мультитурного REPL (отдельный thread_id)."""
    session_id = str(uuid.uuid4())
    user_id = os.environ.get("AGENT_USER_ID") or settings.agent_user_id
    with AgentService() as service:
        result = run_turn_blocking_hitl(service, query, session_id, user_id)
        _print_answer(result.answer or "(пустой ответ)")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    init_observability()
    try:
        check_llm_reachable()
        args_query = " ".join(sys.argv[1:]).strip()
        if args_query:
            run_once(args_query)
        else:
            run_session()
    except ConnectionError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    finally:
        flush_observability()


if __name__ == "__main__":
    main()
