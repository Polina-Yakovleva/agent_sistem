"""Лёгкий вывод прогресса выполнения графа.

Сообщения пишутся в stderr (чтобы не смешиваться с итоговым ответом в stdout)
и дублируются в structlog (JSON для Loki) при включённой observability.
"""

import sys

from app.config import settings
from app.observability.logging import get_logger

# Человекочитаемые названия субагентов.
AGENT_TITLES = {
    "flight": "агент рейсов",
    "booking": "агент бронирования",
    "compliance": "агент регуляторной информации",
    "external": "внешний агент",
}


def progress(stage: str, message: str) -> None:
    """Вывести строку прогресса вида: [этап] сообщение."""
    print(f"[{stage}] {message}", file=sys.stderr, flush=True)
    if settings.observability_enabled:
        get_logger("agent.progress").info(
            "stage",
            stage=stage,
            message=message,
            message_length=len(message),
        )
