"""Инструменты Compliance Agent (RAG).

Два инструмента, которые под капотом вызывают retrieval-конвейер (app.rag):
- check_visa_requirements — doc_type=visa, entity=«КД МИД»; страна в тексте запроса;
- get_carrier_policy      — doc_type=baggage/animals, entity=<авиакомпания>.

Аргументы инструмента превращаются в Qdrant-фильтр, а результат retrieval
упаковывается в строку контекста для LLM.
"""

from typing import Optional

from langchain_core.tools import tool

from app.rag import retrieve

# Значения payload-полей в базе знаний (как при индексации в RAG/output).
DOC_TYPE_VISA = "visa"
DOC_TYPE_BAGGAGE = "baggage"
DOC_TYPE_ANIMALS = "animals"
ENTITY_VISA_SOURCE = "КД МИД"

_BAGGAGE_TOPIC_ALIASES = frozenset({"багаж", "ручная кладь", "клади", "luggage", "baggage"})
_ANIMALS_TOPIC_ALIASES = frozenset({"животные", "питомцы", "pets", "animals"})

_AIRLINE_ALIASES: dict[str, str] = {
    "аэрофлот": "Aeroflot",
    "aeroflot": "Aeroflot",
    "победа": "Pobeda",
    "pobeda": "Pobeda",
    "s7": "S7",
    "уральские авиалинии": "Uralairlines",
    "ural airlines": "Uralairlines",
    "uralairlines": "Uralairlines",
}


def _normalize_airline(name: str) -> str:
    """Привести название авиакомпании к канонической форме индекса."""
    cleaned = name.strip()
    if not cleaned:
        return cleaned
    return _AIRLINE_ALIASES.get(cleaned.lower(), cleaned)


def _resolve_carrier_doc_types(topic: Optional[str]) -> list[str]:
    topic_norm = (topic or "").strip().lower()
    if topic_norm in _BAGGAGE_TOPIC_ALIASES:
        return [DOC_TYPE_BAGGAGE]
    if topic_norm in _ANIMALS_TOPIC_ALIASES:
        return [DOC_TYPE_ANIMALS]
    return [DOC_TYPE_BAGGAGE, DOC_TYPE_ANIMALS]


@tool
def check_visa_requirements(country: str, question: Optional[str] = None) -> str:
    """Проверить визовые требования для въезда в страну.

    Под капотом выполняет retrieval с payload-фильтром doc_type="visa" и
    entity="КД МИД". Страна назначения передаётся в текст семантического запроса,
    а не в фильтр по entity.

    Args:
        country: Страна назначения (например, "Турция").
        question: Уточняющий вопрос (например, "виза по прибытии", "срок
            безвизового пребывания"). Необязательно.

    Returns:
        Строка контекста с релевантными фрагментами визовых требований.
    """
    country = country.strip()
    if question and question.strip():
        query = question.strip()
    else:
        query = f"визовые требования для граждан России: {country}"
    return retrieve(
        query=query,
        doc_types=[DOC_TYPE_VISA],
        entity=ENTITY_VISA_SOURCE,
    )


@tool
def get_carrier_policy(
    airline: str,
    topic: Optional[str] = None,
    question: Optional[str] = None,
) -> str:
    """Получить правила перевозчика по багажу и перевозке животных.

    Под капотом выполняет retrieval с payload-фильтром doc_type=baggage/animals и
    entity=<нормализованная авиакомпания>, затем упаковывает найденные фрагменты
    в контекст для LLM.

    Args:
        airline: Авиакомпания (например, "Аэрофлот").
        topic: Тема правил — «багаж», «животные» и синонимы. Если не указана,
            ищется по обеим темам.
        question: Уточняющий вопрос (например, "норма ручной клади",
            "перевозка кошки в салоне"). Необязательно.

    Returns:
        Строка контекста с релевантными фрагментами правил перевозчика.
    """
    doc_types = _resolve_carrier_doc_types(topic)
    airline_entity = _normalize_airline(airline)

    if question and question.strip():
        query = question.strip()
    else:
        topic_norm = (topic or "").strip().lower()
        subject = topic_norm if topic_norm else "багаж и перевозка животных"
        query = f"правила перевозчика {airline_entity}: {subject}"

    return retrieve(query=query, doc_types=doc_types, entity=airline_entity)


COMPLIANCE_TOOLS = [check_visa_requirements, get_carrier_policy]
