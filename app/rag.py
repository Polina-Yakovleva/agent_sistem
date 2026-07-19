"""Retrieval-конвейер поверх Qdrant + BAAI/bge-m3.

Используется инструментами Compliance Agent. Логика:
1. аргументы инструмента превращаются в Qdrant payload-фильтр (doc_type, entity);
2. запрос эмбеддится моделью bge-m3 (та же модель, что и при индексации);
3. выполняется pre-filtered векторный поиск (HNSW) в коллекции;
4. найденные чанки упаковываются в единую строку контекста для LLM.
"""

from functools import lru_cache
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
)

from app.config import settings


@lru_cache(maxsize=1)
def _get_embeddings():
    """Ленивая инициализация эмбеддинг-модели (загружается один раз)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _get_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def _build_filter(doc_types: list[str], entity: Optional[str]) -> Filter:
    """Собрать Qdrant-фильтр из аргументов инструмента.

    doc_type — поле-ключ (visa / baggage / animals), entity — источник или авиакомпания.
    Оба поля проиндексированы как keyword, поэтому фильтрация выполняется
    ДО векторного поиска (pre-filter).
    """
    must: list[FieldCondition] = []
    if len(doc_types) == 1:
        must.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_types[0])))
    elif doc_types:
        must.append(FieldCondition(key="doc_type", match=MatchAny(any=doc_types)))
    if entity:
        must.append(FieldCondition(key="entity", match=MatchValue(value=entity)))
    return Filter(must=must)


def _pack_context(hits) -> str:
    """Упаковать результаты retrieval в строку контекста для LLM."""
    blocks: list[str] = []
    for i, hit in enumerate(hits, 1):
        payload = hit.payload or {}
        text = (payload.get("text") or "").strip()
        meta_bits = []
        if payload.get("entity"):
            meta_bits.append(payload["entity"])
        if payload.get("doc_type"):
            meta_bits.append(payload["doc_type"])
        if payload.get("updated_at"):
            meta_bits.append(f"обновлено {payload['updated_at']}")
        meta = ", ".join(meta_bits)

        block = f"[Фрагмент {i}] (релевантность {hit.score:.3f}"
        block += f"; {meta})" if meta else ")"
        block += f"\n{text}"
        if payload.get("source_url"):
            block += f"\nИсточник: {payload['source_url']}"
        blocks.append(block)
    return "\n\n".join(blocks)


def retrieve(
    query: str,
    doc_types: list[str],
    entity: Optional[str] = None,
    top_k: Optional[int] = None,
) -> str:
    """Выполнить pre-filtered поиск и вернуть строку контекста.

    При отсутствии результатов с фильтром по entity выполняется повторный поиск
    только по doc_type (с пометкой, что точное совпадение по сущности не найдено).
    """
    top_k = top_k or settings.rag_top_k
    embeddings = _get_embeddings()
    client = _get_client()

    query_vector = embeddings.embed_query(query)

    hits = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=_build_filter(doc_types, entity),
        limit=top_k,
        with_payload=True,
    ).points

    fallback_note = ""
    if not hits and entity:
        hits = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            query_filter=_build_filter(doc_types, None),
            limit=top_k,
            with_payload=True,
        ).points
        if hits:
            fallback_note = (
                f"Точных документов по сущности «{entity}» не найдено, "
                f"показаны общие сведения по теме.\n\n"
            )

    if not hits:
        types = " / ".join(doc_types)
        target = f" по сущности «{entity}»" if entity else ""
        return f"В базе знаний не найдено документов ({types}){target}."

    try:
        from app.observability.turn_collector import get_turn_collector

        col = get_turn_collector()
        if col:
            col.add_rag_chunks(len(hits))
    except Exception:
        pass

    return fallback_note + _pack_context(hits)
