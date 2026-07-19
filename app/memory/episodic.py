"""Эпизодическая память в Qdrant."""

import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from app.config import settings
from app.memory.privacy import sanitize_episode_text
from app.rag import _get_embeddings


@lru_cache(maxsize=1)
def _get_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def _collection() -> str:
    return settings.memory_episodic_collection


def ensure_collection() -> None:
    """Создать коллекцию эпизодов, если её ещё нет."""
    client = _get_client()
    name = _collection()
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )


def _ttl_cutoff_ts() -> float:
    days = settings.memory_episodic_ttl_days
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return cutoff.timestamp()


def purge_expired_episodes(user_id: str) -> None:
    """Удалить эпизоды пользователя старше TTL."""
    if not user_id:
        return
    client = _get_client()
    name = _collection()
    if not client.collection_exists(name):
        return
    cutoff = _ttl_cutoff_ts()
    client.delete(
        collection_name=name,
        points_selector=Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="timestamp", range=Range(lt=cutoff)),
            ]
        ),
    )


def store_episode(
    *,
    user_id: str,
    session_id: str,
    text: str,
    metadata: Optional[dict] = None,
) -> bool:
    """Сохранить эпизод. Возвращает False, если текст пуст после санитизации."""
    clean = sanitize_episode_text(text)
    if not user_id or not clean:
        return False
    ensure_collection()
    purge_expired_episodes(user_id)

    embeddings = _get_embeddings()
    vector = embeddings.embed_query(clean)
    now = datetime.now(timezone.utc).timestamp()
    payload = {
        "user_id": user_id,
        "session_id": session_id or "",
        "text": clean,
        "timestamp": now,
        **(metadata or {}),
    }
    _get_client().upsert(
        collection_name=_collection(),
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
        ],
    )
    return True


def retrieve_episodes(user_id: str, query: str, top_k: Optional[int] = None) -> list[str]:
    """Семантический поиск эпизодов пользователя."""
    if not user_id or not query:
        return []
    client = _get_client()
    name = _collection()
    if not client.collection_exists(name):
        return []

    top_k = top_k or settings.memory_top_k
    embeddings = _get_embeddings()
    vector = embeddings.embed_query(query)
    cutoff = _ttl_cutoff_ts()

    hits = client.query_points(
        collection_name=name,
        query=vector,
        query_filter=Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="timestamp", range=Range(gte=cutoff)),
            ]
        ),
        limit=top_k,
        with_payload=True,
    ).points

    texts: list[str] = []
    for hit in hits:
        text = (hit.payload or {}).get("text", "").strip()
        if text:
            texts.append(text)
    return texts


def format_episodes_for_prompt(episodes: list[str]) -> str:
    if not episodes:
        return ""
    lines = [f"- {t}" for t in episodes]
    return "Релевантные прошлые обращения:\n" + "\n".join(lines)
