"""Слой доступа к PostgreSQL на psycopg 3.

Предоставляет:
- get_connection() — открыть соединение с dict-строками;
- fetch_all / fetch_one — удобные обёртки для read-запросов;
- transaction() — контекстный менеджер для атомарных write-операций.
"""

from contextlib import contextmanager
from typing import Any, Iterator, Optional, Sequence

import psycopg
from psycopg.rows import dict_row

from app.config import settings
from app.observability.metrics import observe_db, record_error


def get_connection() -> psycopg.Connection:
    """Открыть новое соединение. Строки возвращаются как dict."""
    return psycopg.connect(settings.pg_conninfo, row_factory=dict_row)


def fetch_all(query: str, params: Optional[Sequence[Any]] = None) -> list[dict]:
    """Выполнить SELECT и вернуть все строки."""
    try:
        with observe_db("fetch_all"):
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    return cur.fetchall()
    except Exception:
        record_error("db")
        raise


def fetch_one(query: str, params: Optional[Sequence[Any]] = None) -> Optional[dict]:
    """Выполнить SELECT и вернуть первую строку (или None)."""
    try:
        with observe_db("fetch_one"):
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    return cur.fetchone()
    except Exception:
        record_error("db")
        raise


@contextmanager
def transaction() -> Iterator[psycopg.Cursor]:
    """Атомарная транзакция.

    Внутри блока доступен курсор. При выходе без исключений изменения
    фиксируются (COMMIT), при исключении — откатываются (ROLLBACK).

        with transaction() as cur:
            cur.execute(...)
    """
    conn = get_connection()
    try:
        with observe_db("transaction"):
            with conn.transaction():
                with conn.cursor() as cur:
                    yield cur
    except Exception:
        record_error("db")
        raise
    finally:
        conn.close()
