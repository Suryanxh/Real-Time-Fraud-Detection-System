"""
PostgreSQL connection pool and query helpers for the fraud detection POC.

Reads connection details from environment variables (see .env.example).
Uses a small connection pool since FastAPI can serve concurrent requests --
a single shared connection would serialize every /score call on DB I/O.
"""

import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://fraud_user:fraud_pass@localhost:5432/fraud_db"
)

# Lazy, not created at import time. A hard-crash on import would take down
# the whole API (model scoring included) if Postgres just happens to be
# down or not-yet-started when the API boots -- exactly the kind of single
# point of failure the fail-open design elsewhere (model errors, DB write
# errors) is meant to avoid. Every caller already handles get_cursor()
# raising (see api.py's try/except around db.* calls), so raising here on
# a connection failure is safe -- it degrades that one call, not the process.
_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=DATABASE_URL)
    return _pool


@contextmanager
def get_cursor():
    """
    Yields a dict-cursor within a transaction; commits on success,
    rolls back and re-raises on any exception, always returns the
    connection to the pool. Raises if the DB is unreachable -- callers
    (api.py) catch this and fail open rather than crashing the request.
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------
def insert_model_metrics(precision: float, recall: float, f1: float, threshold: float,
                          review_threshold: float | None = None, auc_roc: float | None = None):
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO model_metrics (precision, recall, f1, threshold, review_threshold, auc_roc)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (precision, recall, f1, threshold, review_threshold, auc_roc),
        )

