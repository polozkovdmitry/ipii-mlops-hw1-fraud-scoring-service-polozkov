"""Model history in Postgres: one row per trained model, exactly one of them active."""
import contextlib
import os

import psycopg2
from psycopg2.extras import Json, RealDictCursor

POSTGRES_DSN = os.getenv('POSTGRES_DSN', 'postgresql://app:app@postgres:5432/fraud')

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id          SERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    name        TEXT NOT NULL,
    params      JSONB NOT NULL,
    metrics     JSONB NOT NULL,
    threshold   DOUBLE PRECISION NOT NULL,
    path        TEXT NOT NULL,
    is_default  BOOLEAN NOT NULL DEFAULT FALSE,
    is_active   BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE UNIQUE INDEX IF NOT EXISTS models_one_active ON models (is_active) WHERE is_active;
"""


@contextlib.contextmanager
def _cursor():
    conn = psycopg2.connect(POSTGRES_DSN)
    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:  # `with conn` commits / rolls back
            yield cur
    finally:
        conn.close()


def init_db():
    with _cursor() as cur:
        cur.execute(SCHEMA)


def add_model(name, params, metrics, threshold, path, is_default=False):
    with _cursor() as cur:
        cur.execute(
            'INSERT INTO models (name, params, metrics, threshold, path, is_default) '
            'VALUES (%s, %s, %s, %s, %s, %s) RETURNING id',
            (name, Json(params), Json(metrics), threshold, path, is_default))
        return cur.fetchone()['id']


def activate(model_id):
    """Make the model active; returns its row, or None if there is no such model."""
    with _cursor() as cur:
        cur.execute('SELECT 1 FROM models WHERE id = %s', (model_id,))
        if cur.fetchone() is None:
            return None
        cur.execute('UPDATE models SET is_active = FALSE WHERE is_active')
        cur.execute('UPDATE models SET is_active = TRUE WHERE id = %s RETURNING *', (model_id,))
        return cur.fetchone()


def count():
    with _cursor() as cur:
        cur.execute('SELECT count(*) AS n FROM models')
        return cur.fetchone()['n']


def get_active():
    with _cursor() as cur:
        cur.execute('SELECT * FROM models WHERE is_active')
        return cur.fetchone()


def get_default():
    with _cursor() as cur:
        cur.execute('SELECT * FROM models WHERE is_default ORDER BY id LIMIT 1')
        return cur.fetchone()


def get_model(model_id):
    with _cursor() as cur:
        cur.execute('SELECT * FROM models WHERE id = %s', (model_id,))
        return cur.fetchone()


def list_models():
    with _cursor() as cur:
        cur.execute('SELECT * FROM models ORDER BY id DESC')
        return cur.fetchall()


def set_threshold(model_id, threshold):
    with _cursor() as cur:
        cur.execute('UPDATE models SET threshold = %s WHERE id = %s RETURNING id', (threshold, model_id))
        return cur.fetchone() is not None
