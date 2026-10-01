import json
import logging
import os
import time

import psycopg2
from confluent_kafka import Consumer
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
SCORING_TOPIC = os.getenv("KAFKA_SCORING_TOPIC", "scores")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://app:app@postgres:5432/fraud")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))

INSERT_SQL = """
    INSERT INTO scores (transaction_id, score, fraud_flag)
    VALUES %s
    ON CONFLICT (transaction_id) DO NOTHING
"""


def connect_db(retries=30, delay=2):
    """Connect to Postgres, retrying while it is still starting up."""
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg2.connect(POSTGRES_DSN)
            logger.info("Connected to Postgres")
            return conn
        except psycopg2.OperationalError as e:
            logger.warning(f"Postgres not ready (attempt {attempt}/{retries}): {e}")
            time.sleep(delay)
    raise RuntimeError("Could not connect to Postgres")


def parse_message(raw):
    """Return list of (transaction_id, score, fraud_flag).

    The producer writes a single JSON object; a list of objects is accepted as well."""
    payload = json.loads(raw.decode('utf-8'))
    records = payload if isinstance(payload, list) else [payload]
    return [(r['transaction_id'], float(r['score']), int(r['fraud_flag'])) for r in records]


def main():
    conn = connect_db()
    consumer = Consumer({
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'group.id': 'score-writer',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': False,
    })
    consumer.subscribe([SCORING_TOPIC])
    logger.info(f"Score writer started, reading topic '{SCORING_TOPIC}'")

    while True:
        msgs = consumer.consume(num_messages=BATCH_SIZE, timeout=1.0)
        rows = []
        for msg in msgs:
            if msg.error():
                logger.error(f"Kafka error: {msg.error()}")
                continue
            try:
                rows.extend(parse_message(msg.value()))
            except Exception as e:
                logger.error(f"Invalid message skipped: {e}")
        if not msgs:
            continue

        if rows:
            try:
                with conn, conn.cursor() as cur:
                    execute_values(cur, INSERT_SQL, rows)
            except psycopg2.Error as e:
                # Don't commit offsets: the batch is redelivered after reconnect
                logger.error(f"DB insert failed, reconnecting: {e}")
                conn.close()
                conn = connect_db()
                consumer.close()
                consumer = Consumer({
                    'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
                    'group.id': 'score-writer',
                    'auto.offset.reset': 'earliest',
                    'enable.auto.commit': False,
                })
                consumer.subscribe([SCORING_TOPIC])
                continue
            logger.info(f"Saved {len(rows)} scores to Postgres")
        consumer.commit(asynchronous=False)


if __name__ == "__main__":
    main()
