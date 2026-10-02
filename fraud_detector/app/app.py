import json
import logging
import os
import sys
import threading
import uuid

import pandas as pd
import uvicorn
from confluent_kafka import Consumer, Producer

sys.path.append(os.path.abspath('./src'))
sys.path.append(os.path.abspath('./app'))
import scorer
from api import app as api_app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TRANSACTIONS_TOPIC = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")
SCORING_TOPIC = os.getenv("KAFKA_SCORING_TOPIC", "scores")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
API_PORT = int(os.getenv("API_PORT", "8000"))


def parse_message(raw):
    """Return (transaction_id, row dict).

    Accepts the UI envelope {"transaction_id": ..., "data": {...}} as well as a bare
    test.csv row (an ID is generated if there is none)."""
    payload = json.loads(raw.decode('utf-8'))
    if 'data' in payload:
        return payload.get('transaction_id') or str(uuid.uuid4()), payload['data']
    row = {k: v for k, v in payload.items() if k != 'transaction_id'}
    return payload.get('transaction_id') or str(uuid.uuid4()), row


class ProcessingService:
    def __init__(self):
        self.consumer = Consumer({
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'group.id': 'ml-scorer',
            'auto.offset.reset': 'earliest'
        })
        self.consumer.subscribe([TRANSACTIONS_TOPIC])
        self.producer = Producer({'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS})

    def score(self, items):
        """items: list of (transaction_id, row dict) -> list of result dicts."""
        ids = [i for i, _ in items]
        df = pd.DataFrame([row for _, row in items])
        result = scorer.make_pred(df, "kafka_stream")
        result.insert(0, 'transaction_id', ids)
        return result.to_dict(orient='records')

    def score_safely(self, items):
        """Score a batch; if it fails, retry row by row so one bad message
        doesn't drop the whole batch."""
        try:
            return self.score(items)
        except Exception as e:
            logger.warning(f"Batch of {len(items)} failed ({e}), retrying row by row")
        results = []
        for item in items:
            try:
                results.extend(self.score([item]))
            except Exception as e:
                logger.error(f"Skipping transaction {item[0]}: {e}")
        return results

    def process_messages(self):
        while True:
            msgs = self.consumer.consume(num_messages=BATCH_SIZE, timeout=1.0)
            items = []
            for msg in msgs:
                if msg.error():
                    logger.error(f"Kafka error: {msg.error()}")
                    continue
                try:
                    items.append(parse_message(msg.value()))
                except Exception as e:
                    logger.error(f"Invalid message skipped: {e}")
            if not items:
                continue

            results = self.score_safely(items)
            for res in results:
                self.producer.produce(SCORING_TOPIC, value=json.dumps(res))
            self.producer.flush()
            logger.info(f"Scored {len(results)}/{len(items)} transactions")


if __name__ == "__main__":
    logger.info('Starting Kafka ML scoring service...')
    scorer.init()
    # The API (retrain / model switch / threshold) runs in its own thread, the Kafka loop stays in the main one
    threading.Thread(
        target=uvicorn.run, args=(api_app,), kwargs={'host': '0.0.0.0', 'port': API_PORT, 'log_level': 'warning'},
        daemon=True).start()
    service = ProcessingService()
    try:
        service.process_messages()
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
