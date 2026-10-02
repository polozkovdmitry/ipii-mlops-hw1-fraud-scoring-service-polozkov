"""Send rows of a test.csv-format file to the Kafka topic `transactions` (same envelope as the UI).

Usage (from the host, Kafka is exposed on localhost:9095):
    pip install kafka-python pandas
    python scripts/send_sample.py misc/test_382.csv [--bootstrap localhost:9095] [--topic transactions]
"""
import argparse
import json
import uuid

import pandas as pd
from kafka import KafkaProducer

parser = argparse.ArgumentParser()
parser.add_argument('csv')
parser.add_argument('--bootstrap', default='localhost:9095')
parser.add_argument('--topic', default='transactions')
args = parser.parse_args()

df = pd.read_csv(args.csv)
producer = KafkaProducer(bootstrap_servers=args.bootstrap,
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))
for row in df.to_dict(orient='records'):
    producer.send(args.topic, {'transaction_id': str(uuid.uuid4()), 'data': row})
producer.flush()
print(f'Sent {len(df)} transactions to {args.topic}')
