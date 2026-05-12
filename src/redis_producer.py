"""
Redis Streams Producer — streams happiness features record by record.
Functionally equivalent to the Kafka producer but uses Redis XADD.

Usage:
    python src/redis_producer.py
"""
import csv
import json
import time
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

import redis

REDIS_URL  = os.getenv("REDIS_URL")
STREAM_KEY = os.getenv("REDIS_STREAM_KEY", "happiness-stream")
DATA_FILE  = Path(__file__).parent.parent / "outputs" / "test_data.csv"
MAX_LEN    = 1000   # keep stream bounded


def run():
    if not DATA_FILE.exists():
        print(f"[Producer] {DATA_FILE} not found. Run 02_model_training.py first.")
        sys.exit(1)

    r = redis.from_url(REDIS_URL, decode_responses=True)
    r.ping()
    print(f"[Producer] Connected to Redis. Streaming to '{STREAM_KEY}'...")

    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Redis XADD accepts a flat dict of strings
            r.xadd(STREAM_KEY, row, maxlen=MAX_LEN, approximate=True)
            time.sleep(0.05)
            if (i + 1) % 50 == 0:
                print(f"[Producer] {i + 1} records sent...")

    print(f"[Producer] Done. Stream length: {r.xlen(STREAM_KEY)}")


if __name__ == "__main__":
    run()
