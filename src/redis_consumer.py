"""
Redis Streams Consumer — reads happiness records, predicts score, stores in Supabase.
Functionally equivalent to the Kafka consumer but uses Redis XREAD with consumer groups.

Usage:
    python src/redis_consumer.py
"""
import json
import pickle
import sys
import os
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

import redis

from src.db import init_db, insert_predictions

REDIS_URL   = os.getenv("REDIS_URL")
STREAM_KEY  = os.getenv("REDIS_STREAM_KEY", "happiness-stream")
GROUP_NAME  = "happiness-consumers"
CONSUMER_ID = "consumer-1"
MODEL_PATH  = Path(__file__).parent.parent / "models" / "happiness_model.pkl"
BATCH_SIZE  = 10


def load_model():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["features"]


def parse_record(data: dict, features: list) -> tuple:
    row_features = []
    for f in features:
        try:
            row_features.append(float(data.get(f, 0) or 0))
        except (ValueError, TypeError):
            row_features.append(0.0)
    return np.array(row_features).reshape(1, -1), {
        "country":         data.get("country"),
        "year":            int(data.get("year", 0)) if data.get("year") else None,
        "gdp_per_capita":  float(data.get("gdp_per_capita", 0) or 0),
        "social_support":  float(data.get("social_support", 0) or 0),
        "life_expectancy": float(data.get("life_expectancy", 0) or 0),
        "freedom":         float(data.get("freedom", 0) or 0),
        "generosity":      float(data.get("generosity", 0) or 0),
        "corruption":      float(data.get("corruption", 0) or 0),
        "actual_score":    float(data.get("happiness_score", 0) or 0),
    }


def ensure_group(r: redis.Redis):
    try:
        r.xgroup_create(STREAM_KEY, GROUP_NAME, id="0", mkstream=True)
        print(f"[Consumer] Consumer group '{GROUP_NAME}' created.")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            print(f"[Consumer] Consumer group '{GROUP_NAME}' already exists.")
        else:
            raise


def run():
    print("[Consumer] Loading model...")
    model, features = load_model()
    print(f"[Consumer] Model ready. Features: {features}")

    init_db()

    r = redis.from_url(REDIS_URL, decode_responses=True)
    r.ping()
    ensure_group(r)
    print(f"[Consumer] Listening on stream '{STREAM_KEY}'...")

    batch = []
    total = 0
    empty_polls = 0

    try:
        while empty_polls < 6:   # stop after ~6 s of silence
            messages = r.xreadgroup(
                GROUP_NAME, CONSUMER_ID, {STREAM_KEY: ">"}, count=BATCH_SIZE, block=1000
            )

            if not messages:
                empty_polls += 1
                print(f"[Consumer] No new messages ({empty_polls}/6)...")
                continue

            empty_polls = 0
            _, entries = messages[0]

            for msg_id, data in entries:
                X_row, record = parse_record(data, features)
                record["predicted_score"] = float(model.predict(X_row)[0])
                batch.append(record)
                r.xack(STREAM_KEY, GROUP_NAME, msg_id)

            if len(batch) >= BATCH_SIZE:
                insert_predictions(batch)
                total += len(batch)
                print(f"[Consumer] Inserted {len(batch)} records (total={total})")
                batch = []

    except KeyboardInterrupt:
        print("\n[Consumer] Interrupted by user.")
    finally:
        if batch:
            insert_predictions(batch)
            total += len(batch)
            print(f"[Consumer] Final flush: {len(batch)} records (total={total})")
        print(f"[Consumer] Done. Total records stored in Supabase: {total}")


if __name__ == "__main__":
    run()
