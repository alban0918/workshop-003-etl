"""
Kafka Consumer — receives streamed happiness records, predicts the happiness score,
and stores input features + actual + predicted scores in Supabase.

Usage:
    python src/kafka_consumer.py
"""
import json
import pickle
import sys
import os
import numpy as np
from pathlib import Path
from confluent_kafka import Consumer, KafkaError

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

from src.db import init_db, insert_predictions

BOOTSTRAP_SERVERS  = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_API_KEY      = os.getenv("KAFKA_API_KEY")
KAFKA_API_SECRET   = os.getenv("KAFKA_API_SECRET")
SASL_MECHANISM     = os.getenv("KAFKA_SASL_MECHANISM", "PLAIN")
TOPIC              = os.getenv("KAFKA_TOPIC", "happiness-predictions")
GROUP_ID           = os.getenv("KAFKA_GROUP_ID", "happiness-consumer-group")
MODEL_PATH         = Path(__file__).parent.parent / "models" / "happiness_model.pkl"

BATCH_SIZE = 10


def load_model():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["features"]


def build_config() -> dict:
    cfg = {
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "group.id": GROUP_ID,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": True,
    }
    if KAFKA_API_KEY and KAFKA_API_SECRET:
        cfg.update({
            "security.protocol": "SASL_SSL",
            "sasl.mechanisms": SASL_MECHANISM,
            "sasl.username": KAFKA_API_KEY,
            "sasl.password": KAFKA_API_SECRET,
        })
    return cfg


def parse_record(data: dict, features: list[str]) -> tuple[np.ndarray, dict]:
    row_features = []
    for f in features:
        try:
            row_features.append(float(data.get(f, 0) or 0))
        except (ValueError, TypeError):
            row_features.append(0.0)
    return np.array(row_features).reshape(1, -1), {
        "country":        data.get("country"),
        "year":           int(data.get("year", 0)) if data.get("year") else None,
        "gdp_per_capita": float(data.get("gdp_per_capita", 0) or 0),
        "social_support": float(data.get("social_support", 0) or 0),
        "life_expectancy":float(data.get("life_expectancy", 0) or 0),
        "freedom":        float(data.get("freedom", 0) or 0),
        "generosity":     float(data.get("generosity", 0) or 0),
        "corruption":     float(data.get("corruption", 0) or 0),
        "actual_score":   float(data.get("happiness_score", 0) or 0),
    }


def run():
    print("[Consumer] Loading model...")
    model, features = load_model()
    print(f"[Consumer] Model loaded. Features: {features}")

    init_db()

    consumer = Consumer(build_config())
    consumer.subscribe([TOPIC])
    print(f"[Consumer] Subscribed to '{TOPIC}'. Waiting for messages...")

    batch = []
    total = 0
    try:
        while True:
            msg = consumer.poll(timeout=5.0)

            if msg is None:
                if batch:
                    insert_predictions(batch)
                    total += len(batch)
                    print(f"[Consumer] Flushed {len(batch)} records (total={total})")
                    batch = []
                print("[Consumer] No messages (5s timeout). Still listening...")
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    print(f"[Consumer] End of partition reached.")
                else:
                    print(f"[Consumer] Error: {msg.error()}")
                continue

            data = json.loads(msg.value().decode("utf-8"))
            X_row, record = parse_record(data, features)
            record["predicted_score"] = float(model.predict(X_row)[0])
            batch.append(record)

            if len(batch) >= BATCH_SIZE:
                insert_predictions(batch)
                total += len(batch)
                print(f"[Consumer] Inserted batch of {len(batch)} (total={total})")
                batch = []

    except KeyboardInterrupt:
        if batch:
            insert_predictions(batch)
            total += len(batch)
        print(f"\n[Consumer] Stopped. Total records stored: {total}")
    finally:
        consumer.close()


if __name__ == "__main__":
    run()
