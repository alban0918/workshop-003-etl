"""
Kafka Producer — streams transformed happiness features record by record.
Reads the test split saved by model training (outputs/test_data.csv).

Usage:
    python src/kafka_producer.py
"""
import csv
import json
import time
import sys
import os
from pathlib import Path
from confluent_kafka import Producer

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()

BOOTSTRAP_SERVERS  = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
KAFKA_API_KEY      = os.getenv("KAFKA_API_KEY")
KAFKA_API_SECRET   = os.getenv("KAFKA_API_SECRET")
SASL_MECHANISM     = os.getenv("KAFKA_SASL_MECHANISM", "PLAIN")  # PLAIN or SCRAM-SHA-256
TOPIC              = os.getenv("KAFKA_TOPIC", "happiness-predictions")
DATA_FILE          = Path(__file__).parent.parent / "outputs" / "test_data.csv"


def build_config() -> dict:
    cfg = {"bootstrap.servers": BOOTSTRAP_SERVERS}
    if KAFKA_API_KEY and KAFKA_API_SECRET:
        cfg.update({
            "security.protocol": "SASL_SSL",
            "sasl.mechanisms": SASL_MECHANISM,
            "sasl.username": KAFKA_API_KEY,
            "sasl.password": KAFKA_API_SECRET,
        })
    return cfg


def delivery_report(err, msg):
    if err:
        print(f"[Producer] Delivery failed: {err}")
    else:
        print(f"[Producer] Sent offset={msg.offset()} key={msg.key()}")


def run():
    if not DATA_FILE.exists():
        print(f"[Producer] {DATA_FILE} not found. Run 02_model_training.py first.")
        sys.exit(1)

    producer = Producer(build_config())
    print(f"[Producer] Connected. Streaming to topic '{TOPIC}'...")

    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            payload = json.dumps(row).encode("utf-8")
            producer.produce(TOPIC, value=payload, callback=delivery_report)
            producer.poll(0)
            time.sleep(0.05)   # 50 ms between records
            if (i + 1) % 50 == 0:
                print(f"[Producer] {i + 1} records sent...")

    producer.flush()
    print("[Producer] All records sent. Done.")


if __name__ == "__main__":
    run()
