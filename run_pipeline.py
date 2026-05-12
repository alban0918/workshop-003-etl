"""
Runs producer and consumer concurrently using threads.
Usage: python run_pipeline.py
"""
import threading
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from src.redis_producer import run as producer_run
from src.redis_consumer import run as consumer_run

if __name__ == "__main__":
    consumer_thread = threading.Thread(target=consumer_run, daemon=False)
    producer_thread = threading.Thread(target=producer_run, daemon=False)

    consumer_thread.start()

    import time; time.sleep(1)   # give consumer time to initialize

    producer_thread.start()

    producer_thread.join()
    consumer_thread.join(timeout=20)
    print("\nPipeline complete.")
