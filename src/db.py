"""
Database helper — creates the predictions table in Supabase (PostgreSQL) and
provides an insert function used by the Kafka consumer.
"""
import os
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("SUPABASE_DB_URL")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS happiness_predictions (
    id               SERIAL PRIMARY KEY,
    country          TEXT,
    year             INTEGER,
    gdp_per_capita   FLOAT,
    social_support   FLOAT,
    life_expectancy  FLOAT,
    freedom          FLOAT,
    generosity       FLOAT,
    corruption       FLOAT,
    actual_score     FLOAT,
    predicted_score  FLOAT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
"""


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
        conn.commit()
    print("[DB] Table happiness_predictions ready.")


def insert_predictions(records: list[dict]):
    """records: list of dicts matching the table columns (except id/created_at)."""
    if not records:
        return
    cols = [
        "country", "year", "gdp_per_capita", "social_support",
        "life_expectancy", "freedom", "generosity", "corruption",
        "actual_score", "predicted_score",
    ]
    values = [tuple(r.get(c) for c in cols) for r in records]
    sql = f"INSERT INTO happiness_predictions ({', '.join(cols)}) VALUES %s"
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, values)
        conn.commit()
