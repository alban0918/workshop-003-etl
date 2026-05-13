# Workshop 003 — ETL Process with Apache Kafka + Machine Learning Juan Jose Alban

**Course:** ETL — Data Engineering and Artificial Intelligence  
**Dataset:** World Happiness Report (2015–2019)  
**Goal:** Predict the Happiness Score using a regression model trained on socioeconomic features, streamed via Apache Kafka, with predictions stored in Supabase (PostgreSQL).

---

## Project Structure

```
workshop-003/
├── data/               # Raw CSV files (2015-2019)
├── notebooks/
│   ├── 01_eda.py       # EDA: loading, cleaning, visualizations
│   └── 02_model_training.py  # Model training + evaluation
├── src/
│   ├── preprocess.py   # Data loading, normalization, feature engineering
│   ├── db.py           # Supabase (PostgreSQL) helper
│   ├── kafka_producer.py  # Kafka producer
│   └── kafka_consumer.py  # Kafka consumer + prediction + DB insert
├── models/
│   └── happiness_model.pkl  # Serialized best model
├── outputs/            # Figures, metrics, combined CSV, test split
├── reports/
│   └── summary_report.md
├── requirements.txt
└── .env.example        # Environment variables template
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/workshop-003-etl.git
cd workshop-003-etl
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `SUPABASE_DB_URL` | PostgreSQL connection string from Supabase |
| `KAFKA_BOOTSTRAP_SERVERS` | Confluent Cloud bootstrap URL |
| `KAFKA_API_KEY` | Confluent Cloud API key |
| `KAFKA_API_SECRET` | Confluent Cloud API secret |
| `KAFKA_TOPIC` | Kafka topic name (default: `happiness-predictions`) |

#### Getting Supabase credentials
1. Go to [supabase.com](https://supabase.com) → your project
2. Settings → Database → Connection string → URI mode
3. Copy the full URI

#### Getting Confluent Cloud credentials
1. Sign up at [confluent.io](https://confluent.io) (free tier available)
2. Create a Basic cluster
3. Go to API Keys → Create key (Global access)
4. Copy the bootstrap server, API key and secret

---

## Running the Pipeline

### Step 1 — EDA
```bash
python notebooks/01_eda.py
```
Outputs 6 figures to `outputs/`.

### Step 2 — Model Training
```bash
python notebooks/02_model_training.py
```
Trains Linear Regression and Gradient Boosting, saves the best model to `models/happiness_model.pkl` and the test split to `outputs/test_data.csv`.

### Step 3 — Start Kafka Consumer (in one terminal)
```bash
python src/kafka_consumer.py
```
The consumer loads the model, creates the DB table, and waits for messages.

### Step 4 — Start Kafka Producer (in another terminal)
```bash
python src/kafka_producer.py
```
Streams all rows from `outputs/test_data.csv` to Kafka one by one.

---

## Model Results

| Model | MAE | RMSE | R² | CV R² |
|---|---|---|---|---|
| Linear Regression | 0.4486 | 0.5843 | 0.7266 | 0.7394 |
| **Gradient Boosting** | **0.4168** | **0.5332** | **0.7723** | **0.7576** |

Best model: **Gradient Boosting Regressor** (R² = 0.77)
