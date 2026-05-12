# Summary Report — Workshop 003: ETL with Apache Kafka + ML

## 1. Dataset Description

The dataset consists of five CSV files from the **World Happiness Report**, one per year (2015–2019), covering happiness scores and socioeconomic indicators for ~150 countries each year. After combining and normalizing, the unified dataset contains **782 rows** and 9 columns.

| Column | Description |
|---|---|
| `country` | Country name |
| `year` | Survey year (2015–2019) |
| `happiness_score` | Target variable (0–10 scale) |
| `gdp_per_capita` | Economic output per person (normalized) |
| `social_support` | Perceived social network strength |
| `life_expectancy` | Healthy life expectancy (normalized) |
| `freedom` | Freedom to make life choices |
| `generosity` | Charitable donations index |
| `corruption` | Perceptions of government corruption |

**Key challenge**: The 5 files used inconsistent column names across years (e.g., `Family` in 2015–2016, `Social support` in 2018–2019). The 2017 file used dot notation (`Happiness.Score`). A unified mapping table resolved all variants.

---

## 2. Key Findings from EDA

- **Score stability**: Mean happiness score remained stable at ~5.37–5.40 across all years. No major global trend up or down.
- **GDP is the strongest predictor** (correlation = 0.79 with happiness score), followed by social support (0.74) and life expectancy (0.72).
- **Corruption and generosity** are weaker predictors (correlations ~0.40 and ~0.15 respectively).
- **No missing values** after loading: all columns had complete data for all 782 rows across the five yearly files.
- **Top countries**: Finland, Denmark, Norway, Iceland, and Netherlands consistently rank highest (2019). Bottom: Afghanistan, Central African Republic, South Sudan.
- **Distribution**: Happiness scores follow a roughly normal distribution with a slight left skew (range: ~2.9–7.8).

---

## 3. Feature Selection

Selected features based on:
1. **Correlation analysis**: All 6 features have Pearson correlation > 0.10 with the target.
2. **Domain relevance**: GDP, social support, and life expectancy are the three pillars of the UN Human Development Index.
3. **Data availability**: All features are present in all 5 years (after column normalization).

Features excluded: `Happiness Rank` (redundant — derived from the score), `Standard Error` / `Whisker` columns (metadata, not predictors), `Region` (categorical, not available in all years).

---

## 4. Model Training

**Split**: 70% training (547 rows) / 30% test (235 rows), random_state=42.

Two models were trained:

| Model | MAE | RMSE | R² | CV R² (5-fold) |
|---|---|---|---|---|
| Linear Regression | 0.4486 | 0.5843 | 0.7266 | 0.7394 |
| **Gradient Boosting** | **0.4168** | **0.5332** | **0.7723** | **0.7576** |

**Best model: Gradient Boosting Regressor** (`n_estimators=200`, `learning_rate=0.05`, `max_depth=4`).

The model explains **77.2% of the variance** in happiness scores with a mean absolute error of **0.42 points** on a 0–10 scale — a strong result given the limited feature set.

**Feature importance** (Gradient Boosting):
1. GDP per capita (~38%)
2. Social support (~26%)
3. Life expectancy (~20%)
4. Freedom (~10%)
5. Corruption (~4%)
6. Generosity (~2%)

---

## 5. Streaming Process (Kafka)

**Architecture**:
```
test_data.csv  →  Kafka Producer  →  Confluent Cloud Topic  →  Kafka Consumer  →  Supabase DB
```

**Producer** (`src/kafka_producer.py`):
- Reads `outputs/test_data.csv` (235 rows — the test split)
- Serializes each row as JSON
- Produces one message every 50ms to the `happiness-predictions` topic
- Uses SASL_SSL authentication with Confluent Cloud

**Consumer** (`src/kafka_consumer.py`):
- Subscribes to `happiness-predictions`
- For each message: deserializes JSON → extracts 6 features → calls `model.predict()`
- Accumulates records in batches of 10 → bulk inserts into Supabase
- Stores: input features + actual happiness score + predicted happiness score

**Database schema** (`happiness_predictions` table):
```sql
id, country, year, gdp_per_capita, social_support, life_expectancy,
freedom, generosity, corruption, actual_score, predicted_score, created_at
```

---

## 6. Evaluation Metrics

| Metric | Value |
|---|---|
| R² (test set) | 0.7723 |
| MAE (test set) | 0.4168 |
| RMSE (test set) | 0.5332 |
| CV R² (5-fold) | 0.7576 |

The model generalizes well — the cross-validation R² (0.758) is close to the test R² (0.772), indicating no overfitting. An MAE of 0.42 on a happiness scale of ~2.9–7.8 is acceptable for real-world socioeconomic prediction.

---

## 7. Challenges and Assumptions

- **Inconsistent schemas**: Solved via a comprehensive column mapping dictionary in `preprocess.py`.
- **No missing values**: All datasets were complete, so no imputation was needed. The code includes median imputation as a safeguard for future data.
- **Kafka requires cloud service**: No local Java/Docker environment was available. Confluent Cloud free tier was used with SASL_SSL authentication.
- **Assumption**: The happiness "score" columns across years use comparable scales (0–10), which is confirmed by the UN methodology.
