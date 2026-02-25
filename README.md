# Real-Time Fraud Detection Demo on ClickHouse

A pipeline that simulates, detects, and visualizes financial fraud in real time using ClickHouse Cloud, an IsolationForest executable UDF, and Streamlit.

## Stack

- **ClickHouse Cloud** — storage, ingestion, and MV-based ML scoring
- **scikit-learn (IsolationForest)** — ML anomaly detection
- **Streamlit** — live dashboard

## Architecture

```
[01] Data Generator
        │
        ▼
fraud.transactions
        │
        ├──► mv_user_stats ──► fraud.user_stats (per-user baselines)
        │
        └──► mv_ml_alerts  ──► fraud.alerts_live
                  │
                  └── calls fraud_score() UDF (03_udf_scorer.py)
                            │
                            └── IsolationForest model (fraud_model.pkl)
                                      │
                                   [02] trained by 02_train_model.py

[04] Streamlit Dashboard  ◄──── auto-refresh every 15s
```

## Components

### `DDL.sql` — Schema

Creates all tables and Materialized Views:

| Object | Type | Purpose |
|---|---|---|
| `fraud.transactions` | Table (MergeTree) | Raw transaction stream |
| `fraud.user_stats` | Table (AggregatingMergeTree) | Per-user average amount baselines |
| `fraud.mv_user_stats` | Materialized View | Incrementally maintains `user_stats` on every insert |
| `fraud.alerts_live` | Table (MergeTree) | ML anomaly alerts written by the scoring MV |
| `fraud.mv_ml_alerts` | Materialized View | Calls the `fraud_score` UDF on every insert, writes anomalies to `alerts_live` |

Apply once to ClickHouse Cloud before running anything else.

---

### `01_ingest.py` — Data Generator

Runs in a continuous loop, inserting ~60,000 synthetic transactions per second into `fraud.transactions` using a server-side `INSERT INTO ... SELECT FROM numbers(60000)`.

Each row contains:

| Field | Description |
|---|---|
| `ts` | Timestamp (now minus up to 60s random offset) |
| `user_id` | Random user ID (1–100,000) |
| `amount` | 98% normal ($5–$250), 2% suspicious ($1,500–$5,000) |
| `country` | DE, ES, FR, IT, or US |
| `channel` | agency, call_center, mobile, or web |

---

### `02_train_model.py` — Model Trainer

Run once (after some data has been ingested) to train an `IsolationForest` on recent transactions and serialize it to `fraud_model.pkl`.

- Pulls the last 30 minutes of data (up to 10,000 rows) from ClickHouse
- Engineers 6 features: `amount`, `log(amount)`, per-user mean, user deviation ratio, country code, channel code
- Trains `IsolationForest` (200 estimators, 2% contamination)
- Saves `fraud_model.pkl` — loaded by `03_udf_scorer.py` at UDF startup

Re-run any time you want to refresh the model.

---

### `03_udf_scorer.py` — Executable UDF (IsolationForest Scorer)

The executable UDF that ClickHouse calls to score transactions. It is **not run directly** — it is deployed to ClickHouse's `user_scripts` directory and registered as a function named `fraud_score`.

- Reads tab-separated feature rows from stdin using ClickHouse's chunk-header protocol
- Scores the entire chunk in one vectorized `decision_function` call
- Outputs one `Float64` anomaly score per row to stdout

`mv_ml_alerts` calls `fraud_score()` on every insert into `fraud.transactions`, pre-filtered to `amount > 200`. Rows where `score < 0` are written to `fraud.alerts_live` with a severity label:

| Severity | Score threshold |
|---|---|
| `CRITICAL` | score ≤ -0.015 |
| `HIGH` | score ≤ -0.010 |
| `MEDIUM` | score ≤ -0.005 |
| `LOW` | score < 0 |

---

### `04_dashboard.py` — Streamlit Dashboard

Live dashboard (auto-refreshes every 15 seconds):

- **KPIs**: total transactions, transactions/5 min, ML alerts/5 min, fraud rate
- **Charts**: transactions per minute, ML alerts per minute, alerts by country, alert severity breakdown
- **Alert table**: last 30 ML alerts from `fraud.alerts_live`, color-coded by severity

---

## Running the Demo

### Prerequisites

1. Apply the schema to ClickHouse Cloud:
   ```sql
   -- run DDL.sql in the ClickHouse Cloud SQL console
   ```

2. Copy `03_udf_scorer.py` and `fraud_model.pkl` to the ClickHouse `user_scripts` directory and register the UDF:
   ```sql
   CREATE FUNCTION fraud_score AS executable ...
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Run each script in a separate terminal

```bash
# 1. Start ingesting transactions
python 01_ingest.py

# 2. Train the model (once enough data is present — at least 200 rows)
python 02_train_model.py

# 3. Launch the dashboard
streamlit run 04_dashboard.py
```

> `03_udf_scorer.py` is deployed to ClickHouse as an executable UDF — ClickHouse calls it automatically on each insert via `mv_ml_alerts`. It does not need to be run manually.

# Contributors
[Oussama Chakri](https://github.com/Ochakri-CHDB) is the original creator of this demo
[Caio Ishizaka Costa](https://github.com/ch-caioishizaka) created the UDF and feature store for real time scoring