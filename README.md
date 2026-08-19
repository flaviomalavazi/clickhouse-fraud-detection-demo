# Real-Time Fraud Detection Demo on ClickHouse

A pipeline that simulates, detects, and visualizes financial fraud in real time using ClickHouse Cloud, an IsolationForest executable UDF, and Streamlit.

## Stack

- **ClickHouse Cloud** — storage, ingestion, and MV-based ML scoring
- **scikit-learn (IsolationForest)** — ML anomaly detection
- **Streamlit** — live dashboard
- **uv** — Python environment and dependency management

## Architecture

```
[01] Data Generator
        │
        ▼
fraud.transactions
        │
        ├──► mv_user_stats ──► fraud.user_stats (per-user baselines)
        │
        └──► mv_score_transactions  ──► fraud.transactions_scored  (score < 0 = alert)
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
| `fraud.transactions_scored` | Table (MergeTree) | All scored transactions with their anomaly score — filter `score < 0` for alerts |
| `fraud.mv_score_transactions` | Materialized View | Calls the `fraud_score` UDF on every insert, writes all scored rows to `transactions_scored` |

Apply once to ClickHouse Cloud before running anything else.

---

### `01_ingest.py` — Data Generator

Runs in a continuous loop, inserting ~60,000 synthetic transactions per second into `fraud.transactions` using a server-side `INSERT INTO ... SELECT FROM numbers(60000)`.

Each row contains:

| Field | Description |
|---|---|
| `transaction_id` | Random UUID generated server-side via `generateUUIDv4()` |
| `ts` | Timestamp (now minus up to 60s random offset) |
| `user_id` | Random user ID (1–100,000) |
| `amount` | 98% normal ($5–$250), 2% suspicious ($1,500–$5,000) |
| `country` | DE, ES, FR, IT, or US |
| `channel` | agency, call_center, mobile, or web |

---

### `02_train_model.py` — Model Trainer

Run once (after some data has been ingested) to train an `IsolationForest` on recent transactions and package it as a deployable UDF bundle.

- Pulls the last 30 minutes of data (up to 10,000 rows) from ClickHouse
- Engineers 6 features: `amount`, `log(amount)`, per-user mean, user deviation ratio, country code, channel code
- Trains `IsolationForest` (200 estimators, 2% contamination)
- Writes a new timestamped bundle under `function/`:

```
function/20260818T231341Z/
├── fraud_model.pkl    the serialized IsolationForest
├── main.py            copy of 03_udf_scorer.py (the UDF entrypoint)
└── requirements.txt    pinned runtime deps for this exact model
```

Each run creates a new UTC-timestamped directory, so versions never overwrite each other and any past bundle can be redeployed as-is. The bundle is self-contained — nothing else from this repo needs to ship with it.

`requirements.txt` is generated per version and pins `scikit-learn`, `pandas` and `numpy` to the exact versions that trained the pickle. This matters: a scikit-learn pickle is only guaranteed to load under the scikit-learn version that wrote it, so a drifting UDF host will warn or fail to unpickle. It is deliberately minimal — `scipy`, `joblib` and `threadpoolctl` arrive as scikit-learn dependencies.

Because those pins have to be installable on the UDF host, **this project is pinned to Python 3.11** (`.python-version`, and `requires-python = ">=3.11,<3.12"`), matching the host. Training on a newer Python emits pins the host can't install — e.g. `numpy` 2.5 requires Python ≥ 3.12 — so the trainer prints a warning if the two ever diverge. If your UDF host runs a different Python, change `TARGET_PYTHON` in `02_train_model.py` and the project's Python to match.

Re-run any time you want to refresh the model.

---

### `03_udf_scorer.py` — Executable UDF (IsolationForest Scorer)

The executable UDF that ClickHouse calls to score transactions. It is **not run directly and not deployed from here** — `02_train_model.py` copies it into each versioned bundle as `main.py`, and that copy is what gets uploaded and registered as a function named `fraud_score`. See [Deploying the UDF to ClickHouse Cloud](#deploying-the-udf-to-clickhouse-cloud).

- Reads tab-separated feature rows from stdin using ClickHouse's chunk-header protocol
- Scores the entire chunk in one vectorized `decision_function` call
- Writes one score per row to stdout as a plain decimal string, e.g. `-0.02080221580521524` — consumed by ClickHouse as **`Float64`**, which is the return type you must declare when registering the function

Edit this file, not the copies under `function/`: every bundle gets a fresh copy at training time, so changes here reach the next bundle you build and past bundles stay frozen with the model they were built for.

`mv_score_transactions` calls `fraud_score()` on every insert into `fraud.transactions`. All scored rows are written to `fraud.transactions_scored`; the dashboard filters `score < 0` at query time to show alerts. Severity label is populated for anomalies only:

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
- **Alert table**: last 30 ML alerts from `fraud.transactions_scored WHERE score < 0`, color-coded by severity

---

## Deploying the UDF to ClickHouse Cloud

ClickHouse Cloud gives you no filesystem access to the server's `user_scripts` directory, so the UDF is uploaded as a **zip archive through the SQL console UI** rather than copied onto a host. Reference: [User-defined functions in the ClickHouse Cloud SQL console](https://clickhouse.com/docs/products/cloud/features/sql-console-features/user-defined-functions#ui-udfs).

The bundle layout that `02_train_model.py` produces is exactly what Cloud expects — `main.py` at the archive root, with `requirements.txt` beside it for dependency installation:

### 1. Train the model to build a bundle

```bash
uv run 02_train_model.py
```

This writes `function/<timestamp>/` containing `main.py`, `fraud_model.pkl` and `requirements.txt`. Use the newest timestamp; older ones remain deployable if you need to roll back.

### 2. Zip the bundle

`main.py` must sit at the **root** of the archive, not inside a subdirectory — so zip from *inside* the version directory, naming the three files explicitly:

```bash
cd function/20260818T231341Z
zip -q ../../fraud_score.zip main.py requirements.txt fraud_model.pkl
cd ../..
```

Two reasons to list the files rather than `zip -r .`: a recursive zip sweeps up anything else that happens to be sitting in the directory (including a previous archive), and writing the output to the repo root keeps the new zip from ending up inside itself.

Cloud **rejects archives containing symbolic links**, so build the zip from real files as above. Verify what you're about to upload:

```bash
unzip -l fraud_score.zip    # expect exactly main.py, requirements.txt, fraud_model.pkl at the root
```

### 3. Register it in the Cloud console

1. Open the organization menu → **User-defined functions**
2. Click **Set up a UDF**, and name it `fraud_score` — this name is what [`DDL.sql`](DDL.sql) calls, so it must match exactly
3. Choose function type **Executable pool** — the process is kept alive between queries, so the ~2.4 MB pickle is unpickled once at startup instead of on every invocation
4. Upload `fraud_score.zip` via **Browse File**
5. Add the six arguments **in this exact order** — the UDF receives them positionally as tab-separated fields, so a wrong order silently produces wrong scores rather than an error:

   | # | Argument | Type | Passed by `mv_score_transactions` as |
   |---|---|---|---|
   | 1 | `amount` | `Float32` | `t.amount` |
   | 2 | `log_amount` | `Float64` | `log(t.amount + 1.0)` |
   | 3 | `user_amount_mean` | `Float64` | `coalesce(us.avg_amount, toFloat64(t.amount))` |
   | 4 | `user_amount_ratio` | `Float64` | `t.amount / (user_amount_mean + 1.0)` |
   | 5 | `country_encoded` | `Int8` | `transform(t.country, ['DE','ES','FR','IT','US'], [0,1,2,3,4], -1)` |
   | 6 | `channel_encoded` | `Int8` | `transform(t.channel, ['agency','call_center','mobile','web'], [0,1,2,3], -1)` |

6. Set **`send_chunk_header` to `true`** in the function's settings — `main.py` requires it (see [Required setting: `send_chunk_header`](#required-setting-send_chunk_header))
7. Set the **return type** to **`Float64`** — `main.py` prints one raw `decision_function` score per row (negative = anomaly), and `mv_score_transactions` narrows it with `toFloat32(score)` before storing it in `fraud.transactions_scored.score`
8. Click **Create UDF** and wait for the deployment status to report success

### 4. Create the scoring materialized view

`fraud.mv_score_transactions` references `fraud_score()`, so it can only be created **after** the function exists. Re-run that part of [`DDL.sql`](DDL.sql) once the UDF is deployed. Verify with:

```sql
SELECT count() FROM system.functions WHERE name = 'fraud_score';   -- expect 1
SELECT count() FROM fraud.transactions_scored;                     -- grows once ingestion runs
```

Then confirm scoring behaves sanely end-to-end:

```bash
uv run tests.py
```

### Required setting: `send_chunk_header`

`main.py` speaks ClickHouse's **chunk-header protocol** — it reads a row count, then that many rows, and scores them in one vectorized `decision_function` call, roughly 100–1000× faster than scoring row by row.

That means **`send_chunk_header` must be set to `true`** in the UDF's settings in the Cloud UI. It is not optional: without it ClickHouse sends data rows with no count in front of them, `main.py` tries to parse the first row as a row count, and the function dies with

```
ValueError: invalid literal for int() with base 10: '120.0\t4.796\t150.0\t0.795\t2\t3'
```

If you see that in the UDF logs, this setting is the cause.

### Refreshing the model later

Re-run `02_train_model.py` to build a new timestamped bundle, zip it, and upload it to the same UDF. Nothing in the schema changes — the MV keeps calling `fraud_score()` with the same six arguments — so a model refresh is upload-only. Because each `requirements.txt` is pinned to the versions that trained *its* pickle, an old bundle stays reproducible even after this repo's dependencies move on.

---

## Running the Demo

### Prerequisites

1. Apply the schema to ClickHouse Cloud:
   ```sql
   -- run DDL.sql in the ClickHouse Cloud SQL console
   ```

2. Build the environment (step 3), ingest some data and train a model (`uv run 02_train_model.py`),
   then deploy the resulting `function/<timestamp>/` bundle as the `fraud_score` UDF — full
   walkthrough in [Deploying the UDF to ClickHouse Cloud](#deploying-the-udf-to-clickhouse-cloud).
   The scoring materialized view in `DDL.sql` can only be created after the function exists.

3. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the virtual environment:
   ```bash
   uv sync
   ```
   This creates `.venv/` with the exact versions from `uv.lock`, downloading the
   Python version in `.python-version` if it isn't already installed. Nothing else
   to set up — no manual `venv` creation, no `pip install`.

   > Prefer plain pip? `requirements.txt` is exported from `uv.lock` and still works:
   > `python -m venv .venv && .venv/bin/pip install -r requirements.txt`.
   > Regenerate it after changing dependencies with
   > `uv export --no-hashes --no-dev --no-emit-project -o requirements.txt`.

4. Copy `.env.example` to `.env` and fill in your ClickHouse Cloud credentials
   (service → **Connect** → **HTTPS**):
   ```bash
   cp .env.example .env
   ```

### Run each script in a separate terminal

`uv run` executes inside the project environment, so there is no venv to activate:

```bash
# 1. Start ingesting transactions
uv run 01_ingest.py

# 2. Train the model (once enough data is present — at least 200 rows)
uv run 02_train_model.py

# 3. Launch the dashboard
uv run streamlit run 04_dashboard.py

# Optional: verify the deployed UDF against the live cluster
uv run tests.py
```

### Managing dependencies

```bash
uv add <package>       # add a dependency (updates pyproject.toml + uv.lock)
uv remove <package>    # drop one
uv lock --upgrade      # refresh the lock within the bounds in pyproject.toml
```

> `03_udf_scorer.py` is deployed to ClickHouse as an executable UDF — ClickHouse calls it automatically on each insert via `mv_score_transactions`. It does not need to be run manually.

# Contributors
[Oussama Chakri](https://github.com/Ochakri-CHDB) is the original creator of this demo
[Caio Ishizaka Costa](https://github.com/ch-caioishizaka) created the UDF and feature store for real time scoring