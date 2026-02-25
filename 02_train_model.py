"""
Train an IsolationForest on recent transactions from ClickHouse and serialize
it to fraud_model.pkl so the executable UDF (03_udf_scorer.py)
can load it for scoring inside ClickHouse.

Run this script once before registering the UDF, and re-run it whenever
you want to refresh the model (e.g. nightly, or on demand).
"""

import os
import pickle
import pathlib
import numpy as np
import pandas as pd
import clickhouse_connect
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest

load_dotenv()

CLICKHOUSE_HOST = os.environ["CLICKHOUSE_HOST"]
CLICKHOUSE_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8443"))
DATABASE = os.environ.get("CLICKHOUSE_DATABASE", "fraud")
TX_TABLE = "transactions"
USERNAME = os.environ.get("CLICKHOUSE_USER", "default")
PASSWORD = os.environ["CLICKHOUSE_PASSWORD"]

MODEL_PATH = pathlib.Path(__file__).parent / "fraud_model.pkl"

FEATURES = [
    "amount",
    "log_amount",
    "user_amount_mean",
    "user_amount_ratio",
    "country_encoded",
    "channel_encoded",
]


def get_client():
    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
        secure=True,
        verify=False,
    )
    client.ping()
    return client


def load_training_data(client, minutes: int = 30, limit: int = 10000) -> pd.DataFrame:
    query = f"""
        SELECT ts, user_id, amount, country, channel
        FROM {DATABASE}.{TX_TABLE}
        WHERE ts >= now() - INTERVAL {minutes} MINUTE
        ORDER BY ts DESC
        LIMIT {limit}
    """
    return client.query_df(query)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df["amount"] = df["amount"].astype(float)
    df["log_amount"] = np.log(df["amount"] + 1.0)

    user_means = df.groupby("user_id")["amount"].transform("mean")
    df["user_amount_mean"] = user_means
    df["user_amount_ratio"] = df["amount"] / (user_means + 1.0)

    # Alphabetical encoding — must match the transform() expressions in ClickHouse:
    #   country: DE=0, ES=1, FR=2, IT=3, US=4
    #   channel: agency=0, call_center=1, mobile=2, web=3
    df["country_encoded"] = df["country"].astype("category").cat.codes
    df["channel_encoded"] = df["channel"].astype("category").cat.codes

    return df


def train_and_serialize():
    print("Connecting to ClickHouse...")
    client = get_client()

    print("Loading training data (last 30 minutes, up to 10k rows)...")
    df = load_training_data(client)

    if len(df) < 200:
        raise RuntimeError(f"Not enough data to train: {len(df)} rows (need at least 200)")

    print(f"Loaded {len(df)} rows. Building features...")
    df = build_features(df)

    print("Training IsolationForest...")
    model = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42,
    )
    model.fit(df[FEATURES])

    print(f"Serializing model to {MODEL_PATH}...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Done. Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_serialize()
