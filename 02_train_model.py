"""
Train an IsolationForest on recent transactions from ClickHouse and package it
as a self-contained, versioned UDF bundle under function/<timestamp>/:

    function/20260818T181500Z/
        fraud_model.pkl    the serialized IsolationForest
        main.py            copy of 03_udf_scorer.py (the UDF entrypoint)
        requirements.txt   pinned runtime deps for that exact model

Each run writes a new timestamped directory, so model versions never overwrite
each other and any bundle can be redeployed as-is. The bundle is everything the
UDF host needs — nothing else from this repo has to be shipped alongside it.

Run this script once before registering the UDF, and re-run it whenever you
want to refresh the model (e.g. nightly, or on demand).
"""

import os
import pickle
import pathlib
import platform
import shutil
import datetime as dt
from importlib import metadata
import numpy as np
import pandas as pd
import sklearn
import clickhouse_connect
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest

# Load the .env next to this script, and let it win over any pre-existing
# shell variables (override=True) so a stray CLICKHOUSE_* export in the
# environment can't silently redirect the demo at another database.
load_dotenv(dotenv_path=pathlib.Path(__file__).parent / ".env", override=True)

CLICKHOUSE_HOST = os.environ["CLICKHOUSE_HOST"]
CLICKHOUSE_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8443"))
DATABASE = os.environ.get("CLICKHOUSE_DATABASE", "fraud")
TX_TABLE = "transactions"
USERNAME = os.environ.get("CLICKHOUSE_USER", "default")
PASSWORD = os.environ["CLICKHOUSE_PASSWORD"]

REPO_ROOT = pathlib.Path(__file__).parent
FUNCTION_DIR = REPO_ROOT / "function"
UDF_SOURCE = REPO_ROOT / "03_udf_scorer.py"

# main.py loads the pickle by this name from its own directory, so the filename
# has to stay in sync with MODEL_PATH in 03_udf_scorer.py.
MODEL_FILENAME = "fraud_model.pkl"
UDF_FILENAME = "main.py"

# The only third-party packages main.py needs at runtime: pandas builds the
# feature frame, scikit-learn (with numpy underneath) unpickles and scores the
# model. Everything else — scipy, joblib, threadpoolctl — is pulled in as a
# scikit-learn dependency and does not need to be listed.
UDF_RUNTIME_PACKAGES = ["scikit-learn", "pandas", "numpy"]

# Python version the bundle is meant to be installed on, on the UDF host.
TARGET_PYTHON = "3.11"

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


def make_version_dir() -> pathlib.Path:
    """
    Create function/<UTC timestamp>/ for this training run.

    UTC and a sortable, filesystem-safe format (20260818T181500Z) so versions
    order lexicographically and mean the same thing regardless of where the
    trainer ran.
    """
    version = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    version_dir = FUNCTION_DIR / version
    version_dir.mkdir(parents=True, exist_ok=False)
    return version_dir


def write_requirements(version_dir: pathlib.Path) -> pathlib.Path:
    """
    Pin main.py's runtime deps to the versions that produced the pickle.

    A scikit-learn pickle is only guaranteed to load under the same
    scikit-learn version that wrote it — a mismatch on the UDF host raises
    InconsistentVersionWarning at best and fails to unpickle at worst. Pinning
    exactly is the point of generating this per model version rather than
    sharing one requirements.txt across all of them.
    """
    requirements_path = version_dir / "requirements.txt"
    pins = [f"{pkg}=={metadata.version(pkg)}" for pkg in UDF_RUNTIME_PACKAGES]

    # The pins describe *this* interpreter's packages, so training on a different
    # Python than the host can emit versions the host cannot install (numpy 2.5,
    # for example, requires >=3.12). Fail loudly rather than ship a bundle that
    # only breaks at deploy time.
    running = ".".join(platform.python_version_tuple()[:2])
    if running != TARGET_PYTHON:
        print(
            f"  WARNING: training on Python {running}, but the bundle targets "
            f"{TARGET_PYTHON}. The pins below may not install on the UDF host, "
            f"and a scikit-learn version mismatch can break unpickling. "
            f"Re-run with Python {TARGET_PYTHON} (`uv sync` uses .python-version)."
        )
    header = [
        f"# Runtime dependencies for {UDF_FILENAME} / {MODEL_FILENAME} in this directory.",
        f"# Generated by 02_train_model.py — do not edit by hand.",
        "#",
        f"# Pinned to the environment that trained the model:",
        f"#   Python {platform.python_version()}, scikit-learn {sklearn.__version__}",
        "#",
        f"# Install on the UDF host (Python {TARGET_PYTHON}):",
        f"#   python{TARGET_PYTHON} -m pip install -r requirements.txt",
        "",
    ]
    requirements_path.write_text("\n".join(header + pins) + "\n")
    return requirements_path


def write_bundle(model) -> pathlib.Path:
    """
    Write the deployable bundle: pickle + UDF entrypoint + pinned requirements.
    """
    version_dir = make_version_dir()

    model_path = version_dir / MODEL_FILENAME
    with open(model_path, "wb") as f:
        # Protocol 5 is the default on 3.8+ and readable by the target 3.11 host.
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  {model_path.relative_to(REPO_ROOT)}")

    # 03_udf_scorer.py is not importable under its own name (a leading digit is
    # not a valid identifier), so it ships as main.py — which is also the
    # entrypoint name the UDF host expects.
    udf_path = version_dir / UDF_FILENAME
    shutil.copyfile(UDF_SOURCE, udf_path)
    print(f"  {udf_path.relative_to(REPO_ROOT)}  (copied from {UDF_SOURCE.name})")

    requirements_path = write_requirements(version_dir)
    print(f"  {requirements_path.relative_to(REPO_ROOT)}")

    return version_dir


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

    print("Writing UDF bundle:")
    version_dir = write_bundle(model)

    print(f"\nDone. Model version {version_dir.name} ready to deploy from {version_dir}")


if __name__ == "__main__":
    train_and_serialize()
