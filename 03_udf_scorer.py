#!/usr/bin/env python3
"""
ClickHouse Executable UDF — IsolationForest fraud scorer (chunk-batch mode).

Reads tab-separated feature rows from stdin in chunk-header protocol,
scores the entire chunk in one vectorized call, outputs one Float64
anomaly score per line to stdout.

send_chunk_header must be enabled in the ClickHouse UDF registration so
that ClickHouse sends a row-count header before each chunk. This allows
the script to collect all rows, build a single DataFrame, and call
decision_function once per chunk instead of once per row — roughly
100-1000x faster for typical batch sizes.

Input protocol (send_chunk_header = true):
    <N>\n                    ← chunk header: number of rows in this chunk
    col1\tcol2\t...\n        ← row 1
    col1\tcol2\t...\n        ← row 2
    ...                      ← rows 3..N
    <M>\n                    ← next chunk header
    ...

Expected input columns (tab-separated, in this order):
    1. amount             Float32   — raw transaction amount
    2. log_amount         Float64   — log(amount + 1)
    3. user_amount_mean   Float64   — per-user avg amount from fraud.user_stats
    4. user_amount_ratio  Float64   — amount / (user_amount_mean + 1)
    5. country_encoded    Int8      — see mapping below
    6. channel_encoded    Int8      — see mapping below

Country encoding (alphabetical, must match training in train_model.py):
    DE=0, ES=1, FR=2, IT=3, US=4

Channel encoding (alphabetical, must match training in train_model.py):
    agency=0, call_center=1, mobile=2, web=3
"""

import sys
import pickle
import pathlib
import numpy as np
import pandas as pd

MODEL_PATH = pathlib.Path(__file__).parent / "fraud_model.pkl"

FEATURES = [
    "amount",
    "log_amount",
    "user_amount_mean",
    "user_amount_ratio",
    "country_encoded",
    "channel_encoded",
]


def main():
    if not MODEL_PATH.exists():
        print(f"ERROR: model file not found at {MODEL_PATH}", file=sys.stderr, flush=True)
        sys.exit(1)

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    stdin = sys.stdin

    while True:
        # Read chunk header: number of rows in this chunk
        header = stdin.readline()
        if not header:
            break  # EOF — no more chunks

        n_rows = int(header.strip())
        if n_rows == 0:
            continue

        # Read all rows for this chunk into a list
        rows = []
        for _ in range(n_rows):
            line = stdin.readline().rstrip("\n")
            parts = line.split("\t")
            if len(parts) != len(FEATURES):
                print(
                    f"ERROR: expected {len(FEATURES)} columns, got {len(parts)}: {line!r}",
                    file=sys.stderr, flush=True,
                )
                rows.append([0.0] * len(FEATURES))
            else:
                rows.append([float(p) for p in parts])

        # Score the entire chunk in one vectorized call
        features = pd.DataFrame(rows, columns=FEATURES)
        scores = model.decision_function(features)

        # Output one score per line, then flush once for the whole chunk
        sys.stdout.write("\n".join(str(float(s)) for s in scores) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()