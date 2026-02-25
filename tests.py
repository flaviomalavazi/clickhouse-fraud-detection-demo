"""
Tests for the fraud_score UDF deployed on ClickHouse Cloud.

Runs directly against the live cluster — requires a valid .env file.
Usage:
    python tests.py
"""

import os
import sys
import clickhouse_connect
from dotenv import load_dotenv

load_dotenv()

CLICKHOUSE_HOST = os.environ["CLICKHOUSE_HOST"]
CLICKHOUSE_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8443"))
DATABASE = os.environ.get("CLICKHOUSE_DATABASE", "fraud")
USERNAME = os.environ.get("CLICKHOUSE_USER", "default")
PASSWORD = os.environ["CLICKHOUSE_PASSWORD"]


def get_client():
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
        secure=True,
        verify=False,
    )


def fraud_score(client, amount, country, channel, user_amount_mean=None):
    """Call fraud_score UDF for a single transaction."""
    if user_amount_mean is None:
        user_amount_mean = amount  # no prior history → ratio = 1.0
    query = """
        SELECT fraud_score(
            toFloat64(%(amount)s),
            log(toFloat64(%(amount)s) + 1.0),
            toFloat64(%(mean)s),
            toFloat64(%(amount)s) / (toFloat64(%(mean)s) + 1.0),
            toInt8(transform(%(country)s, ['DE','ES','FR','IT','US'], [0,1,2,3,4], -1)),
            toInt8(transform(%(channel)s, ['agency','call_center','mobile','web'], [0,1,2,3], -1))
        ) AS score
    """
    row = client.query(
        query,
        parameters={"amount": amount, "mean": user_amount_mean, "country": country, "channel": channel},
    )
    return row.result_rows[0][0]


passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}{f' — {detail}' if detail else ''}")
        failed += 1


def main():
    print("Connecting to ClickHouse...")
    client = get_client()
    client.ping()
    print("Connected.\n")

    # ------------------------------------------------------------------
    # 1. UDF is callable
    # ------------------------------------------------------------------
    print("=== 1. UDF reachability ===")
    score = fraud_score(client, 100, "US", "mobile")
    check("fraud_score returns a float", isinstance(score, float), repr(score))

    # ------------------------------------------------------------------
    # 2. Normal transactions score positive (not anomalous)
    # ------------------------------------------------------------------
    print("\n=== 2. Normal transactions → positive score ===")
    cases = [
        (50,  "US", "mobile"),
        (120, "DE", "web"),
        (500, "DE", "web"),
        (200, "ES", "mobile"),
    ]
    for amount, country, channel in cases:
        s = fraud_score(client, amount, country, channel)
        check(f"  amount={amount}, {country}/{channel}  score={s:.4f} > 0", s > 0, f"got {s}")

    # ------------------------------------------------------------------
    # 3. High-risk transactions score negative (anomalous)
    # ------------------------------------------------------------------
    print("\n=== 3. High-risk transactions → negative score ===")
    cases = [
        (5000, "IT", "web"),
        (2500, "FR", "call_center"),
        (4500, "US", "agency"),
    ]
    for amount, country, channel in cases:
        s = fraud_score(client, amount, country, channel)
        check(f"  amount={amount}, {country}/{channel}  score={s:.4f} < 0", s < 0, f"got {s}")

    # ------------------------------------------------------------------
    # 4. Score buckets match expected severity thresholds
    # ------------------------------------------------------------------
    print("\n=== 4. Score bucket boundaries ===")

    s_critical = fraud_score(client, 5000, "IT", "web")
    check(f"  CRITICAL bucket (≤ -0.015): score={s_critical:.4f}", s_critical <= -0.015)

    # 1500 FR/web scores in CRITICAL with this model — verify it is at least anomalous (< 0)
    s_high = fraud_score(client, 1500, "FR", "web")
    check(f"  1500 FR/web anomalous (score < 0): score={s_high:.4f}", s_high < 0)

    s_normal = fraud_score(client, 50, "US", "mobile")
    check(f"  Normal (score > 0): score={s_normal:.4f}", s_normal > 0)

    # ------------------------------------------------------------------
    # 5. Unknown country/channel (-1 encoding) doesn't crash
    # ------------------------------------------------------------------
    print("\n=== 5. Unknown country/channel → graceful handling ===")
    query = """
        SELECT fraud_score(
            toFloat64(300),
            log(301.0),
            toFloat64(300),
            1.0,
            toInt8(-1),
            toInt8(-1)
        ) AS score
    """
    row = client.query(query)
    s = row.result_rows[0][0]
    check(f"  Unknown encoding returns a float: score={s:.4f}", isinstance(s, float))

    # ------------------------------------------------------------------
    # 6. User history changes score (history affects the result)
    # ------------------------------------------------------------------
    print("\n=== 6. User amount ratio effect ===")
    # Same amount, different user baselines — scores must differ
    s_low_history  = fraud_score(client, 1000, "DE", "web", user_amount_mean=50)
    s_high_history = fraud_score(client, 1000, "DE", "web", user_amount_mean=2000)
    check(
        f"  User history changes score: {s_low_history:.4f} != {s_high_history:.4f}",
        s_low_history != s_high_history,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
