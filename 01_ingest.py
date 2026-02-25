import os
import random
import sys
import time
import clickhouse_connect
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------
# ClickHouse Cloud connection settings
# -----------------------------------------
CLICKHOUSE_HOST = os.environ["CLICKHOUSE_HOST"]
CLICKHOUSE_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8443"))
DATABASE = os.environ.get("CLICKHOUSE_DATABASE", "fraud")
TABLE = "transactions"
USERNAME = os.environ.get("CLICKHOUSE_USER", "default")
PASSWORD = os.environ["CLICKHOUSE_PASSWORD"]


def get_client():
    """
    Create a ClickHouse Connect client for ClickHouse Cloud.
    Cloud requires TLS (secure=True) and port 8443.
    See docs: Python integration with ClickHouse Connect.
    """
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


def insert_batch(client, batch_size: int):
    """
    Insert `batch_size` rows using INSERT ... SELECT and the numbers() table function.

    We generate:
    - ts: now minus a random offset of up to 60 seconds
    - user_id: random user ID
    - amount: 98% normal range, 2% very high amounts (fraud-like)
    - country / channel: random categorical values
    - is_anomaly: always 0 here (model will create alerts separately)
    """
    insert_sql = f"""
        INSERT INTO {DATABASE}.{TABLE}
        SELECT
            now() - INTERVAL randCanonical() * 60 SECOND AS ts,
            toUInt32(1 + randCanonical() * 100000)     AS user_id,
            -- 98% normal amounts, 2% suspiciously high
            toFloat32(
                if(randBernoulli(0.98),
                   5  + randCanonical() * 245,   -- ~5..250
                   1500 + randCanonical() * 3500 -- ~1500..5000
                )
            )                                        AS amount,
            ['FR', 'DE', 'US', 'ES', 'IT'][1 + (rand() % 5)]       AS country,
            ['web', 'mobile', 'call_center', 'agency'][1 + (rand() % 4)] AS channel,
            toUInt8(0)                              AS is_anomaly
        FROM numbers({batch_size})
    """
    # numbers(N) returns N rows with a sequential UInt64 "number" column,
    client.command(insert_sql)


def main():
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    client = get_client()
    print(f"Starting ingestion into fraud.transactions ({batch_size} rows per batch)...")
    try:
        while True:
            actual = random.randint(int(batch_size * 0.75), int(batch_size * 1.25))
            start = time.time()
            insert_batch(client, actual)
            elapsed = time.time() - start
            print(f"Inserted {actual} rows in {elapsed:.3f} seconds")
            # Try to keep total rate around one batch per second
            sleep_time = max(0.0, 1.0 - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Ingestion stopped by user.")


if __name__ == "__main__":
    main()