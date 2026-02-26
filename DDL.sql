-- Create a dedicated database for the fraud demo
CREATE DATABASE IF NOT EXISTS fraud;

-- Main transaction stream
CREATE TABLE IF NOT EXISTS fraud.transactions
(
    transaction_id UUID,
    ts             DateTime,
    user_id        UInt32,
    amount         Float32,
    country        String,
    channel        String,
    is_anomaly     UInt8
)
ENGINE = MergeTree
ORDER BY (ts, user_id);

-- ======================================================
-- User statistics (AggregatingMergeTree)
-- Incrementally maintains per-user average transaction amount.
-- The scoring MV JOINs against this to compute user_amount_mean
-- and user_amount_ratio without scanning the full transactions table.
--
-- Note: because mv_user_stats and mv_score_transactions both fire on the same
-- insert batch, the stats here always reflect *prior* history —
-- the current batch is not included in the user mean yet. This is
-- the correct semantics for anomaly detection.
-- ======================================================
DROP VIEW IF EXISTS fraud.mv_user_stats;
DROP TABLE IF EXISTS fraud.user_stats;

CREATE TABLE IF NOT EXISTS fraud.user_stats
(
    user_id     UInt32,
    amount_avg  AggregateFunction(avg, Float32)
)
ENGINE = AggregatingMergeTree
ORDER BY user_id;

CREATE MATERIALIZED VIEW fraud.mv_user_stats
TO fraud.user_stats AS
SELECT
    user_id,
    avgState(amount) AS amount_avg
FROM fraud.transactions
GROUP BY user_id;

-- ======================================================
-- transactions_scored: stores every transaction that passed through
-- the IsolationForest UDF with its raw anomaly score.
-- Query with WHERE score < 0 to isolate fraud alerts.
-- ======================================================
CREATE TABLE IF NOT EXISTS fraud.transactions_scored
(
    transaction_id UUID,
    ts             DateTime,
    user_id        UInt32,
    amount         Float32,
    country        String,
    channel        String,
    score          Float32,
    model          String,
    reason         String
)
ENGINE = MergeTree
ORDER BY (ts, user_id);

-- ======================================================
-- ML scoring MV — calls the IsolationForest executable UDF
--
-- Fires on every insert into fraud.transactions.
-- Joins with fraud.user_stats to get per-user baselines.
-- Writes ALL scored rows to fraud.transactions_scored.
-- Filter on score < 0 at query time to isolate alerts.
--
-- Category encodings (alphabetical — must match pandas cat.codes
-- used in train_model.py):
--   country: DE=0, ES=1, FR=2, IT=3, US=4
--   channel: agency=0, call_center=1, mobile=2, web=3
-- ======================================================
DROP VIEW IF EXISTS fraud.mv_score_transactions;

CREATE MATERIALIZED VIEW fraud.mv_score_transactions
TO fraud.transactions_scored AS
SELECT
    transaction_id,
    ts,
    user_id,
    amount,
    country,
    channel,
    toFloat32(score) AS score,
    'IsolationForest(amount,user,country,channel)' AS model,
    multiIf(
        score >= 0,      '',
        score <= -0.015, 'CRITICAL: Strongly isolated by the model',
        score <= -0.010, 'HIGH: Highly anomalous transaction pattern',
        score <= -0.005, 'MEDIUM: Moderately unusual transaction',
                         'LOW: Slight deviation from expected behavior'
    ) AS reason
FROM (
    SELECT
        t.transaction_id,
        t.ts,
        t.user_id,
        t.amount,
        t.country,
        t.channel,
        fraud_score(
            t.amount,
            log(t.amount + 1.0),
            coalesce(us.avg_amount, toFloat64(t.amount)),
            t.amount / (coalesce(us.avg_amount, toFloat64(t.amount)) + 1.0),
            toInt8(transform(t.country, ['DE', 'ES', 'FR', 'IT', 'US'], [0, 1, 2, 3, 4], -1)),
            toInt8(transform(t.channel, ['agency', 'call_center', 'mobile', 'web'], [0, 1, 2, 3], -1))
        ) AS score
    FROM fraud.transactions AS t
    LEFT JOIN (
        SELECT user_id, avgMerge(amount_avg) AS avg_amount
        FROM fraud.user_stats
        GROUP BY user_id
    ) AS us ON t.user_id = us.user_id
);