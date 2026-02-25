#!/usr/bin/env python3
"""
Real-Time Fraud Detection Dashboard
ClickHouse Cloud · IsolationForest UDF · Materialized Views
"""

import os
import pandas as pd
import altair as alt
import streamlit as st
import clickhouse_connect
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

load_dotenv()

CLICKHOUSE_HOST = os.environ["CLICKHOUSE_HOST"]
CLICKHOUSE_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8443"))
DATABASE        = os.environ.get("CLICKHOUSE_DATABASE", "fraud")
USERNAME        = os.environ.get("CLICKHOUSE_USER", "default")
PASSWORD        = os.environ["CLICKHOUSE_PASSWORD"]

# ── ClickHouse brand palette ──────────────────────────────────────────────────
CH_YELLOW  = "#FAFF69"
CH_BG      = "#0B0D11"
CH_SURFACE = "#13161C"
CH_CARD    = "#1A1E27"
CH_BORDER  = "#2A2F3D"
CH_TEXT    = "#F0F6FC"
CH_SUBTLE  = "#8B949E"
CH_BLUE    = "#58A6FF"
CH_GREEN   = "#3FB950"
CH_RED     = "#F85149"
CH_ORANGE  = "#D29922"
CH_AMBER   = "#E3B341"

SEVERITY_COLOR = {
    "CRITICAL": CH_RED,
    "HIGH":     CH_ORANGE,
    "MEDIUM":   CH_AMBER,
    "LOW":      CH_GREEN,
}

# ── Altair dark theme ─────────────────────────────────────────────────────────

@alt.theme.register("ch", enable=True)
def _ch_theme() -> alt.theme.ThemeConfig:
    return alt.theme.ThemeConfig({
        "config": {
            "background":   CH_CARD,
            "view":         {"stroke": "transparent", "fill": CH_CARD},
            "axis": {
                "gridColor":   CH_BORDER,
                "domainColor": CH_BORDER,
                "tickColor":   CH_BORDER,
                "labelColor":  CH_SUBTLE,
                "titleColor":  CH_SUBTLE,
                "labelFont":   "Inter, sans-serif",
                "titleFont":   "Inter, sans-serif",
            },
            "legend": {
                "labelColor":  CH_SUBTLE,
                "titleColor":  CH_SUBTLE,
                "labelFont":   "Inter, sans-serif",
            },
            "title": {
                "color":    CH_TEXT,
                "fontSize": 13,
                "font":     "Inter, sans-serif",
                "fontWeight": "600",
                "anchor":   "start",
                "offset":   8,
            },
        }
    })

# ── CSS injection ─────────────────────────────────────────────────────────────

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
#MainMenu, footer, header {{ visibility: hidden; }}

/* App shell */
.stApp {{
    background-color: {CH_BG};
}}
.block-container {{
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}}

/* Metrics */
div[data-testid="stMetric"] {{
    background: {CH_CARD};
    border: 1px solid {CH_BORDER};
    border-radius: 12px;
    padding: 18px 22px 14px;
}}
div[data-testid="stMetric"] label {{
    color: {CH_SUBTLE} !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
div[data-testid="stMetricValue"] {{
    color: {CH_TEXT} !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    line-height: 1.1;
}}
div[data-testid="stMetricDelta"] {{
    color: {CH_GREEN} !important;
}}

/* Chart containers */
div[data-testid="stArrowVegaLiteChart"] > div {{
    border-radius: 12px;
    border: 1px solid {CH_BORDER};
    overflow: hidden;
    padding: 4px;
}}

/* Tabs */
div[data-baseweb="tab-list"] {{
    background: transparent;
    border-bottom: 1px solid {CH_BORDER};
    gap: 4px;
}}
button[data-baseweb="tab"] {{
    color: {CH_SUBTLE} !important;
    font-weight: 500;
    background: transparent !important;
    padding: 10px 20px !important;
    border-radius: 8px 8px 0 0 !important;
}}
button[data-baseweb="tab"]:hover {{
    color: {CH_TEXT} !important;
    background: {CH_CARD} !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {CH_YELLOW} !important;
    border-bottom: 2px solid {CH_YELLOW} !important;
    background: {CH_CARD} !important;
}}
div[data-baseweb="tab-panel"] {{
    padding-top: 16px;
}}

/* Dataframe */
div[data-testid="stDataFrame"] {{
    border: 1px solid {CH_BORDER};
    border-radius: 10px;
    overflow: hidden;
}}

/* Section headers */
.section-header {{
    color: {CH_SUBTLE};
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 24px 0 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid {CH_BORDER};
}}

/* Live badge */
.live-badge {{
    display: inline-block;
    background: {CH_RED}22;
    color: {CH_RED};
    border: 1px solid {CH_RED}55;
    border-radius: 20px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    padding: 3px 10px;
    animation: pulse 2s ease-in-out infinite;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.5; }}
}}
</style>
"""

# ── Connection ────────────────────────────────────────────────────────────────

@st.cache_resource
def get_client():
    c = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=USERNAME,
        password=PASSWORD,
        database=DATABASE,
        secure=True,
        verify=False,
    )
    c.ping()
    return c

# ── Queries ───────────────────────────────────────────────────────────────────

def load_kpis() -> dict:
    c = get_client()
    row = c.query("""
        SELECT
            (SELECT count() FROM fraud.transactions WHERE ts >= now() - INTERVAL 5 MINUTE) AS tx_5m,
            (SELECT count() FROM fraud.alerts_live  WHERE ts >= now() - INTERVAL 5 MINUTE) AS ml_5m,
            (SELECT count() FROM fraud.transactions)                                         AS total_tx,
            (SELECT count() FROM fraud.alerts_live)                                          AS total_ml,
            (SELECT avg(amount) FROM fraud.alerts_live WHERE ts >= now() - INTERVAL 1 HOUR) AS avg_amt
    """).result_rows[0]
    return {
        "tx_5m":    row[0], "ml_5m":   row[1],
        "total_tx": row[2], "total_ml": row[3], "avg_amt": float(row[4] or 0),
    }


def load_tx_per_minute(minutes: int = 30) -> pd.DataFrame:
    return get_client().query_df(f"""
        SELECT toStartOfMinute(ts) AS minute, count() AS tx_count
        FROM fraud.transactions
        WHERE ts >= now() - INTERVAL {minutes} MINUTE
        GROUP BY minute ORDER BY minute
    """)


def load_alerts_per_minute(minutes: int = 30) -> pd.DataFrame:
    return get_client().query_df(f"""
        SELECT toStartOfMinute(ts) AS minute, count() AS ml_alerts
        FROM fraud.alerts_live
        WHERE ts >= now() - INTERVAL {minutes} MINUTE
        GROUP BY minute ORDER BY minute
    """)


def load_alerts_by_country() -> pd.DataFrame:
    return get_client().query_df("""
        SELECT country, count() AS alerts
        FROM fraud.alerts_live
        WHERE ts >= now() - INTERVAL 1 HOUR
        GROUP BY country ORDER BY alerts DESC
    """)


def load_severity_dist() -> pd.DataFrame:
    df = get_client().query_df("""
        SELECT
            multiIf(score <= -0.015,'CRITICAL', score <= -0.010,'HIGH',
                    score <= -0.005,'MEDIUM', 'LOW') AS severity,
            count() AS n
        FROM fraud.alerts_live
        WHERE ts >= now() - INTERVAL 1 HOUR
        GROUP BY severity
    """)
    order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    df["severity"] = pd.Categorical(df["severity"], categories=order, ordered=True)
    return df.sort_values("severity")


def load_recent_ml_alerts(limit: int = 30) -> pd.DataFrame:
    return get_client().query_df(f"""
        SELECT ts, user_id, amount, country, channel, round(score, 4) AS score, reason
        FROM fraud.alerts_live ORDER BY ts DESC LIMIT {limit}
    """)


# ── Charts ────────────────────────────────────────────────────────────────────

def chart_tx(df: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(df if not df.empty else pd.DataFrame({"minute": [], "tx_count": []}),
                     title="Transactions per minute")
    area = base.mark_area(
        line={"color": CH_YELLOW, "strokeWidth": 2},
        color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color=CH_YELLOW + "44", offset=0),
                   alt.GradientStop(color=CH_YELLOW + "00", offset=1)],
            x1=1, x2=1, y1=1, y2=0,
        ),
    ).encode(
        x=alt.X("minute:T", title=None, axis=alt.Axis(format="%H:%M")),
        y=alt.Y("tx_count:Q", title="Transactions"),
        tooltip=[alt.Tooltip("minute:T", title="Time", format="%H:%M"),
                 alt.Tooltip("tx_count:Q", title="Transactions", format=",")],
    )
    return area.properties(height=230)


def chart_alerts(df: pd.DataFrame) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame()).properties(height=230)
    return (
        alt.Chart(df, title="ML fraud alerts per minute")
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=40, color=CH_YELLOW),
                   color=CH_YELLOW)
        .encode(
            x=alt.X("minute:T", title=None, axis=alt.Axis(format="%H:%M")),
            y=alt.Y("ml_alerts:Q", title="Alerts"),
            tooltip=[alt.Tooltip("minute:T", title="Time", format="%H:%M"),
                     alt.Tooltip("ml_alerts:Q", title="ML Alerts")],
        )
        .properties(height=230)
    )


def chart_country(df: pd.DataFrame) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame()).properties(height=210)
    return (
        alt.Chart(df, title="ML alerts by country (last hour)")
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, color=CH_YELLOW)
        .encode(
            x=alt.X("country:N", title=None, sort="-y"),
            y=alt.Y("alerts:Q", title="Alerts"),
            tooltip=["country:N", "alerts:Q"],
        )
        .properties(height=210)
    )


def chart_severity(df: pd.DataFrame) -> alt.Chart:
    if df.empty:
        return alt.Chart(pd.DataFrame()).properties(height=210)
    return (
        alt.Chart(df, title="Alert severity breakdown (last hour)")
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("severity:N", title=None,
                    sort=["CRITICAL", "HIGH", "MEDIUM", "LOW"]),
            y=alt.Y("n:Q", title="Count"),
            color=alt.Color("severity:N",
                scale=alt.Scale(
                    domain=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                    range=[CH_RED, CH_ORANGE, CH_AMBER, CH_GREEN],
                ),
                legend=None,
            ),
            tooltip=["severity:N", "n:Q"],
        )
        .properties(height=210)
    )

# ── Helpers ───────────────────────────────────────────────────────────────────

def severity_from_reason(reason: str) -> str:
    if not isinstance(reason, str):
        return "LOW"
    for level in SEVERITY_COLOR:
        if reason.upper().startswith(level):
            return level
    return "LOW"


def color_score(val):
    if not isinstance(val, (int, float)):
        return ""
    if val <= -0.015:
        return f"color: {CH_RED}; font-weight: 600"
    if val <= -0.010:
        return f"color: {CH_ORANGE}; font-weight: 600"
    if val <= -0.005:
        return f"color: {CH_AMBER}"
    return f"color: {CH_GREEN}"


# ── App ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Fraud Detection · ClickHouse",
                       layout="wide", page_icon="⚡")
    st.markdown(CSS, unsafe_allow_html=True)
    st_autorefresh(interval=15000, key="autorefresh")

    # Header
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;">
        <span style="font-size:1.7rem;font-weight:800;color:{CH_TEXT};">⚡ Fraud Detection</span>
        <span class="live-badge">● LIVE</span>
        <span style="color:{CH_BORDER};font-size:1.2rem;">|</span>
        <span style="color:{CH_SUBTLE};font-size:0.85rem;">
            ClickHouse Cloud &nbsp;·&nbsp; IsolationForest UDF &nbsp;·&nbsp; Materialized View
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    try:
        kpis        = load_kpis()
        tx_df       = load_tx_per_minute(30)
        alerts_df   = load_alerts_per_minute(30)
        country_df  = load_alerts_by_country()
        severity_df = load_severity_dist()
        ml_df       = load_recent_ml_alerts(30)
    except Exception as e:
        st.error(f"ClickHouse connection error: {e}")
        return

    fraud_rate = (kpis["ml_5m"] / kpis["tx_5m"] * 100) if kpis["tx_5m"] > 0 else 0.0

    # ── KPI row ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Overview — last 5 minutes</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total transactions",   f"{kpis['total_tx']:,}")
    c2.metric("Transactions / 5 min", f"{kpis['tx_5m']:,}")
    c3.metric("ML alerts / 5 min",    f"{kpis['ml_5m']:,}")
    c4.metric("Fraud rate",           f"{fraud_rate:.2f}%")

    # ── Charts row 1 ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Transaction & alert volume — last 30 min</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart_tx(tx_df), use_container_width=True)
    with col2:
        st.altair_chart(chart_alerts(alerts_df), use_container_width=True)

    # ── Charts row 2 ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Fraud breakdown — last hour</div>',
                unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        st.altair_chart(chart_country(country_df), use_container_width=True)
    with col4:
        st.altair_chart(chart_severity(severity_df), use_container_width=True)

    # ── Alert tables ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Recent ML alerts — IsolationForest UDF</div>',
                unsafe_allow_html=True)
    if ml_df.empty:
        st.info("No ML alerts yet — insert some transactions to trigger scoring.")
    else:
        styled = (
            ml_df.style
            .map(color_score, subset=["score"])
            .format({"amount": "{:,.2f}", "score": "{:.4f}",
                     "ts": lambda v: str(v)[:19]})
            .set_properties(**{"background-color": CH_CARD, "color": CH_TEXT})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # Footer
    st.markdown(f"""
    <div style="text-align:center;color:{CH_BORDER};font-size:0.7rem;margin-top:40px;">
        Auto-refreshes every 15 seconds · Avg fraud amount (1h): ${kpis['avg_amt']:,.2f}
        · Total ML alerts: {kpis['total_ml']:,}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()