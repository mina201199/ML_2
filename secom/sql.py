"""SQL 層（DuckDB）。

為什麼這個檔案存在
------------------
兩個理由，一個技術一個現實。

技術上：資料側寫與時間窗聚合這類工作，SQL 寫起來比 pandas 清楚，
而且這是資料倉儲裡真正會發生的事 —— 特徵不會在 notebook 裡生，
會在倉儲裡用 SQL 生好，再餵給模型。

現實上：幾乎每一份 Data Analyst 的職缺說明第一條就是 SQL。一個純 pandas 的
專案沒辦法證明這件事。這裡的 SQL 是真的在做事，不是擺著好看。

DuckDB 直接讀 parquet，不需要架資料庫，所以 repo 保持零外部依賴。
"""

from __future__ import annotations

import duckdb
import pandas as pd

from . import config as cfg_mod


def connect(cfg) -> duckdb.DuckDBPyConnection:
    """開一個 in-process 連線，並把 parquet 註冊成 view `lots`。"""
    path = cfg_mod.resolve_dir(cfg.data.processed_dir) / "secom.parquet"
    con = duckdb.connect(":memory:")
    con.execute(
        f"CREATE VIEW lots AS SELECT * FROM read_parquet('{path.as_posix()}')"
    )
    return con


# ── 側寫：README 與 EDA 的數字全部由這支查詢產生 ──────────────────
PROFILE_SQL = """
SELECT
    COUNT(*)                                        AS n_lots,
    SUM(fail)                                       AS n_fail,
    COUNT(*) - SUM(fail)                            AS n_pass,
    ROUND(AVG(fail), 6)                             AS fail_rate,
    ROUND((COUNT(*) - SUM(fail)) / SUM(fail), 2)    AS imbalance_ratio,
    MIN(ts)                                         AS ts_from,
    MAX(ts)                                         AS ts_to,
    DATE_DIFF('day', MIN(ts), MAX(ts))              AS span_days
FROM lots
"""


def profile(con) -> pd.DataFrame:
    return con.execute(PROFILE_SQL).df()


# ── 漂移：本專題最重要的資料事實，用 SQL 表達最直接 ────────────────
DRIFT_SQL = """
WITH weekly AS (
    SELECT
        DATE_TRUNC('week', ts)  AS week,
        COUNT(*)                AS n_lots,
        SUM(fail)               AS n_fail,
        AVG(fail)               AS fail_rate
    FROM lots
    GROUP BY 1
)
SELECT
    week,
    n_lots,
    n_fail,
    ROUND(fail_rate, 4) AS fail_rate,
    -- 與整體盛行率的比值：> 1 表示這週比平均差
    ROUND(fail_rate / (SELECT AVG(fail) FROM lots), 2) AS vs_overall,
    -- 四週移動平均，用來看趨勢而不是單週雜訊
    ROUND(AVG(fail_rate) OVER (
        ORDER BY week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ), 4) AS fail_rate_ma4
FROM weekly
ORDER BY week
"""


def drift(con) -> pd.DataFrame:
    return con.execute(DRIFT_SQL).df()


# ── 產出時間特徵：真實廠內這一步就是在倉儲裡做的 ──────────────────
def build_time_features(con, feature_cols: list[str]) -> pd.DataFrame:
    """為選定的感測器產生時間窗特徵：與近期基線的偏離量。

    這正是製程監控的核心直覺 —— 感測器的**絕對值**通常不重要，
    重要的是它**相對於最近的常態漂移了多少**。這種特徵在 SQL 裡用
    window function 表達最自然，而且是可以直接搬到生產倉儲的寫法。

    對每個感測器 f 產生兩個欄位：
        f_dev20  當前值減去前 20 批的移動平均（偏離量）
        f_z20    偏離量除以前 20 批的標準差（標準化偏離，跨感測器可比）

    窗口刻意用 `ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING` ——
    不含當前列。含了就是用當下的值去算自己的基線，那是洩漏。
    """
    parts = []
    for c in feature_cols:
        parts.append(
            f"""
    {c} - AVG({c}) OVER w_{c} AS {c}_dev20,
    ({c} - AVG({c}) OVER w_{c})
        / NULLIF(STDDEV_SAMP({c}) OVER w_{c}, 0) AS {c}_z20"""
        )
    windows = ",\n    ".join(
        f"w_{c} AS (ORDER BY ts ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)"
        for c in feature_cols
    )
    sql = f"""
SELECT
    ts,
    fail,{','.join(parts)}
FROM lots
WINDOW
    {windows}
ORDER BY ts
"""
    return con.execute(sql).df()


# ── 缺值模式：缺值本身帶不帶訊息，用 SQL 一次算完 ─────────────────
def missingness_signal(con, feature_cols: list[str]) -> pd.DataFrame:
    """每個感測器「有量到 vs 沒量到」時的 fail 率落差。

    落差大表示「這個站點當時沒量到」本身就是訊號 ——
    這是主模型刻意不填補缺值的理由。
    """
    unions = "\nUNION ALL\n".join(
        f"""SELECT '{c}' AS feature,
       AVG(CASE WHEN {c} IS NULL THEN fail END)     AS fail_when_missing,
       AVG(CASE WHEN {c} IS NOT NULL THEN fail END) AS fail_when_present,
       AVG(CASE WHEN {c} IS NULL THEN 1.0 ELSE 0 END) AS missing_frac
FROM lots"""
        for c in feature_cols
    )
    sql = f"""
WITH per_feature AS (
{unions}
)
SELECT
    feature,
    ROUND(missing_frac, 4)       AS missing_frac,
    ROUND(fail_when_missing, 4)  AS fail_when_missing,
    ROUND(fail_when_present, 4)  AS fail_when_present,
    ROUND(ABS(fail_when_missing - fail_when_present), 4) AS gap
FROM per_feature
WHERE missing_frac BETWEEN 0.02 AND 0.98
ORDER BY gap DESC
"""
    return con.execute(sql).df()
