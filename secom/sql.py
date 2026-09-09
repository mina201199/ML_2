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


def augment_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """為每個感測器產生「與近期基線的偏離量」，用 SQL window function 算。

    這正是製程監控的核心直覺 —— 感測器的**絕對值**通常不重要，重要的是它
    **相對於最近的常態漂移了多少**。這種特徵在 SQL 裡表達最自然，而且是可以
    直接搬到生產倉儲的寫法（真實廠內這一步就是在倉儲裡做的，不是在 notebook）。

    對每個原始感測器產生兩欄：
        偏離量      當前值減去前 20 筆的移動平均
        標準化偏離  偏離量除以前 20 筆的標準差（跨感測器可比）

    窗口刻意用 `ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING` —— 不含當前列，
    以維持「與過去基線比較」的特徵定義。當前可取得的量測本身不必然構成洩漏，
    但基線必須只由過去構成，否則就是用未來資訊算特徵。

    欄位命名：第 i 個原始感測器（f000 起算）對應
        f{900 + 2i}   偏離量
        f{901 + 2i}   標準化偏離
    所以 f000 -> f900 / f901，f001 -> f902 / f903，依此類推。取這個編號區間是
    為了讓 data.FEATURE_RE 仍然收得到它們，同時不與 f000..f589 相撞。

    不做任何監督式篩選、不使用標籤，也不看未來的列 —— 這是為了讓消融實驗不受
    「用測試集 SHAP 挑特徵」的污染。同時間戳的排序依輸入順序，假設它就是產線順序。
    """
    from .data import feature_cols

    raw = [c for c in feature_cols(df) if int(c[1:]) < 590]
    ordered = df.reset_index(drop=True).assign(_row_id=range(len(df)))

    expressions = []
    for i, col in enumerate(raw):
        expressions.extend([
            f"{col} - AVG({col}) OVER w AS f{900 + 2 * i:03d}",
            f"({col} - AVG({col}) OVER w)"
            f" / NULLIF(STDDEV_SAMP({col}) OVER w, 0) AS f{901 + 2 * i:03d}",
        ])

    query = (
        "SELECT " + ", ".join(expressions)
        + " FROM observations"
        + " WINDOW w AS (ORDER BY _row_id ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)"
        + " ORDER BY _row_id"
    )
    with duckdb.connect(":memory:") as con:
        con.register("observations", ordered)
        extra = con.execute(query).df()
    return pd.concat([df.reset_index(drop=True), extra], axis=1)


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
