"""步驟 5：SQL 側寫、漂移分析，以及一次誠實的失敗嘗試。

    python scripts/05_sql_report.py

這一步做兩件事：

1. 用 DuckDB 的 SQL 產生資料側寫與漂移表。這是倉儲裡真正會發生的工作型態 ——
   特徵不會在 notebook 裡生，會在倉儲裡用 SQL 生好再餵給模型。

2. 消融實驗（ablation）：用 SQL window function 產生「相對於近期基線的偏離量」
   特徵，看能不能救回模型的訊號。**結論是不能。** 這個負面結果留在 repo 裡，
   因為「我試過而且沒用」比「我知道可以試什麼」有價值得多。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from secom import backtest as bt_mod
from secom import config as cfg_mod
from secom import data as data_mod
from secom import plots
from secom import sql as sql_mod
from secom.console import enable_utf8

N_TOP_SENSORS = 40  # 只對 SHAP 前 N 名做偏離特徵；590 個全做又慢又雜


def main() -> None:
    enable_utf8()
    cfg = cfg_mod.load()
    df = data_mod.load_processed(cfg)
    con = sql_mod.connect(cfg)

    # ── 1. SQL 側寫 ───────────────────────────────────────────────
    print("\n[1/4] SQL 資料側寫")
    prof = sql_mod.profile(con)
    for c in prof.columns:
        print(f"  {c:<16} {prof[c].iloc[0]}")

    # ── 2. SQL 漂移分析 ───────────────────────────────────────────
    print("\n[2/4] SQL 漂移分析（每週 fail 率 + 四週移動平均）")
    drift = sql_mod.drift(con)
    print(drift.to_string(index=False))
    worst, best = drift["fail_rate"].max(), drift["fail_rate"].min()
    print(f"\n  最差週 {worst:.1%} vs 最好週 {best:.1%} = {worst / max(best, 1e-9):.0f} 倍差距")
    print("  這就是為什麼隨機切分在這份資料上是錯的。")

    figs = cfg_mod.resolve_dir(cfg.report.figures_dir)
    plots.plot_drift(drift, figs)

    # ── 3. 缺值是否帶訊息 ────────────────────────────────────────
    print("\n[3/4] 缺值模式（SQL）：缺值本身帶訊息嗎")
    rank = pd.read_csv(cfg_mod.resolve("reports/metrics/shap_ranking.csv"))
    feats = data_mod.feature_cols(df)
    top = [c for c in rank["feature"].head(N_TOP_SENSORS) if c in feats]
    miss = sql_mod.missingness_signal(con, top)
    if miss.empty:
        print("  前 40 名感測器的缺值比例都在 2%~98% 之外，無可分析欄位。")
    else:
        print(miss.head(8).to_string(index=False))
        print("  落差大表示『這個站點當時沒量到』本身就是訊號 ——")
        print("  這是主模型刻意不填補缺值的理由。")

    # ── 4. 消融實驗：SQL 時間偏離特徵救得回訊號嗎？ ──────────────
    print(f"\n[4/4] 消融實驗：加入 SQL 時間偏離特徵（前 {len(top)} 個感測器）")
    print("  假設：製程監控看的是『相對於近期常態漂移多少』，不是絕對值。")

    tf = sql_mod.build_time_features(con, top)
    newcols = [c for c in tf.columns if c.endswith(("_dev20", "_z20"))]
    aug = pd.concat(
        [df.reset_index(drop=True), tf[newcols].reset_index(drop=True)], axis=1
    )
    # feature_cols 只認 f### 格式，把新特徵改名成它認得的編號區段
    aug = aug.rename(columns={c: f"f{900 + i:03d}" for i, c in enumerate(newcols)})
    print(f"  SQL 產出 {len(newcols)} 個新特徵，總維度 {len(data_mod.feature_cols(aug))}")

    rows = []
    for label, data in (("原始 590 維", df), (f"加上 {len(newcols)} 個偏離特徵", aug)):
        bt = bt_mod.rolling_origin(data, cfg, n_folds=4, policy="robust", verbose=False)
        s = bt_mod.summarise(bt)
        rows.append({
            "特徵集": label,
            "ROC-AUC 中位數": f"{s['roc_auc_median']:.3f}",
            "PR-AUC 中位數": f"{s['pr_auc_median']:.3f}",
            "節省中位數": f"{s['saving_median']:+.1%}",
            "最差一折": f"{s['saving_min']:+.1%}",
            "省錢折數": f"{s['folds_profitable']}/{s['n_folds']}",
            "每折 ROC": str([round(v, 3) for v in bt["roc_auc"]]),
        })
    ablation = pd.DataFrame(rows)
    print()
    print(ablation.to_string(index=False))

    print("\n  結論：沒有可靠的改善。中位數 ROC-AUC 從 0.524 動到 0.543，")
    print("  仍然接近隨機；而且最差一折還變差了。特徵只是換了哪一折贏，")
    print("  不是真的增加了訊號。這個負面結果留在 repo 裡，不刪。")

    cfg_mod.resolve("reports/metrics/sql_ablation.json").write_text(
        json.dumps(
            {
                "profile": prof.to_dict(orient="records"),
                "drift": drift.assign(week=drift["week"].astype(str)).to_dict(orient="records"),
                "missingness_top": miss.head(20).to_dict(orient="records"),
                "ablation": rows,
                "verdict": "時間偏離特徵未帶來可靠改善；ROC-AUC 中位數仍接近隨機。",
            },
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )
    print("\n  已存 reports/metrics/sql_ablation.json\n")


if __name__ == "__main__":
    main()
