"""步驟 4：滾動原點回測 —— 把「一個數字」變成「一個分布」。

    python scripts/04_backtest.py

這一步回答的是主管最後一定會問的那句：「換個時間窗還成立嗎？」
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
from secom.console import enable_utf8


def main() -> None:
    enable_utf8()
    cfg = cfg_mod.load()
    df = data_mod.load_processed(cfg)

    c_ins, c_esc = cfg.cost.c_inspect, cfg.cost.c_escape
    print(f"\n成本假設   加驗 {c_ins:,} / 流出 {c_esc:,} {cfg.cost.currency}"
          f"   比值 {c_esc / c_ins:.0f}:1")

    results = {}
    for policy in ("threshold", "quantile", "robust"):
        label = {"quantile": "分位數（標記前 k%）",
                 "robust": "分位數＋基準率保守上界",
                 "threshold": "絕對機率門檻"}[policy]
        print(f"\n[{policy}] {label}")
        bt = bt_mod.rolling_origin(df, cfg, n_folds=4, policy=policy, verbose=True)
        if bt.empty:
            print("  沒有可用的折")
            continue
        s = bt_mod.summarise(bt)
        results[policy] = {"summary": s, "folds": bt.to_dict(orient="records")}
        print(
            f"  → 中位數 {s['saving_median']:+.1%}   "
            f"範圍 {s['saving_min']:+.1%} ~ {s['saving_max']:+.1%}   "
            f"{s['folds_profitable']}/{s['n_folds']} 折省錢   "
            f"ROC-AUC 中位數 {s['roc_auc_median']:.3f}"
        )

    if not results:
        print("\n無法完成回測。")
        return

    # ── 三種策略對照 ──────────────────────────────────────────────
    print("\n=== 三種決策參數化的穩定度對照 ===")
    cmp_rows = []
    for policy, r in results.items():
        s = r["summary"]
        cmp_rows.append({
            "策略": {"quantile": "分位數", "robust": "分位數+保守上界",
                     "threshold": "絕對門檻"}[policy],
            "節省中位數": f"{s['saving_median']:+.1%}",
            "最差一折": f"{s['saving_min']:+.1%}",
            "最好一折": f"{s['saving_max']:+.1%}",
            "省錢折數": f"{s['folds_profitable']}/{s['n_folds']}",
            "攔截率中位數": f"{s['catch_rate_median']:.0%}",
        })
    print(pd.DataFrame(cmp_rows).to_string(index=False))

    best = max(results, key=lambda p: results[p]["summary"]["saving_median"])
    names = {"quantile": "分位數策略", "robust": "分位數＋基準率保守上界",
             "threshold": "絕對機率門檻"}
    print(f"\n  跨時間窗最穩的是：{names[best]}")

    worst = results[best]["summary"]["saving_min"]
    if worst <= 0:
        print(f"  但最差的一折是 {worst:+.1%} —— 這個策略不是每個時期都賺錢，"
              "報告要照實寫。")
    else:
        print(f"  而且最差的一折仍有 {worst:+.1%} —— 每一折都省錢。")

    # ── 圖表與存檔 ────────────────────────────────────────────────
    figs = cfg_mod.resolve_dir(cfg.report.figures_dir)
    plots.plot_backtest(
        pd.DataFrame(results[best]["folds"]), figs,
        policy_label={"quantile": "quantile policy", "robust": "robust policy (conservative base rate)",
                      "threshold": "absolute threshold"}[best],
        currency=cfg.cost.currency,
    )

    cfg_mod.resolve("reports/metrics/backtest.json").write_text(
        json.dumps(
            {"cost_assumptions": {"c_inspect": c_ins, "c_escape": c_esc},
             "best_policy": best, "results": results},
            indent=2, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )
    print("\n  已存 reports/metrics/backtest.json")
    print("  下一步： python scripts/05_sql_report.py\n")


if __name__ == "__main__":
    main()
