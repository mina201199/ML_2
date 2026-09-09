"""步驟 5：SQL 資料側寫、漂移查詢，以及無洩漏的歷史特徵消融。

    python scripts/05_sql_report.py

消融實驗刻意對**全部 590 個感測器**都產生歷史偏離特徵，不用 SHAP 排名挑特徵 ——
用測試集的 SHAP 排名去選特徵，等於用答案挑題目，那是舊版的錯誤。

代價是維度從 590 變成 1,770，這一支因此是整條流程裡最慢的。這裡沒有用 04 的
預測快取，因為快取幫不上忙：每個特徵集只跑一次 rolling_origin、單一策略，
沒有重複的 (模型, 折) 可以重用。真正的成本在 config.yaml 的 model.lgbm.n_estimators
—— 內層 CV 每折都要跑滿那個上限才能取得完整的 AUC 曲線。
"""

from __future__ import annotations

import pandas as pd

from secom import backtest, config, data, drift, plots, sql
from secom.console import enable_utf8
from secom.provenance import write_json
from secom.reporting import write_summary

ABLATION_POLICY = "robust"

FEATURE_PROTOCOL = (
    "All 590 raw sensors; preceding 20 rows; no SHAP-based selection. "
    "Same-timestamp ties use input sequence."
)
LIMITATION = (
    "Exploratory ablation on reused evaluation windows; "
    "not independent confirmation."
)


def main() -> None:
    enable_utf8()
    cfg = config.load()
    df = data.load_processed(cfg)

    print("\n[1/4] SQL 側寫與漂移")
    with sql.connect(cfg) as con:
        profile, drift_weekly = sql.profile(con), sql.drift(con)
    plots.plot_drift(drift_weekly, config.resolve_dir(cfg.report.figures_dir))

    print("\n[2/4] 產生全感測器歷史特徵")
    augmented = sql.augment_time_features(df)
    print(f"  {len(data.feature_cols(df))} 維 -> "
          f"{len(data.feature_cols(augmented))} 維")

    print("\n[3/4] 消融回測")
    rows = {}
    for label, frame in [("raw", df), ("all_sensor_history", augmented)]:
        bt = backtest.rolling_origin(
            frame, cfg, policy=ABLATION_POLICY, verbose=False,
        )
        summary = backtest.summarise(bt)
        rows[label] = {
            "n_features": len(data.feature_cols(frame)),
            "summary": summary,
            "folds": bt.to_dict(orient="records"),
        }
        print(f"  {label:<20} {rows[label]['n_features']:>5} 維  "
              f"成本/筆 {summary['cost_per_lot']:>9.1f}  "
              f"加驗率中位數 {summary['flag_rate_median']:>6.1%}  "
              f"漏放 {summary['total_missed']:>2}", flush=True)

    # ── 漂移偵測與重訓觸發（D3）──
    # 參考窗固定用第一折的訓練窗：模擬「模型在 T 時點被訓練並部署」，
    # 然後逐一問「在每一個後續評估窗，這個部署還處在同一個決策體制裡嗎」。
    print("\n[4/4] 漂移偵測與重訓觸發")
    folds = pd.DataFrame(rows["raw"]["folds"])
    reference = df[df["ts"] <= folds["train_end"].iloc[0]]
    alerts = []
    for _, fold in folds.iterrows():
        current = df[(df["ts"] >= fold["eval_from"]) & (df["ts"] <= fold["eval_to"])]
        trigger = drift.retraining_trigger(reference, current, cfg)
        trigger["fold"] = int(fold["fold"])
        trigger["eval_from"] = str(fold["eval_from"])
        trigger["eval_to"] = str(fold["eval_to"])
        alerts.append(trigger)
        flag = "觸發重訓" if trigger["retrain"] else "維持"
        print(f"  fold {trigger['fold']}  {fold['eval_from']:%m/%d}–{fold['eval_to']:%m/%d}  "
              f"盛行率體制 {trigger['current_regime']['regime']:<11} "
              f"PSI>0.25 的特徵 {trigger['psi_features_major']:>3}/"
              f"{trigger['psi_features_usable']:<3}  {flag}")

    write_json("reports/metrics/sql_ablation.json", {
        "feature_protocol": FEATURE_PROTOCOL,
        "profile": profile.to_dict(orient="records"),
        "drift": drift_weekly.to_dict(orient="records"),
        "drift_monitoring": {
            "rule": (
                "retrain when the prevalence Clopper-Pearson interval moves to the "
                "other side of the break-even prevalence c_inspect/c_escape, or when "
                "more than 10% of usable features have PSI > 0.25"
            ),
            "breakeven_prevalence": drift.breakeven_prevalence(cfg),
            "reference_window": {
                "n": int(len(reference)),
                "n_fail": int(reference["fail"].sum()),
                "to": str(folds["train_end"].iloc[0]),
            },
            "alerts": alerts,
        },
        "ablation": rows,
        "limitation": LIMITATION,
    })
    write_summary(include_ablation=True)
    print("\n  消融結果已存，報告已更新。\n")


if __name__ == "__main__":
    main()
