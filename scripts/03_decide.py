"""步驟 3：把機率翻譯成「要不要加驗」的決策，並在測試窗上計價。

    python scripts/03_decide.py

門檻一律在驗證窗選定，只在測試窗報告。成本比掃描也是每個比值都重新在驗證窗
選一次門檻 —— 在測試窗上調門檻等於在考卷上作答後改題目。

注意這一支只處理**單次時間切分**，它是對照組，不是結論來源。單一時間窗只有
17 個正樣本，任何正號都在滾動回測顯示的折間變異範圍之內。主結論請看
scripts/04_backtest.py。
"""

from __future__ import annotations

import joblib

from secom import config, decision, evaluate, plots
from secom.console import enable_utf8
from secom.provenance import write_json
from secom.reporting import write_summary

REGIMES = [("unconstrained", None), ("capacity_capped", "from_config")]
SHOWN_COLUMNS = ["flag_rate", "missed", "cost_per_lot", "feasible", "節省比例"]


def main() -> None:
    enable_utf8()
    cfg = config.load()

    bundle = joblib.load(config.resolve("models/fitted.pkl"))
    model, split = bundle["models"]["lgbm"], bundle["split"]
    p_val = model.predict_proba(split.X_val)
    p_test = model.predict_proba(split.X_test)
    c_ins, c_esc = cfg.cost.c_inspect, cfg.cost.c_escape

    cases = {}
    for name, cap_spec in REGIMES:
        cap = cfg.cost.inspect_capacity_frac if cap_spec == "from_config" else None

        chosen = decision.optimal_threshold(split.y_val, p_val, c_ins, c_esc, cap)
        threshold = chosen["threshold"]
        if cap is not None:
            # 驗證窗選出的門檻在測試窗上可能超出產能，取較嚴格的那一個
            threshold = max(threshold, decision.threshold_for_flag_rate(p_test, cap))

        table = decision.policy_table(
            split.y_test, p_test, threshold, c_ins, c_esc,
            y_select=split.y_val, capacity_frac=cap,
        )
        sens = decision.sensitivity(
            split.y_test, p_test,
            y_select=split.y_val, p_select=p_val, capacity_frac=cap,
        )

        cases[name] = {
            "threshold_from_val": chosen,
            "applied_threshold": threshold,
            "baseline_name": table.attrs["baseline_name"],
            "test_policies": table.reset_index().to_dict(orient="records"),
            "test_at_threshold": evaluate.at_threshold(split.y_test, p_test, threshold),
            "sensitivity": sens.reset_index().to_dict(orient="records"),
        }

        print(f"\n  ── {name} ──")
        print(table[SHOWN_COLUMNS].to_string())

        if cap is None:
            figs = config.resolve_dir(cfg.report.figures_dir)
            plots.plot_sensitivity(sens, figs)
            plots.plot_cost_curve(
                decision.cost_curve(split.y_test, p_test, c_ins, c_esc),
                threshold, table, figs, cfg.cost.currency,
            )

    write_json("reports/metrics/decision.json", {
        "evaluation_protocol": (
            "validation-selected thresholds and baseline; "
            "test labels only for scoring"
        ),
        "cost_assumptions": {
            "c_inspect": c_ins,
            "c_escape": c_esc,
            "capacity_frac": cfg.cost.inspect_capacity_frac,
        },
        "cases": cases,
    })
    write_summary(include_backtest=False)
    print("\n  單次切分結果已存。主結論請執行： python scripts/04_backtest.py\n")


if __name__ == "__main__":
    main()
