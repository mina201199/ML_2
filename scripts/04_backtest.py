"""步驟 4：在完全相同的未來窗上比較模型、策略與產能情境。

    python scripts/04_backtest.py

這一支是本專題的主證據。它同時回答兩個問題：

  1. 模型導向策略在每一個部署時點各表現如何（滾動原點回測）
  2. **完全不需要模型的固定策略表現如何**（reference_costs）

第 2 點是這一版新增的，也是結論的關鍵。`cost_baseline` 是每折用校準窗事前挑
出來的基準，它會挑錯；而「永遠全檢」「永遠不檢」「按配額隨機加驗」這三個固定
策略連挑錯的機會都沒有。任何模型導向策略如果贏不過其中最便宜的那一個，就沒有
存在的理由 —— 這個判斷寫進 dominance 區塊，讓報告不必靠人去讀表。
"""

from __future__ import annotations

import pandas as pd

from secom import backtest, config, data, plots, stats
from secom.console import enable_utf8
from secom.provenance import write_json
from secom.reporting import write_summary

MODELS = ["majority", "logreg", "lgbm"]
POLICIES = ["threshold", "quantile", "robust"]

# 預先指定要畫圖與詳述的配置。刻意寫死，避免用評估標籤事後挑最好看的那一個。
DISPLAY_KEY = "unconstrained/lgbm/robust"

# 要做不確定性量化的配置也預先指定（每個情境的 LightGBM 保守策略），
# 不挑「事後看起來最好」的那一個 —— 那會把挑選偏誤混進區間與 p 值。
UNCERTAINTY_MODEL, UNCERTAINTY_POLICY = "lgbm", "robust"


def main() -> None:
    enable_utf8()
    cfg = config.load()
    df = data.load_processed(cfg)

    regimes = [
        ("unconstrained", None),
        ("capacity_capped", cfg.cost.inspect_capacity_frac),
    ]

    # 同一個 (模型, 折) 的預測在九種策略／情境組合之間共用
    cache: dict = {}
    results: dict = {}
    folds_by_regime: dict = {}

    for regime, cap in regimes:
        for model in MODELS:
            for policy in POLICIES:
                key = f"{regime}/{model}/{policy}"
                bt = backtest.rolling_origin(
                    df, cfg, model_name=model, policy=policy,
                    capacity_frac=cap, prediction_cache=cache, verbose=False,
                )
                results[key] = {
                    "summary": backtest.summarise(bt),
                    "folds": bt.to_dict(orient="records"),
                }
                folds_by_regime.setdefault(regime, bt)
                print(f"  {key:<42} 成本/筆 {results[key]['summary']['cost_per_lot']:>9.1f}",
                      flush=True)

    # ── 零模型參考策略，以及「有沒有任何配置贏過它」 ──
    reference = {}
    dominance = {}
    for regime, cap in regimes:
        ref = backtest.reference_costs(folds_by_regime[regime], cfg, cap)
        reference[regime] = ref
        beaten = {
            key: results[key]["summary"]["cost_per_lot"]
            for key in results
            if key.startswith(f"{regime}/")
            and results[key]["summary"]["cost_per_lot"] < ref["best_cost"]
        }
        cheapest = min(
            (k for k in results if k.startswith(f"{regime}/")),
            key=lambda k: results[k]["summary"]["cost_per_lot"],
        )
        dominance[regime] = {
            "best_model_free_action": ref["best_action"],
            "best_model_free_cost": ref["best_cost"],
            "cheapest_model_driven": cheapest,
            "cheapest_model_driven_cost": results[cheapest]["summary"]["cost_per_lot"],
            "n_configurations": sum(1 for k in results if k.startswith(f"{regime}/")),
            "n_beating_model_free": len(beaten),
            "configurations_beating_model_free": beaten,
        }
        print(f"\n  [{regime}] 零模型最佳 = {ref['best_action']} @ {ref['best_cost']:,.1f}")
        print(f"  [{regime}] 最便宜的模型導向 = {cheapest} @ "
              f"{results[cheapest]['summary']['cost_per_lot']:,.1f}")
        print(f"  [{regime}] 贏過零模型的配置數 = {len(beaten)} / "
              f"{dominance[regime]['n_configurations']}")

    # ── 不確定性量化（B 層）──
    uncertainty = {}
    for regime, cap in regimes:
        key = f"{regime}/{UNCERTAINTY_MODEL}/{UNCERTAINTY_POLICY}"
        folds = results[key]["folds"]
        summary = results[key]["summary"]
        n_pos = summary["eval_fails_total"]
        n_lots = summary["eval_lots_total"]
        uncertainty[key] = {
            "power": stats.min_detectable_auc(n_pos, n_lots - n_pos),
            "cost_resolution": stats.cost_resolution(n_lots, cfg.cost.c_escape),
            "bootstrap": stats.bootstrap_cost_ci(
                folds, cfg, reference[regime]["best_action"], cap
            ),
            "permutation_vs_random": stats.permutation_test_vs_random(folds, cfg),
        }
        u = uncertainty[key]
        print()
        print(f"  [{key}] 可偵測最小 AUC "
              f"{u['power']['min_detectable_auc']:.3f}（{n_pos} 正樣本）")
        print(f"  [{key}] 相對節省 95% CI "
              f"{u['bootstrap']['relative_saving_ci'][0]:+.1%} ~ "
              f"{u['bootstrap']['relative_saving_ci'][1]:+.1%}")
        print(f"  [{key}] 勝過同配額隨機的置換檢定 p = "
              f"{u['permutation_vs_random']['p_value_one_sided']:.3f}")

    write_json("reports/metrics/backtest.json", {
        "evaluation_protocol": (
            "expanding train; calibration selects policy and feasible baseline; "
            "all future windows retained; reference_policies are fixed rules that "
            "use no scores at all"
        ),
        "cost_assumptions": {
            "c_inspect": cfg.cost.c_inspect,
            "c_escape": cfg.cost.c_escape,
            "capacity_frac": cfg.cost.inspect_capacity_frac,
        },
        "reference_policies": reference,
        "dominance": dominance,
        "uncertainty": uncertainty,
        "display_key": DISPLAY_KEY,
        "results": results,
    })

    figs = config.resolve_dir(cfg.report.figures_dir)
    display = pd.DataFrame(results[DISPLAY_KEY]["folds"])
    plots.plot_backtest(display, figs, policy_label="LightGBM robust",
                        currency=cfg.cost.currency)
    plots.plot_feasibility(display, cfg, figs, model_label="LightGBM")

    write_summary()


if __name__ == "__main__":
    main()
