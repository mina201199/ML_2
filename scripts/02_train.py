"""步驟 2：時序切分、訓練三個模型、評估、SHAP 歸因。

    python scripts/02_train.py

門檻不在這裡決定 —— 那是步驟 3 的事。這裡只負責產生「排序能力」，
決定「要不要動作」是另一個問題，刻意分開兩個檔案就是為了讓這件事講得清楚。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from secom import config as cfg_mod
from secom import data as data_mod
from secom import evaluate, models, plots
from secom.console import enable_utf8
from secom.pipeline import describe_reduction


def shap_ranking(model, X: pd.DataFrame) -> pd.DataFrame:
    """算出每個特徵的平均絕對 SHAP 值，由大到小排序。"""
    import shap

    Xt = model.preprocessor.transform(X)
    explainer = shap.TreeExplainer(model.estimator)
    values = explainer.shap_values(Xt)

    # shap 對二元分類的回傳形狀在不同版本間會變，三種都接
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., -1]

    return (
        pd.DataFrame(
            {
                "feature": Xt.columns,
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def main() -> None:
    enable_utf8()
    cfg = cfg_mod.load()
    np.random.seed(cfg.seed)

    df = data_mod.load_processed(cfg)

    print("\n[1/5] 時序切分")
    split = data_mod.time_split(df, cfg)
    summary = split.summary()
    for _, r in summary.iterrows():
        print(
            f"  {r['split']:<6} n={r['n']:>4}  fail={r['n_fail']:>3} "
            f"({r['fail_rate']:.2%})  {r['from']:%Y-%m-%d} → {r['to']:%Y-%m-%d}"
        )
    print("  切分依時間先後，train 的每一筆都早於 test 的每一筆。")

    print("\n[2/5] 訓練")
    fitted = models.fit_all(split, cfg, verbose=True)
    lgbm = fitted["lgbm"]
    print(
        "  前處理  "
        + describe_reduction(lgbm.preprocessor, lgbm.n_features_in, lgbm.n_features_out)
    )
    print(f"  LightGBM 棵數 {lgbm.best_iteration}（來源：{lgbm.meta['n_estimators_source']}，{lgbm.meta['cv']}）")
    print(f"  scale_pos_weight = {lgbm.meta['scale_pos_weight']:.2f}")

    print("\n[3/5] 評估（val 與 test）")
    val_tbl = evaluate.compare(fitted, split.X_val, split.y_val)
    test_tbl = evaluate.compare(fitted, split.X_test, split.y_test)
    pd.set_option("display.width", 150, "display.float_format", "{:.3f}".format)
    print("\n  ── val ──")
    print(val_tbl.to_string())
    print("\n  ── test（只碰一次）──")
    print(test_tbl.to_string())

    floor = float(split.y_test.mean())
    best_ap = test_tbl.loc["lgbm", "pr_auc"]
    print(
        f"\n  test 盛行率地板 {floor:.3f}；LightGBM PR-AUC {best_ap:.3f}"
        f" = 地板的 {best_ap / floor:.1f} 倍"
    )
    if best_ap <= test_tbl.loc["logreg", "pr_auc"]:
        print("  注意：LightGBM 沒有贏過線性 baseline。這件事要寫進報告，不要藏。")

    print("\n[4/5] SHAP 歸因")
    ranking = shap_ranking(lgbm, split.X_test)
    top = ranking.head(8)
    for _, r in top.iterrows():
        print(f"  {r['feature']}  {r['mean_abs_shap']:.4f}")
    print("  SECOM 欄位是匿名的，所以只能給索引。真實廠內這裡會是「站點+參數」清單。")

    print("\n[5/5] 存檔與繪圖")
    figs = cfg_mod.resolve_dir(cfg.report.figures_dir)
    plots.plot_split_timeline(split, figs)
    plots.plot_pr_curves(fitted, split.X_test, split.y_test, figs)
    plots.plot_calibration(fitted, split.X_test, split.y_test, figs)
    plots.plot_top_features(ranking, figs, cfg.report.top_n_features)

    joblib.dump(
        {"models": fitted, "split": split, "shap_ranking": ranking},
        cfg_mod.resolve("models/fitted.pkl"),
    )
    ranking.to_csv(cfg_mod.resolve("reports/metrics/shap_ranking.csv"), index=False)
    cfg_mod.resolve("reports/metrics/model_scores.json").write_text(
        json.dumps(
            {
                "split": summary.astype(str).to_dict(orient="records"),
                "val": val_tbl.to_dict(orient="index"),
                "test": test_tbl.to_dict(orient="index"),
                "lgbm_best_iteration": lgbm.best_iteration,
                "n_features_after_preprocess": lgbm.n_features_out,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("  模型已存 models/fitted.pkl")
    print("\n  下一步： python scripts/03_decide.py\n")


if __name__ == "__main__":
    main()
