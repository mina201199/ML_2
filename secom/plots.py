"""圖表。

標籤刻意全部用英文 —— 這些圖會直接貼進給外商看的 executive summary 和
LinkedIn 貼文，用英文才不用重畫一次，也順便避開 matplotlib 的中文字型問題。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from .evaluate import pr_curve

INK = "#152B31"
COPPER = "#A9542C"
TEAL = "#25707F"
GREY = "#8A9A9D"
PALETTE = {"lgbm": COPPER, "logreg": TEAL, "majority": GREY}

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": "#D3DAD8",
        "grid.linewidth": 0.6,
    }
)


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  圖表 {path.name}")
    return path


def plot_pr_curves(models, X, y, out_dir: Path) -> Path:
    """PR 曲線。虛線是盛行率地板 —— 沒有超過它的模型等於沒用。"""
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    prevalence = float(np.mean(y))

    for name, m in models.items():
        if name == "majority":
            continue
        p = m.predict_proba(X)
        c = pr_curve(y, p)
        from sklearn.metrics import average_precision_score

        ap = average_precision_score(y, p)
        ax.step(
            c["recall"], c["precision"], where="post",
            color=PALETTE.get(name, INK), lw=1.9, label=f"{name}  (PR-AUC {ap:.3f})",
        )

    ax.axhline(prevalence, ls="--", lw=1.2, color=GREY,
               label=f"prevalence floor ({prevalence:.3f})")
    ax.set_xlabel("Recall (share of failures caught)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall, held-out test set")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, min(1.0, max(0.45, prevalence * 6)))
    ax.grid(alpha=0.45)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    return _save(fig, out_dir, "pr_curve.png")


def plot_cost_curve(curve: pd.DataFrame, t_star: float, policies: pd.DataFrame,
                    out_dir: Path, currency: str = "TWD") -> Path:
    """主圖：成本 vs 加驗比例，標出最佳點與兩個基準策略。

    這是整個作品集裡最重要的一張圖。它把「模型分數」翻譯成「錢」。

    x 軸刻意用「加驗比例」而不是「機率門檻」：
      1. 校準後的機率只跨 0.027~0.054，畫在 0~1 的軸上有九成面積是平的
      2. 主管關心的是「我要驗幾成的批次」，門檻是實作細節
    門檻值改放在標註裡當次要資訊。
    """
    fig, ax = plt.subplots(figsize=(6.6, 4.3))

    # 依加驗比例排序後畫，同一比例取成本最低者（門檻不同但比例相同時）
    c = (curve.sort_values("flag_rate")
              .groupby("flag_rate", as_index=False)["cost_per_lot"].min())
    ax.plot(c["flag_rate"] * 100, c["cost_per_lot"], color=COPPER, lw=2.1,
            label="Model-guided policy")

    c_none = policies.loc[policies.index.str.startswith("不檢"), "cost_per_lot"].iloc[0]
    c_all = policies.loc[policies.index.str.startswith("全檢"), "cost_per_lot"].iloc[0]
    ax.axhline(c_none, ls="--", lw=1.3, color=GREY,
               label=f"Inspect nothing ({c_none:,.0f})")
    ax.axhline(c_all, ls=":", lw=1.6, color=TEAL,
               label=f"Inspect everything ({c_all:,.0f})")

    # 標註用政策表裡那一列的真實數字，確保圖與表完全一致
    best = policies[policies.index.str.startswith("模型導向")].iloc[0]
    bx, by = best["flag_rate"] * 100, best["cost_per_lot"]
    ax.scatter([bx], [by], s=72, zorder=5, color=COPPER,
               edgecolor="white", linewidth=1.5)
    ax.annotate(
        f"optimum: inspect {best['flag_rate']:.0%} of lots\n"
        f"{by:,.0f} {currency}/lot  ·  catch {best['catch_rate']:.0%}\n"
        f"(threshold t*={t_star:.4f})",
        xy=(bx, by), xytext=(-26, 96), textcoords="offset points", fontsize=8.5,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=COPPER, lw=1.0),
        arrowprops=dict(arrowstyle="-", color=COPPER, lw=1.0),
    )

    ax.set_xlabel("Share of lots sent for extra inspection (%)")
    ax.set_ylabel(f"Expected cost per lot ({currency})")
    ax.set_title("Cost-optimal inspection rate")
    ax.set_xlim(-2, 102)
    ax.set_ylim(min(c["cost_per_lot"].min(), c_all) * 0.94,
                max(c["cost_per_lot"].max(), c_none) * 1.10)
    ax.grid(alpha=0.45)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    return _save(fig, out_dir, "cost_curve.png")


def plot_calibration(models, X, y, out_dir: Path, n_bins: int = 8) -> Path:
    """校準圖。成本計算會拿機率去乘損失，機率不準的話那個乘法沒有意義。"""
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.1, color=GREY, label="perfectly calibrated")

    for name, m in models.items():
        if name == "majority":
            continue
        p = m.predict_proba(X)
        try:
            frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        except ValueError:
            continue
        ax.plot(mean_pred, frac, "o-", ms=4.5, lw=1.6,
                color=PALETTE.get(name, INK), label=name)

    ax.set_xlabel("Mean predicted P(fail)")
    ax.set_ylabel("Observed failure rate")
    ax.set_title("Calibration (test set)")
    ax.grid(alpha=0.45)
    ax.legend(frameon=False, fontsize=8.5)
    return _save(fig, out_dir, "calibration.png")


def plot_top_features(shap_df: pd.DataFrame, out_dir: Path, top_n: int = 20) -> Path:
    """SHAP 平均絕對值前 N 名感測器。

    SECOM 的欄位是匿名的（f000..f589），所以這裡只能給索引，給不出物理意義。
    在真實廠內，這一步的產出就是「站點 + 參數」的清單，會直接交給製程工程師。
    """
    top = shap_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.6, max(3.2, 0.28 * len(top) + 1.1)))
    ax.barh(top["feature"], top["mean_abs_shap"], color=TEAL, height=0.72)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {len(top)} sensors driving predicted failure")
    ax.grid(axis="x", alpha=0.45)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=8.5)
    return _save(fig, out_dir, "top_features.png")


def plot_sensitivity(sens: pd.DataFrame, out_dir: Path,
                     breakeven: float | None = None) -> Path:
    """損益兩平圖：成本比要多高，模型才打敗全檢與不檢。

    breakeven 由 decision.breakeven_ratio 二分搜出（連續值）。不傳的話會退回
    用離散網格中第一個勝出的點 —— 那個值會偏大，跟報告裡的數字對不起來。
    """
    fig, ax = plt.subplots(figsize=(6.0, 4.1))
    x = sens.index.values
    ax.plot(x, sens["cost_不檢"], ls="--", lw=1.4, color=GREY, label="Inspect nothing")
    ax.plot(x, sens["cost_全檢"], ls=":", lw=1.7, color=TEAL, label="Inspect everything")
    ax.plot(x, sens["cost_模型"], lw=2.1, color=COPPER, label="Model-guided")

    win = sens[sens["模型勝出"]]
    start = breakeven if breakeven is not None else (
        win.index.min() if not win.empty else None
    )
    if start is not None:
        ax.axvspan(start, x.max(), color=COPPER, alpha=0.07)
        ax.annotate(
            f"model wins from\nratio ≈ {start:.1f} : 1",
            xy=(start, ax.get_ylim()[1] * 0.55),
            xytext=(8, 0), textcoords="offset points", fontsize=8.5, color=COPPER,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Cost ratio  (escape cost ÷ inspection cost)")
    ax.set_ylabel("Total cost (units of inspection cost)")
    ax.set_title("Break-even sensitivity")
    ax.grid(alpha=0.4, which="both")
    ax.legend(frameon=False, fontsize=8.5)
    return _save(fig, out_dir, "sensitivity.png")


def plot_split_timeline(split, out_dir: Path) -> Path:
    """時序切分示意圖。用來向面試官證明你沒有隨機切分。"""
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    for name, color in (("train", TEAL), ("val", GREY), ("test", COPPER)):
        ts = getattr(split, f"ts_{name}")
        y = getattr(split, f"y_{name}")
        ax.scatter(ts, np.full(len(ts), 0), s=6, color=color, alpha=0.35)
        fails = ts[y == 1]
        ax.scatter(fails, np.full(len(fails), 0.35), s=18, color=color,
                   marker="|", linewidth=1.4,
                   label=f"{name}: n={len(ts)}, fail={int(y.sum())}")
    ax.set_yticks([0, 0.35], ["all lots", "failures"], fontsize=8.5)
    ax.set_ylim(-0.25, 0.7)
    ax.set_title("Chronological split — no future information leaks into training")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", alpha=0.4)
    return _save(fig, out_dir, "split_timeline.png")
