"""決策層 —— 這個專題真正的差異化就在這個檔案。

模型輸出機率之後，還有一步沒做完：**要不要對這一批下加驗？**
0.5 這個門檻是套件的預設值，不是產線的答案。誤攔一顆好品跟誤放一顆壞品的
代價差了一個數量級，所以最佳門檻一定不在 0.5。

成本模型
--------
對每一批，模型給出 fail 機率；超過門檻 t 就標記加驗。

    總成本(t) = 標記數(t) × c_inspect + 漏掉的壞品數(t) × c_escape

    標記數        = TP + FP，每一批都要付加驗成本，不論它真的壞不壞
    漏掉的壞品數  = FN，沒攔下來就流到客戶端
    真陰性        = 0 成本（正常放行）

三種策略互相比較
----------------
    不檢      門檻 > 1，全部放行         成本 = 全部壞品 × c_escape
    全檢      門檻 ≤ 0，全部加驗         成本 = 全部批次 × c_inspect
    模型導向  門檻 = t*（在 val 上選）   成本 = 上面的公式

只有在「模型導向」同時打敗「不檢」和「全檢」時，這個模型才有商業價值。
很多論文級的模型過不了這一關 —— 而這正是面試時最值得講的一段。

門檻一律在 val 上選，只在 test 上報告。在 test 上調門檻等於在考卷上作答後改題目。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _counts(y_true: np.ndarray, y_score: np.ndarray, t: float) -> tuple[int, int, int]:
    """回傳 (標記數, 漏掉的壞品數, 攔到的壞品數)。"""
    flag = y_score >= t
    n_flag = int(flag.sum())
    caught = int((flag & (y_true == 1)).sum())
    missed = int(((~flag) & (y_true == 1)).sum())
    return n_flag, missed, caught


def cost_at(
    y_true, y_score, t: float, c_inspect: float, c_escape: float
) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    n_flag, missed, caught = _counts(y_true, y_score, t)
    total = n_flag * c_inspect + missed * c_escape
    n_fail = int(y_true.sum())
    return {
        "threshold": float(t),
        "n": n,
        "n_flagged": n_flag,
        "flag_rate": n_flag / n,
        "caught": caught,
        "missed": missed,
        "catch_rate": caught / n_fail if n_fail else float("nan"),
        "inspect_cost": n_flag * c_inspect,
        "escape_cost": missed * c_escape,
        "total_cost": total,
        "cost_per_lot": total / n,
    }


def cost_curve(y_true, y_score, c_inspect: float, c_escape: float) -> pd.DataFrame:
    """掃過所有候選門檻，算出整條成本曲線。畫出來就是主圖。"""
    y_score = np.asarray(y_score)
    candidates = np.unique(np.concatenate([[0.0], y_score, [1.0 + 1e-9]]))
    return pd.DataFrame(
        [cost_at(y_true, y_score, t, c_inspect, c_escape) for t in candidates]
    )


def optimal_threshold(
    y_true,
    y_score,
    c_inspect: float,
    c_escape: float,
    capacity_frac: float | None = None,
) -> dict:
    """找出總成本最低的門檻。

    capacity_frac 有給的話，只考慮標記比例不超過產能上限的門檻 ——
    這是產線真正的約束：一天就是只驗得動那麼多批。
    """
    curve = cost_curve(y_true, y_score, c_inspect, c_escape)
    feasible = curve
    if capacity_frac is not None:
        feasible = curve[curve["flag_rate"] <= capacity_frac]
        if feasible.empty:                     # 連「全部不驗」都超標時的保險
            feasible = curve.nlargest(1, "threshold")
    best = feasible.loc[feasible["total_cost"].idxmin()]
    return best.to_dict()


def policy_table(
    y_true, y_score, t_star: float, c_inspect: float, c_escape: float
) -> pd.DataFrame:
    """三種策略並排比較。這張表就是履歷 bullet 的來源。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n, n_fail = len(y_true), int(y_true.sum())

    rows = [
        {
            "policy": "不檢（放行全部）",
            **cost_at(y_true, y_score, 1.0 + 1e-9, c_inspect, c_escape),
        },
        {
            "policy": "全檢（100% 加驗）",
            **cost_at(y_true, y_score, 0.0, c_inspect, c_escape),
        },
        {
            "policy": f"模型導向（t*={t_star:.4f}）",
            **cost_at(y_true, y_score, t_star, c_inspect, c_escape),
        },
    ]
    df = pd.DataFrame(rows).set_index("policy")

    baseline_best = df.loc[
        ["不檢（放行全部）", "全檢（100% 加驗）"], "total_cost"
    ].min()
    df["vs_最佳基準"] = df["total_cost"] - baseline_best
    df["節省比例"] = -df["vs_最佳基準"] / baseline_best

    df.attrs["n"] = n
    df.attrs["n_fail"] = n_fail
    return df


def threshold_for_flag_rate(y_score, flag_rate: float) -> float:
    """回傳「剛好標記分數最高的 flag_rate 比例」所需的門檻。"""
    y_score = np.asarray(y_score)
    if flag_rate <= 0:
        return float(y_score.max() + 1e-9)
    if flag_rate >= 1:
        return float(y_score.min())
    return float(np.quantile(y_score, 1.0 - flag_rate))


def optimal_flag_rate(
    y_true,
    y_score,
    c_inspect: float,
    c_escape: float,
    capacity_frac: float | None = None,
    grid: int = 101,
) -> dict:
    """分位數策略：選最佳的「標記比例」，而不是最佳的絕對機率門檻。

    為什麼要有這個版本：良率會漂移（本資料 7 月 22% fail → 10 月 1.8%）。
    絕對門檻 p ≥ 0.035 綁在某個時期的基準率上，基準率一變就失準；
    「驗分數最高的前 20%」則是相對規則，跨基準率轉移得穩得多，
    而且產線的產能約束本來就是用比例表達的。

    面試時這是一個很好的對照：同一個模型，兩種決策參數化，哪一種比較耐用。
    """
    y_score = np.asarray(y_score)
    hi = capacity_frac if capacity_frac is not None else 1.0
    rates = np.linspace(0.0, hi, grid)

    best, best_cost = None, np.inf
    for r in rates:
        t = threshold_for_flag_rate(y_score, r)
        row = cost_at(y_true, y_score, t, c_inspect, c_escape)
        row["target_flag_rate"] = float(r)
        if row["total_cost"] < best_cost:
            best, best_cost = row, row["total_cost"]
    return best


def sensitivity(
    y_true,
    y_score,
    ratios=(1, 2, 5, 10, 20, 30, 50, 100, 200),
    capacity_frac: float | None = None,
) -> pd.DataFrame:
    """損益兩平分析：成本比 r = c_escape / c_inspect 在什麼區間，模型才划算？

    這一段回答的是面試官心裡真正的問題：「你怎麼知道這在我們公司也成立？」
    答案是不知道，但你可以給出一個區間，讓對方拿自己的成本數字去對。
    成本以 c_inspect 為單位正規化，所以只有比值有意義。
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n, n_fail = len(y_true), int(y_true.sum())

    rows = []
    for r in ratios:
        best = optimal_threshold(y_true, y_score, 1.0, r, capacity_frac)
        c_none, c_all = r * n_fail, float(n)
        c_model = best["total_cost"]
        baseline = min(c_none, c_all)
        rows.append(
            {
                "cost_ratio": r,
                "t*": best["threshold"],
                "flag_rate": best["flag_rate"],
                "catch_rate": best["catch_rate"],
                "cost_不檢": c_none,
                "cost_全檢": c_all,
                "cost_模型": c_model,
                "節省比例": (baseline - c_model) / baseline,
                "模型勝出": bool(c_model < baseline),
            }
        )
    return pd.DataFrame(rows).set_index("cost_ratio")


def breakeven_ratio(
    y_true, y_score, lo: float = 1.0, hi: float = 500.0, capacity_frac=None
) -> float | None:
    """二分搜出模型開始打敗兩個基準的最小成本比。找不到就回 None。"""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n, n_fail = len(y_true), int(y_true.sum())

    def wins(r: float) -> bool:
        best = optimal_threshold(y_true, y_score, 1.0, r, capacity_frac)
        return best["total_cost"] < min(r * n_fail, float(n))

    if not wins(hi):
        return None
    if wins(lo):
        return lo
    for _ in range(60):
        mid = (lo + hi) / 2
        if wins(mid):
            hi = mid
        else:
            lo = mid
    return hi
