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
    y_true, y_score, t_star: float, c_inspect: float, c_escape: float,
    *, y_select, capacity_frac: float | None = None,
) -> pd.DataFrame:
    """Show model costs and reference policies.

    Select the feasible baseline using y_select only. Random-capacity costs are analytical
    expectations for a fixed integer quota.
    """
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

    rate = np.floor(n * capacity_frac + 1e-10) / n if capacity_frac is not None else 1.0
    select_rate = (
        np.floor(len(y_select) * capacity_frac + 1e-10) / len(y_select)
        if capacity_frac is not None else 1.0
    )
    df['feasible'] = df.flag_rate <= rate + 1e-12
    alternative = '全檢（100% 加驗）'
    if rate < 1:
        alternative = f'隨機加驗 {rate:.0%}（期望值）'
        total = n * rate * c_inspect + n_fail * (1 - rate) * c_escape
        df.loc[alternative] = dict(threshold=float('nan'), n=n,
            n_flagged=n*rate, flag_rate=rate, caught=n_fail*rate,
            missed=n_fail*(1-rate), catch_rate=rate if n_fail else float('nan'),
            inspect_cost=n*rate*c_inspect, escape_cost=n_fail*(1-rate)*c_escape,
            total_cost=total, cost_per_lot=total/n, feasible=True)
    select_escape = np.mean(y_select) * c_escape
    choose_inspect = (
        select_rate * c_inspect + (1 - select_rate) * select_escape < select_escape
    )
    baseline_name = alternative if choose_inspect else '不檢（放行全部）'
    baseline_best = df.loc[baseline_name, 'total_cost']
    df["vs_最佳基準"] = df["total_cost"] - baseline_best
    df["節省比例"] = -df["vs_最佳基準"] / baseline_best if baseline_best else float('nan')
    df.attrs['baseline_name'] = baseline_name

    df.attrs["n"] = n
    df.attrs["n_fail"] = n_fail
    return df


def threshold_for_flag_rate(y_score, flag_rate: float) -> float:
    """At most floor(n * rate) items; boundary ties are excluded together."""
    y_score = np.asarray(y_score)
    if not len(y_score) or not 0 <= flag_rate <= 1:
        raise ValueError('Scores must be nonempty and flag_rate must be in [0, 1].')
    if flag_rate <= 0:
        return float(y_score.max() + 1e-9)
    if flag_rate >= 1:
        return float(y_score.min())
    count = int(np.floor(len(y_score) * flag_rate + 1e-10))
    ordered = np.sort(y_score)[::-1]
    return float(np.nextafter(ordered[count], np.inf))


def optimal_flag_rate(
    y_true,
    y_score,
    c_inspect: float,
    c_escape: float,
    capacity_frac: float | None = None,
    grid: int = 101,
) -> dict:
    """Select a batch inspection fraction using labeled calibration observations.

    Ranking across a future batch requires all its scores to be available together; it is not an
    online threshold.
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


def conservative_base_rate(n_fail: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper bound under a binomial assumption.

    This does not guarantee coverage of a future, drifting population.
    """
    from scipy.stats import beta

    if n == 0:
        return 1.0
    if n_fail >= n:
        return 1.0
    return float(beta.ppf(conf, n_fail + 1, n - n_fail))


def robust_flag_rate(
    y_true,
    y_score,
    c_inspect: float,
    c_escape: float,
    capacity_frac: float | None = None,
    conf: float = 0.95,
    grid: int = 101,
) -> dict:
    """Heuristic cost inflation from base-rate uncertainty.

    With zero observed failures, use homogeneous upper risk to choose an endpoint. This is not a
    distributionally robust cost guarantee.
    """
    y_true = np.asarray(y_true)
    n, n_fail = len(y_true), int(y_true.sum())
    p_hat = n_fail / n if n else 0.0
    p_up = conservative_base_rate(n_fail, n, conf)
    lam = (p_up / p_hat) if p_hat > 0 else 1.0

    if n_fail == 0:
        # No observed escapes does not imply zero future risk. With no ranking
        # evidence, use a homogeneous upper-risk scenario and choose an endpoint.
        cap = capacity_frac if capacity_frac is not None else 1.0
        rate = cap if p_up * c_escape > c_inspect else 0.0
        best = cost_at(y_true, y_score, threshold_for_flag_rate(y_score, rate), c_inspect, c_escape)
        best['target_flag_rate'] = rate
    else:
        best = optimal_flag_rate(
            y_true, y_score, c_inspect, c_escape * lam, capacity_frac, grid
        )
    best = dict(best)
    best["base_rate_observed"] = p_hat
    best["base_rate_upper"] = p_up
    best["escape_inflation"] = lam
    return best


def required_lift(prevalence: float, cost_ratio: float) -> float:
    """模型排序必須達到的 lift@r，才可能打敗最便宜的零模型策略。

    設加驗成本為 1 單位、漏放成本為 R 倍、盛行率 p、加驗比例 r、
    模型在前 r 比例裡攔到的不良品比例為 recall(r)：

        模型成本      = r + (1 - recall) · p · R
        不檢           = p · R
        等額隨機加驗   = r + (1 - r) · p · R      （r = 1 時就是全檢）

    當 p·R > 1（平均而言加驗划算），等額隨機一定比不檢便宜，於是門檻化簡成
    recall > r，也就是 **lift > 1**：模型唯一的工作是打敗同配額的隨機抽驗。
    當 p·R < 1（加驗不划算），門檻是不檢，模型必須做到 lift > 1/(p·R)。

    合起來就是 max(1, 1/(p·R))，且與 r 無關 —— 產能上限只改變「哪一個零模型
    策略在把關」，不改變門檻高度。

    注意適用範圍：p·R < 1 時，這個門檻只在 r ≤ p·R 的加驗比例下成立。r 一旦
    超過 p·R，光是加驗帳單就已經大於「完全不檢」的總成本，再完美的排序也贏不了。
    p·R ≥ 1 時所有 r 都適用。回傳 inf 表示沒有任何 lift 能讓加驗划算。
    """
    pr = prevalence * cost_ratio
    if pr <= 0:
        return float("inf")
    return max(1.0, 1.0 / pr)


def sensitivity(
    y_true,
    y_score,
    *,
    y_select,
    p_select,
    ratios=(1, 2, 5, 10, 20, 30, 50, 100, 200),
    capacity_frac: float | None = None,
) -> pd.DataFrame:
    """For each ratio, select the threshold and feasible baseline on
    y_select/p_select; report costs on y_true/y_score.

    No monotonic break-even claim is made.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n, n_fail = len(y_true), int(y_true.sum())

    rows = []
    for r in ratios:
        selected = optimal_threshold(y_select, p_select, 1.0, r, capacity_frac)
        threshold = selected['threshold']
        if capacity_frac is not None:
            threshold = max(threshold, threshold_for_flag_rate(y_score, capacity_frac))
        best = cost_at(y_true, y_score, threshold, 1.0, r)
        c_none, c_all = r * n_fail, float(n)
        c_model = best["total_cost"]
        rate = np.floor(n * capacity_frac + 1e-10) / n if capacity_frac is not None else 1.0
        select_rate = (
            np.floor(len(y_select) * capacity_frac + 1e-10) / len(y_select)
            if capacity_frac is not None else 1.0
        )
        alternative = n * rate + (1 - rate) * c_none
        choose_inspect = select_rate + (1-select_rate)*np.mean(y_select)*r < np.mean(y_select)*r
        baseline = alternative if choose_inspect else c_none
        rows.append(
            {
                "cost_ratio": r,
                "t*": best["threshold"],
                "flag_rate": best["flag_rate"],
                "catch_rate": best["catch_rate"],
                "cost_不檢": c_none,
                "cost_全檢": c_all,
                "cost_模型": c_model,
                "cost_基準": baseline,
                "節省比例": (baseline - c_model) / baseline if baseline else float('nan'),
                "模型勝出": bool(c_model < baseline),
            }
        )
    return pd.DataFrame(rows).set_index("cost_ratio")
