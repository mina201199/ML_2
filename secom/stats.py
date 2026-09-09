"""不確定性量化。

這個檔案回答面試時的第二個問題：**「你怎麼知道這不是雜訊？」**

原本的報告只給「四折節省比例的中位數」。四個數字的中位數不是一個統計量，
它連方向都保證不了。這裡補三件事：

  1. 可偵測效果量（`min_detectable_auc`）—— 這個設計「本來就看得見」多大的訊號？
     這是最該先算的東西。33 個正樣本能偵測到的最小 AUC 約 0.64，那麼「AUC 0.497
     vs 0.5」這件事根本不構成證據，正反都不構成。
  2. 成本差的 bootstrap 區間（`bootstrap_cost_ci`）—— 給區間，不給點估計。
  3. 排序有效性的精確檢定（`permutation_test_vs_random`）—— 直接檢定
     `decision.required_lift` 指出的那個唯一門檻：同配額之下，模型挑的批次
     裡不良品濃度有沒有高於隨機抽樣。

為什麼 bootstrap 可以只靠 2x2 格數：成本是格數的線性函數
（成本 = (TP+FP)·c_inspect + FN·c_escape），所以「重抽觀測」與「重抽四格的
多項式分布」在分布上等價。因此不需要把每一筆的分數存進 JSON。
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def min_detectable_auc(
    n_pos: int, n_neg: int, alpha: float = 0.05, power: float = 0.80
) -> dict:
    """在 H0: AUC = 0.5 之下，這個樣本量最小能偵測到多大的 AUC。

    用 Mann–Whitney U 在虛無假設下的變異數
    Var(AUC) = (n1 + n2 + 1) / (12 · n1 · n2)，然後要求
    AUC - 0.5 >= (z_{1-alpha/2} + z_{power}) · SE。

    這是近似值：對立假設下的變異數與 H0 不同，且此式假設分數連續無大量同分。
    用途是判斷數量級（「0.5 附近的差異是否可能被看見」），不是精確的檢定力計算。
    """
    if n_pos <= 0 or n_neg <= 0:
        return {"n_pos": int(n_pos), "n_neg": int(n_neg), "min_detectable_auc": None}
    se = np.sqrt((n_pos + n_neg + 1) / (12.0 * n_pos * n_neg))
    delta = (norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se
    return {
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "alpha": alpha,
        "power": power,
        "se_auc_under_null": float(se),
        "min_detectable_auc": float(0.5 + delta),
        "approximation": "Mann-Whitney null variance; continuous scores assumed",
    }


def cost_resolution(n_lots: int, c_escape: float) -> dict:
    """成本指標的解析度：多攔或少攔「一筆」不良品等於每筆成本差多少。

    這是回答「+10.6% 是不是雜訊」最快的方式 —— 如果 10.6% 只等於一兩筆的差別，
    那它就不是一個穩定的結論。
    """
    if n_lots <= 0:
        return {"n_lots": int(n_lots), "twd_per_lot_per_failure": None}
    return {
        "n_lots": int(n_lots),
        "twd_per_lot_per_failure": float(c_escape / n_lots),
    }


def _cells(folds: list[dict]) -> np.ndarray:
    """把每折的結果還原成 2x2 格數：欄位順序為 TP, FP, FN, TN。"""
    out = []
    for f in folds:
        tp = int(f["caught"])
        fn = int(f["missed"])
        fp = int(f["n_flagged"]) - tp
        tn = int(f["n_eval"]) - tp - fp - fn
        if min(tp, fp, fn, tn) < 0:
            raise ValueError(f"fold {f.get('fold')} 的 2x2 格數不自洽：{tp, fp, fn, tn}")
        out.append([tp, fp, fn, tn])
    return np.asarray(out, dtype=float)


def _reference_cost(action: str, prevalence, rate, c_ins: float, c_esc: float):
    if action == "inspect_none":
        return prevalence * c_esc
    if action == "inspect_all":
        return np.full(np.shape(prevalence), float(c_ins))
    if action == "random_at_capacity":
        return rate * c_ins + (1 - rate) * prevalence * c_esc
    raise ValueError(f"未知的零模型策略：{action}")


def bootstrap_cost_ci(
    folds: list[dict],
    cfg,
    reference_action: str,
    capacity_frac: float | None = None,
    n_boot: int = 10000,
    seed: int = 42,
    conf: float = 0.95,
) -> dict:
    """對評估窗內的觀測做 bootstrap，給出成本與成本差的百分位區間。

    重抽在每一折內進行、折的大小固定，因此保留了時間窗結構；折與折之間不混合。
    零模型策略的成本也隨著同一次重抽變動（盛行率會變），所以比較是配對的。

    reference_action 必須是原始資料上事前選定的那一個，不在每次重抽裡重新挑 ——
    重新挑會把「挑選」的樂觀偏誤混進區間。
    """
    cells = _cells(folds)
    n = cells.sum(axis=1)
    c_ins, c_esc = cfg.cost.c_inspect, cfg.cost.c_escape
    rate = (np.floor(n * capacity_frac + 1e-10) / n) if capacity_frac is not None else 1.0

    rng = np.random.default_rng(seed)
    probs = cells / n[:, None]
    # 每折一次抽滿 n_boot 個重抽樣本，形狀 (n_folds, n_boot, 4)
    drawn = np.stack([
        rng.multinomial(int(n[k]), probs[k], size=n_boot) for k in range(len(n))
    ])
    tp, fp, fn = drawn[..., 0], drawn[..., 1], drawn[..., 2]
    w = n / n.sum()
    per_fold_model = ((tp + fp) * c_ins + fn * c_esc) / n[:, None]
    model = (per_fold_model * w[:, None]).sum(axis=0)
    prevalence = (tp + fn) / n[:, None]
    rate_col = rate if np.isscalar(rate) else np.asarray(rate)[:, None]
    per_fold_ref = _reference_cost(reference_action, prevalence, rate_col, c_ins, c_esc)
    reference = (per_fold_ref * w[:, None]).sum(axis=0)

    lo_q, hi_q = (1 - conf) / 2 * 100, (1 + conf) / 2 * 100
    diff = reference - model
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(reference > 0, diff / reference, np.nan)

    def ci(x):
        return [float(np.nanpercentile(x, lo_q)), float(np.nanpercentile(x, hi_q))]

    return {
        "n_boot": n_boot,
        "seed": seed,
        "conf": conf,
        "reference_action": reference_action,
        "model_cost_ci": ci(model),
        "reference_cost_ci": ci(reference),
        "cost_difference_ci": ci(diff),
        "relative_saving_ci": ci(relative),
        "share_of_resamples_model_cheaper": float(np.mean(diff > 0)),
    }


def permutation_test_vs_random(
    folds: list[dict], cfg, n_perm: int = 20000, seed: int = 42
) -> dict:
    """精確檢定：模型的排序有沒有勝過「同配額隨機抽驗」。

    虛無假設是「被標記的那一組批次與標籤無關」。在這個假設下，加驗筆數固定，
    落進標記組的不良品數服從超幾何分布 Hypergeometric(N=n_eval, K=n_fail, n=n_flagged)。
    因此不需要真的去打亂資料，直接從超幾何抽樣就是精確的置換分布。

    檢定統計量取全部評估窗的加權每筆成本（加驗筆數固定，所以成本只隨攔截數變動）。
    單尾 p 值 = P(虛無成本 <= 實測成本)，並用 (1 + 計數) / (1 + n_perm) 修正。

    這一項刻意只檢定 `decision.required_lift` 指出的那個唯一門檻 —— 在 p·R >= 1 的
    成本區域，模型要有價值的充要條件就是 lift > 1，也就是勝過同配額隨機。
    """
    cells = _cells(folds)
    n = cells.sum(axis=1)
    tp, fn = cells[:, 0], cells[:, 2]
    n_flag = cells[:, 0] + cells[:, 1]
    n_fail = tp + fn
    c_ins, c_esc = cfg.cost.c_inspect, cfg.cost.c_escape

    observed = float(np.average((n_flag * c_ins + fn * c_esc) / n, weights=n))

    rng = np.random.default_rng(seed)
    # 每折一次抽滿 n_perm 個超幾何樣本，形狀 (n_folds, n_perm)
    caught_null = np.stack([
        rng.hypergeometric(int(n_fail[k]), int(n[k] - n_fail[k]), int(n_flag[k]),
                           size=n_perm).astype(float)
        if n_flag[k] > 0 and n_fail[k] > 0 else np.zeros(n_perm)
        for k in range(len(n))
    ])
    missed_null = n_fail[:, None] - caught_null
    w = n / n.sum()
    per_fold = (n_flag[:, None] * c_ins + missed_null * c_esc) / n[:, None]
    null = (per_fold * w[:, None]).sum(axis=0)

    p_value = (1 + int(np.sum(null <= observed))) / (1 + n_perm)
    return {
        "n_perm": n_perm,
        "seed": seed,
        "observed_cost_per_lot": observed,
        "null_cost_mean": float(null.mean()),
        "null_cost_ci": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
        "p_value_one_sided": float(p_value),
        "null_hypothesis": (
            "flagged set is independent of labels "
            "(equivalent to random selection at the same quota)"
        ),
    }


def required_positives_for_auc(
    target_auc: float, prevalence: float, alpha: float = 0.05, power: float = 0.80
) -> dict:
    """反過來問：要讓 target_auc 的效果被看見，評估窗需要幾筆不良品。

    沿用 `min_detectable_auc` 的同一個近似。令 k = (1 - p) / p，則
    SE^2 ≈ (1 + k) / (12 · k · n_pos)，解出

        n_pos = (1 + k) · (z_{1-alpha/2} + z_{power})^2 / (12 · k · delta^2)

    這是面試裡「所以你需要多少資料」的答案。它同時說明為什麼本專題的結論只能是
    負面的：33 筆不良品連 AUC 0.60 都看不見，而 0.60 對製程排序來說已經算大。
    """
    delta = target_auc - 0.5
    if delta <= 0 or not 0 < prevalence < 1:
        return {"target_auc": target_auc, "required_positives": None}
    k = (1 - prevalence) / prevalence
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    n_pos = (1 + k) * z**2 / (12 * k * delta**2)
    return {
        "target_auc": target_auc,
        "prevalence": prevalence,
        "required_positives": int(np.ceil(n_pos)),
        "required_lots": int(np.ceil(n_pos / prevalence)),
    }
