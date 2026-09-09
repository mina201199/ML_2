"""漂移偵測與重訓觸發規則。

為什麼需要這個檔案
------------------
本專題的整個論點是「良率會隨時間漂移」—— 這是決定用時序切分、決定做滾動回測的
理由。但在這一版之前，漂移只有一張描述性的圖（`plots.plot_drift`），沒有任何
可執行的規則。面試一定會問「上線之後你監控什麼、什麼時候重訓」，而「我畫了一張
每週不良率的圖」不是答案。

門檻要綁在哪裡
--------------
一般的漂移偵測會說「分布變了就告警」。那在這裡是錯的關注點 —— 分布天天在變，
而我們真正在意的是**最佳動作有沒有翻轉**。

由 `decision.required_lift` 的推導：加驗是否划算的分界在 p·R = 1，也就是

    損益兩平盛行率  p* = c_inspect / c_escape

盛行率在 p* 之上，零模型的最佳動作是「按配額加驗」；在 p* 之下是「完全不檢」。
所以有意義的告警條件是「有證據顯示盛行率已經跨過 p*」，不是「KS 統計量抖動了」。
本專題的 p* = 2,000 / 60,000 = 3.33%，而觀測到的每週不良率從 22% 掃到 1.8% ——
這條線是真的會被跨過的，不是假設性的。

特徵漂移用 PSI 當輔助訊號：它不決定動作，只回答「模型的輸入還像不像訓練時的樣子」。
缺值自成一個桶，因為在製程資料裡「某站點當時沒量到」本身帶訊息（這也是主模型
刻意不填補缺值的理由）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import beta

# PSI 的慣用判讀門檻
PSI_MINOR = 0.10
PSI_MAJOR = 0.25

EPS = 1e-6


def breakeven_prevalence(cfg) -> float:
    """加驗開始划算的盛行率 p* = c_inspect / c_escape。"""
    return cfg.cost.c_inspect / cfg.cost.c_escape


def prevalence_regime(n_fail: int, n: int, breakeven: float,
                      conf: float = 0.95) -> dict:
    """盛行率落在損益兩平線的哪一側，用 Clopper–Pearson 區間判斷。

    回傳的 regime：
        "inspect"    區間整段在 p* 之上 —— 有證據該加驗
        "no_inspect" 區間整段在 p* 之下 —— 有證據不該加驗
        "undecided"  區間跨過 p* —— 這個窗的資料不足以決定動作

    "undecided" 是刻意保留的第三種答案。小窗 + 稀有事件之下它會是常態，
    而把它硬歸類成前兩者就是在假裝知道。
    """
    if n <= 0:
        return {"n": 0, "n_fail": 0, "regime": "undecided"}
    lo = float(beta.ppf((1 - conf) / 2, n_fail, n - n_fail + 1)) if n_fail else 0.0
    hi = float(beta.ppf((1 + conf) / 2, n_fail + 1, n - n_fail)) if n_fail < n else 1.0

    if lo > breakeven:
        regime = "inspect"
    elif hi < breakeven:
        regime = "no_inspect"
    else:
        regime = "undecided"
    return {
        "n": int(n),
        "n_fail": int(n_fail),
        "prevalence": n_fail / n,
        "ci": [lo, hi],
        "breakeven": breakeven,
        "regime": regime,
    }


def population_stability_index(reference, current, bins: int = 10) -> float:
    """PSI。分桶邊界只由參考窗決定，缺值自成一桶。

    分桶邊界必須來自參考窗而不是合併資料 —— 用合併資料分桶會把當前窗的資訊
    洩漏進基線，PSI 就會被系統性低估。
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    ref_valid = reference[~np.isnan(reference)]
    if len(ref_valid) == 0:
        return float("nan")

    edges = np.unique(np.nanquantile(ref_valid, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf

    def shares(x):
        counts = np.histogram(x[~np.isnan(x)], bins=edges)[0].astype(float)
        counts = np.append(counts, np.isnan(x).sum())        # 缺值桶
        total = counts.sum()
        return counts / total if total else counts

    expected, actual = shares(reference), shares(current)
    mask = (expected > 0) | (actual > 0)
    e = np.clip(expected[mask], EPS, None)
    a = np.clip(actual[mask], EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame,
                  cols: list[str], bins: int = 10) -> pd.DataFrame:
    """每個特徵的 PSI，由大到小排序。"""
    rows = [
        {"feature": c,
         "psi": population_stability_index(reference[c], current[c], bins)}
        for c in cols
    ]
    return (pd.DataFrame(rows)
            .sort_values("psi", ascending=False, na_position="last")
            .reset_index(drop=True))


def retraining_trigger(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    cfg,
    cols: list[str] | None = None,
    major_share: float = 0.10,
) -> dict:
    """把兩個訊號合成一條可執行的規則。

    觸發重訓的條件（任一成立）：

      1. **動作翻轉**：當前窗的盛行率區間整段落在損益兩平線的另一側，
         且與參考窗的判定不同。這是成本層面的訊號，優先級最高 ——
         它代表「現在該做的事」跟「模型當初被訓練時該做的事」不一樣了。
      2. **輸入漂移**：PSI > 0.25 的特徵超過 major_share 比例。這是輸入層面的
         訊號，不直接證明決策會變差，但代表模型正在對它沒見過的分布做外插。

    刻意不把「AUC 下降」列為觸發條件：在 33 個正樣本的量級上，AUC 的抖動大於
    任何真實變化（見 stats.min_detectable_auc），拿它當觸發器只會製造假警報。
    """
    from .data import feature_cols

    cols = cols if cols is not None else feature_cols(reference)
    breakeven = breakeven_prevalence(cfg)

    ref_regime = prevalence_regime(
        int(reference["fail"].sum()), len(reference), breakeven)
    cur_regime = prevalence_regime(
        int(current["fail"].sum()), len(current), breakeven)

    decided = {"inspect", "no_inspect"}
    action_flipped = (
        ref_regime["regime"] in decided
        and cur_regime["regime"] in decided
        and ref_regime["regime"] != cur_regime["regime"]
    )

    psi = feature_drift(reference, current, cols)
    usable = psi["psi"].notna().sum()
    n_major = int((psi["psi"] > PSI_MAJOR).sum())
    share_major = n_major / usable if usable else float("nan")
    input_drifted = bool(usable and share_major > major_share)

    reasons = []
    if action_flipped:
        reasons.append(
            f"最佳動作翻轉：參考窗 {ref_regime['regime']} -> 當前窗 {cur_regime['regime']}"
            f"（損益兩平盛行率 {breakeven:.2%}）"
        )
    if input_drifted:
        reasons.append(
            f"輸入漂移：{n_major}/{usable} 個特徵 PSI > {PSI_MAJOR}"
            f"（{share_major:.1%} > 門檻 {major_share:.0%}）"
        )

    return {
        "retrain": bool(reasons),
        "reasons": reasons,
        "reference_regime": ref_regime,
        "current_regime": cur_regime,
        "action_flipped": action_flipped,
        "input_drifted": input_drifted,
        "psi_features_usable": int(usable),
        "psi_features_major": n_major,
        "psi_share_major": share_major,
        "psi_threshold_major": PSI_MAJOR,
        "psi_share_threshold": major_share,
        "psi_top": psi.head(10).to_dict(orient="records"),
    }
