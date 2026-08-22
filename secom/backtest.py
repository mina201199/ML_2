"""滾動原點回測（walk-forward backtest）。

為什麼需要這個檔案
------------------
單一次時序切分只能給你「一個時間窗上的一個數字」。而本專題最重要的資料發現
就是良率會漂移（7 月 22% fail → 10 月 1.8%），這正好意味著**任何單一切分的
結果都可能只是那個時期的巧合**。

正確的問法不是「這個策略在最後 20% 的資料上省了多少錢」，而是
「這個策略在**每一個**我本來就會部署它的時點上，各省了多少錢」。

做法：把時間軸切成連續的區塊，一步一步往前走。每一步：

    [--------- 訓練 ---------][-- 校準/選門檻 --][-- 評估 --]
                                                  ^ 只有這一段算成績
    往前推一格 ->
    [------------ 訓練 ------------][-- 校準/選門檻 --][-- 評估 --]

每一步的訓練資料都只包含該時點之前的批次，門檻都只用該時點可得的資料選定。
這模擬的是真實部署：你在 T 時點只能用 T 之前的東西做決定。

最後得到的是節省比例的**分布**，而不是單一數字。分布才能回答主管真正的問題：
「這個策略穩不穩？最差的情況有多差？」
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import decision
from .data import Split, feature_cols
from .evaluate import scores
from .models import fit_lgbm


@dataclass
class Fold:
    """一次滾動原點的結果。"""

    fold: int
    train_end: pd.Timestamp
    eval_from: pd.Timestamp
    eval_to: pd.Timestamp
    n_train: int
    n_cal: int
    n_eval: int
    n_fail_train: int
    n_fail_cal: int
    n_fail_eval: int
    threshold: float
    target_flag_rate: float
    pr_auc: float
    roc_auc: float
    recall_at_20: float
    cost_model: float
    cost_none: float
    cost_all: float
    saving_vs_best_baseline: float
    flag_rate: float
    catch_rate: float
    policy: str


def _make_split(df: pd.DataFrame, i_tr, i_cal, i_ev) -> Split:
    """把三段索引包成 models.fit_lgbm 能吃的 Split。

    注意 X_test 這一欄放的是評估段 —— fit_lgbm 只用 X_train 和 X_val，
    不會碰到它，所以放進來只是為了型別完整。
    """
    feats = feature_cols(df)
    parts = {}
    for name, idx in (("train", i_tr), ("val", i_cal), ("test", i_ev)):
        chunk = df.iloc[idx]
        parts[f"X_{name}"] = chunk[feats].reset_index(drop=True)
        parts[f"y_{name}"] = chunk["fail"].reset_index(drop=True)
        parts[f"ts_{name}"] = chunk["ts"].reset_index(drop=True)
    return Split(**parts)


def rolling_origin(
    df: pd.DataFrame,
    cfg,
    n_folds: int = 4,
    min_train_frac: float = 0.40,
    cal_frac: float = 0.15,
    policy: str = "quantile",
    verbose: bool = True,
) -> pd.DataFrame:
    """走過 n_folds 個部署時點，每一步都重訓、重選門檻、只在未來段評估。

    policy:
        "quantile" —— 選最佳「加驗比例」（相對規則，抗基準率漂移）
        "threshold" —— 選最佳絕對機率門檻

    回傳每一折一列的 DataFrame。
    """
    n = len(df)
    c_ins, c_esc = cfg.cost.c_inspect, cfg.cost.c_escape

    n_cal = max(int(n * cal_frac), 1)
    start = int(n * min_train_frac)
    # 剩下的空間平均切成 n_folds 個評估段
    room = n - start - n_cal
    if room < n_folds * 30:
        raise ValueError(
            f"資料不足以切 {n_folds} 折（可用 {room} 筆）。減少 n_folds 或 min_train_frac。"
        )
    step = room // n_folds

    rows: list[Fold] = []
    for k in range(n_folds):
        train_end = start + k * step
        cal_end = train_end + n_cal
        eval_end = min(cal_end + step, n)
        if eval_end <= cal_end:
            break

        i_tr = np.arange(0, train_end)
        i_cal = np.arange(train_end, cal_end)
        i_ev = np.arange(cal_end, eval_end)

        y_ev = df["fail"].iloc[i_ev].to_numpy()
        y_cal = df["fail"].iloc[i_cal].to_numpy()
        # 校準段或評估段沒有 fail 的話這一折沒有意義，跳過並說明
        if y_cal.sum() == 0 or y_ev.sum() == 0:
            if verbose:
                print(f"  fold {k + 1}: 校準段 fail={int(y_cal.sum())}、"
                      f"評估段 fail={int(y_ev.sum())} —— 缺正樣本，跳過")
            continue

        split = _make_split(df, i_tr, i_cal, i_ev)
        model = fit_lgbm(split, cfg, verbose=False)

        p_cal = model.predict_proba(split.X_val)
        p_ev = model.predict_proba(split.X_test)

        # 門檻只用校準段（該時點可得的資料）決定
        if policy == "quantile":
            chosen = decision.optimal_flag_rate(y_cal, p_cal, c_ins, c_esc)
            target = chosen["target_flag_rate"]
            t_ev = decision.threshold_for_flag_rate(p_ev, target)
        elif policy == "robust":
            chosen = decision.robust_flag_rate(y_cal, p_cal, c_ins, c_esc)
            target = chosen["target_flag_rate"]
            t_ev = decision.threshold_for_flag_rate(p_ev, target)
        else:
            chosen = decision.optimal_threshold(y_cal, p_cal, c_ins, c_esc)
            target = float("nan")
            t_ev = chosen["threshold"]

        at = decision.cost_at(y_ev, p_ev, t_ev, c_ins, c_esc)
        cost_none = int(y_ev.sum()) * c_esc / len(y_ev)
        cost_all = c_ins
        baseline = min(cost_none, cost_all)
        sc = scores(y_ev, p_ev)

        rows.append(
            Fold(
                fold=k + 1,
                train_end=df["ts"].iloc[train_end - 1],
                eval_from=df["ts"].iloc[i_ev[0]],
                eval_to=df["ts"].iloc[i_ev[-1]],
                n_train=len(i_tr),
                n_cal=len(i_cal),
                n_eval=len(i_ev),
                n_fail_train=int(df["fail"].iloc[i_tr].sum()),
                n_fail_cal=int(y_cal.sum()),
                n_fail_eval=int(y_ev.sum()),
                threshold=t_ev,
                target_flag_rate=target,
                pr_auc=sc["pr_auc"],
                roc_auc=sc["roc_auc"],
                recall_at_20=sc["recall@20%"],
                cost_model=at["cost_per_lot"],
                cost_none=cost_none,
                cost_all=cost_all,
                saving_vs_best_baseline=(baseline - at["cost_per_lot"]) / baseline,
                flag_rate=at["flag_rate"],
                catch_rate=at["catch_rate"],
                policy=policy,
            )
        )
        if verbose:
            r = rows[-1]
            print(
                f"  fold {r.fold}: train={r.n_train:>4}(fail {r.n_fail_train:>2})  "
                f"eval={r.n_eval:>3}(fail {r.n_fail_eval:>2})  "
                f"{r.eval_from:%m/%d}–{r.eval_to:%m/%d}  "
                f"驗{r.flag_rate:>4.0%} 攔{r.catch_rate:>4.0%}  "
                f"省 {r.saving_vs_best_baseline:+6.1%}"
            )

    return pd.DataFrame([vars(r) for r in rows])


def summarise(bt: pd.DataFrame) -> dict:
    """把每折的結果收斂成可以放進履歷與報告的幾個數字。

    刻意報中位數與最差值，不報平均 —— 折數少的時候平均容易被單一折帶走，
    而主管真正想知道的是「最差會多差」。
    """
    if bt.empty:
        return {"n_folds": 0}
    s = bt["saving_vs_best_baseline"]
    return {
        "n_folds": int(len(bt)),
        "saving_median": float(s.median()),
        "saving_min": float(s.min()),
        "saving_max": float(s.max()),
        "folds_profitable": int((s > 0).sum()),
        "roc_auc_median": float(bt["roc_auc"].median()),
        "pr_auc_median": float(bt["pr_auc"].median()),
        "catch_rate_median": float(bt["catch_rate"].median()),
        "flag_rate_median": float(bt["flag_rate"].median()),
        "eval_lots_total": int(bt["n_eval"].sum()),
        "eval_fails_total": int(bt["n_fail_eval"].sum()),
    }
