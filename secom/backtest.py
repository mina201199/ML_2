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
from .models import REGISTRY


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
    model: str
    cost_baseline: float
    baseline_action: str
    cost_oracle_baseline: float
    caught: int
    missed: int
    calibration_from: pd.Timestamp
    calibration_to: pd.Timestamp
    lift_at_20: float = float("nan")
    escape_inflation: float = float("nan")
    # 記精確的標記筆數而不是只記比例：bootstrap 與置換檢定要重抽 2x2 格數，
    # 從 flag_rate 反推會有四捨五入誤差。
    n_flagged: int = 0
    # 實際選出的棵數與來源。min_estimators 下限有沒有生效是必須揭露的事，
    # 光在執行時印警告不夠 —— 報告讀者看不到終端機輸出。
    n_estimators: int = 0
    n_estimators_source: str = ""


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
    model_name: str = 'lgbm',
    capacity_frac: float | None = None,
    prediction_cache: dict | None = None,
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
        eval_end = n if k == n_folds - 1 else min(cal_end + step, n)
        if eval_end <= cal_end:
            break

        i_tr = np.arange(0, train_end)
        i_cal = np.arange(train_end, cal_end)
        i_ev = np.arange(cal_end, eval_end)

        y_ev = df["fail"].iloc[i_ev].to_numpy()
        y_cal = df["fail"].iloc[i_cal].to_numpy()
        split = _make_split(df, i_tr, i_cal, i_ev)
        # Cache is local to a single dataset/config run; never persisted.
        key = (model_name, train_end, cal_end, eval_end)
        if prediction_cache is not None and key in prediction_cache:
            p_cal, p_ev, fit_meta = prediction_cache[key]
        else:
            model = REGISTRY[model_name](split, cfg)
            p_cal = model.predict_proba(split.X_val)
            p_ev = model.predict_proba(split.X_test)
            fit_meta = {
                "n_estimators": model.best_iteration or 0,
                "n_estimators_source": model.meta.get("n_estimators_source", ""),
            }
            if prediction_cache is not None:
                prediction_cache[key] = (p_cal, p_ev, fit_meta)

        # 門檻只用校準段（該時點可得的資料）決定
        if policy == "quantile":
            chosen = decision.optimal_flag_rate(y_cal, p_cal, c_ins, c_esc, capacity_frac)
            target = chosen["target_flag_rate"]
            t_ev = decision.threshold_for_flag_rate(p_ev, target)
        elif policy == "robust":
            chosen = decision.robust_flag_rate(y_cal, p_cal, c_ins, c_esc, capacity_frac)
            target = chosen["target_flag_rate"]
            t_ev = decision.threshold_for_flag_rate(p_ev, target)
        else:
            chosen = decision.optimal_threshold(y_cal, p_cal, c_ins, c_esc, capacity_frac)
            target = float("nan")
            t_ev = chosen["threshold"]

        if capacity_frac is not None:
            t_ev = max(t_ev, decision.threshold_for_flag_rate(p_ev, capacity_frac))

        at = decision.cost_at(y_ev, p_ev, t_ev, c_ins, c_esc)
        cost_none = int(y_ev.sum()) * c_esc / len(y_ev)
        cost_all = c_ins
        # Choose a feasible baseline using ONLY the calibration window.
        # Under a cap compare no inspection with uniform random inspection at cap.
        # Random inspection uses its analytical expected cost (no sampled seed).
        if capacity_frac is not None:
            rate = np.floor(len(y_ev) * capacity_frac + 1e-10) / len(y_ev)
            select_rate = np.floor(len(y_cal) * capacity_frac + 1e-10) / len(y_cal)
        else:
            rate = select_rate = 1.0
        inspect_baseline = rate * c_ins + (1 - rate) * cost_none
        cal_escape = y_cal.mean() * c_esc
        select_inspect = (
            select_rate * c_ins + (1 - select_rate) * cal_escape < cal_escape
        )
        baseline = inspect_baseline if select_inspect else cost_none
        if not select_inspect:
            action = 'inspect_none'
        else:
            action = 'inspect_all' if rate == 1 else 'random_at_capacity'
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
                saving_vs_best_baseline=(
                    (baseline - at["cost_per_lot"]) / baseline
                    if baseline else float('nan')
                ),
                flag_rate=at["flag_rate"],
                catch_rate=at["catch_rate"],
                policy=policy,
                model=model_name,
                cost_baseline=baseline,
                baseline_action=action,
                cost_oracle_baseline=min(cost_none, inspect_baseline),
                caught=at['caught'], missed=at['missed'],
                calibration_from=split.ts_val.iloc[0],
                calibration_to=split.ts_val.iloc[-1],
                lift_at_20=sc["lift@20%"],
                escape_inflation=float(chosen.get("escape_inflation", float("nan"))),
                n_flagged=at["n_flagged"],
                n_estimators=fit_meta["n_estimators"],
                n_estimators_source=fit_meta["n_estimators_source"],
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


def reference_costs(
    bt: pd.DataFrame, cfg, capacity_frac: float | None = None
) -> dict:
    """成本化「完全不需要模型」的固定策略，用同一批評估窗。

    這些策略不需要分數、不需要門檻、不需要校準。因此其中最便宜的那一個，
    就是任何模型導向策略必須跨過的門檻 —— 而且是比 ``cost_baseline`` 更嚴格的
    門檻：``cost_baseline`` 是每折用校準窗「事前挑」出來的，會挑錯；這裡的每一個
    策略都是從頭到尾固定不變，連挑錯的機會都沒有。

    成本單位與 ``summarise`` 一致：每筆評估觀測，並依窗大小加權。
    產能受限時 ``inspect_all`` 不可行，仍列出來當作無限制情境的對照。
    """
    if bt.empty:
        return {}
    c_ins, c_esc = cfg.cost.c_inspect, cfg.cost.c_escape
    w = bt["n_eval"].to_numpy(dtype=float)
    prevalence = bt["n_fail_eval"].to_numpy(dtype=float) / w

    costs = {
        "inspect_none": float(np.average(prevalence * c_esc, weights=w)),
        "inspect_all": float(c_ins),
    }
    feasible = ["inspect_none"]
    if capacity_frac is None:
        feasible.append("inspect_all")
    else:
        # 整數配額，與 decision.threshold_for_flag_rate 的 floor 規則一致
        rate = np.floor(w * capacity_frac + 1e-10) / w
        costs["random_at_capacity"] = float(
            np.average(rate * c_ins + (1 - rate) * prevalence * c_esc, weights=w)
        )
        feasible.append("random_at_capacity")

    best = min(feasible, key=lambda k: costs[k])
    return {
        "costs": costs,
        "feasible": feasible,
        "infeasible": [k for k in costs if k not in feasible],
        "best_action": best,
        "best_cost": costs[best],
        "prevalence_pooled": float(bt["n_fail_eval"].sum() / bt["n_eval"].sum()),
    }


def summarise(bt: pd.DataFrame) -> dict:
    """Report fold variation, sample-weighted cost, and pooled interception counts.

    Fold medians alone are not aggregate business cost.
    """
    if bt.empty:
        return {"n_folds": 0}
    s = bt["saving_vs_best_baseline"]
    return {
        "n_folds": int(len(bt)),
        "saving_median": float(s.median()),
        "saving_q25": float(s.quantile(0.25)),
        "saving_q75": float(s.quantile(0.75)),
        "saving_min": float(s.min()),
        "saving_max": float(s.max()),
        "folds_profitable": int((s > 0).sum()),
        "roc_auc_median": float(bt["roc_auc"].median()),
        "pr_auc_median": float(bt["pr_auc"].median()),
        # lift@20% 的中位數是「模型排序有沒有贏過同配額隨機抽驗」最直接的讀數：
        # decision.required_lift 證明在 p·R >= 1 時，門檻恰好就是 1.0。
        "lift_at_20_median": float(bt["lift_at_20"].median()),
        "folds_at_min_estimators": int(
            (bt["n_estimators_source"] == "min_estimators floor").sum()
        ),
        "catch_rate_median": float(bt["catch_rate"].median()),
        "flag_rate_median": float(bt["flag_rate"].median()),
        "eval_lots_total": int(bt["n_eval"].sum()),
        "eval_fails_total": int(bt["n_fail_eval"].sum()),
        "cost_per_lot": float(np.average(bt.cost_model, weights=bt.n_eval)),
        "baseline_cost_per_lot": float(np.average(bt.cost_baseline, weights=bt.n_eval)),
        "total_caught": int(bt.caught.sum()),
        "total_missed": int(bt.missed.sum()),
        "aggregate_catch_rate": (
            float(bt.caught.sum() / bt.n_fail_eval.sum())
            if bt.n_fail_eval.sum() else float('nan')
        ),
    }
