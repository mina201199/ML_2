"""評估指標。

accuracy 在這裡是禁用字。盛行率 6.6% 的情況下，「全部猜 pass」就有 93.4%
accuracy，卻抓不到任何一顆壞品。用得上的是這幾個：

  PR-AUC (average precision)  主要指標。地板約等於盛行率，不是 0.5。
  ROC-AUC                     次要。在極不平衡下會過度樂觀，只當參考。
  Brier score                 校準品質。成本計算要乘機率，機率不準就白算。
  recall@k                    產線只驗得動 k% 的批次時，抓得到幾成壞品。
  lift@k                      前 k% 的壞品濃度是隨機抽樣的幾倍。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float) -> float:
    """依分數排序取前 k 比例，回傳抓到的壞品佔全部壞品的比例。"""
    n = len(y_true)
    n_flag = max(1, int(np.ceil(n * k)))
    order = np.argsort(-y_score, kind="stable")[:n_flag]
    total_pos = y_true.sum()
    return float(y_true[order].sum() / total_pos) if total_pos else float("nan")


def lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float) -> float:
    """前 k 比例裡的壞品濃度 ÷ 整體盛行率。等於 1 表示模型毫無用處。"""
    base = y_true.mean()
    if base == 0:
        return float("nan")
    n_flag = max(1, int(np.ceil(len(y_true) * k)))
    order = np.argsort(-y_score, kind="stable")[:n_flag]
    return float(y_true[order].mean() / base)


def scores(y_true, y_score, ks=(0.05, 0.10, 0.20)) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out = {
        "n": int(len(y_true)),
        "n_fail": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "pr_auc": float(average_precision_score(y_true, y_score)) if y_true.sum() else float('nan'),
        "roc_auc": (
            float(roc_auc_score(y_true, y_score))
            if len(np.unique(y_true)) == 2 else float('nan')
        ),
        "brier": float(brier_score_loss(y_true, y_score)),
    }
    for k in ks:
        pct = int(round(k * 100))
        out[f"recall@{pct}%"] = recall_at_k(y_true, y_score, k)
        out[f"lift@{pct}%"] = lift_at_k(y_true, y_score, k)
    # PR-AUC 相對於地板（盛行率）的提升倍數 —— 比裸分數好懂
    out["pr_auc_over_floor"] = (
        out["pr_auc"] / out["prevalence"] if out['prevalence'] else float('nan')
    )
    return out


def at_threshold(y_true, y_score, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "flag_rate": float(y_pred.mean()),
    }


def compare(models: dict, X, y, calibrated: bool = True) -> pd.DataFrame:
    """把所有模型在同一份資料上的指標並排，方便一眼看出誰真的贏。"""
    rows = []
    for name, m in models.items():
        p = m.predict_proba(X, calibrated=calibrated)
        rows.append({"model": name, **scores(y, p)})
    df = pd.DataFrame(rows).set_index("model")
    cols = [
        "pr_auc",
        "pr_auc_over_floor",
        "roc_auc",
        "brier",
        "recall@10%",
        "recall@20%",
        "lift@10%",
        "lift@20%",
    ]
    return df[cols]


def pr_curve(y_true, y_score) -> pd.DataFrame:
    precision, recall, thr = precision_recall_curve(y_true, y_score)
    # precision_recall_curve 回傳的 thresholds 比 precision/recall 少一個
    return pd.DataFrame(
        {"threshold": np.append(thr, 1.0), "precision": precision, "recall": recall}
    )
