"""前處理。

整個檔案只為了保證一件事：**所有前處理的參數都只從 train 學來**。

哪些欄位是零變異、哪些欄位缺值太多、中位數是多少 —— 這些全都是「從資料學到的
參數」。如果先在整份資料上算，再切 train/test，那就是資料洩漏，離線分數會虛高。
把它們全部包進 sklearn Pipeline，是用結構強制這件事不會做錯，而不是靠自己記得。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DropConstant(BaseEstimator, TransformerMixin):
    """丟掉在 train 上只有單一取值（含全為 NaN）的欄位。SECOM 有 116 個。"""

    def fit(self, X: pd.DataFrame, y=None):
        nunique = X.nunique(dropna=True)
        self.keep_ = [c for c in X.columns if nunique[c] > 1]
        self.dropped_ = [c for c in X.columns if nunique[c] <= 1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep_, dtype=object)


class DropHighMissing(BaseEstimator, TransformerMixin):
    """丟掉在 train 上缺值比例過高的欄位。"""

    def __init__(self, max_missing_frac: float = 0.5):
        self.max_missing_frac = max_missing_frac

    def fit(self, X: pd.DataFrame, y=None):
        frac = X.isna().mean()
        self.keep_ = [c for c in X.columns if frac[c] <= self.max_missing_frac]
        self.dropped_ = [c for c in X.columns if frac[c] > self.max_missing_frac]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep_, dtype=object)


class DropCorrelated(BaseEstimator, TransformerMixin):
    """成對相關係數超過門檻時，保留前一個、丟掉後一個。

    SECOM 有大量重複量測的感測器。這一步會讓 SHAP 的歸因乾淨很多 ——
    高度共線的特徵會互相分掉重要度，讓「關鍵感測器」的結論失真。
    """

    def __init__(self, threshold: float | None = 0.95):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None):
        if self.threshold is None:
            self.keep_, self.dropped_ = list(X.columns), []
            return self
        corr = X.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.dropped_ = [c for c in upper.columns if (upper[c] > self.threshold).any()]
        self.keep_ = [c for c in X.columns if c not in set(self.dropped_)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.keep_, dtype=object)


def build_preprocessor(cfg, impute: bool = False) -> Pipeline:
    """組出前處理管線。

    impute=False（樹模型用）
        只做欄位剔除，NaN 原封不動留著。LightGBM 原生就會學「缺值該往哪一邊走」,
        而缺值本身在製程資料裡常常帶訊息（某個站點沒量到 = 那台機器當時沒開）。
        硬填中位數會把這個訊息抹掉。

    impute=True（線性 baseline 用）
        額外補中位數填補與標準化，因為邏輯迴歸不吃 NaN 也需要尺度一致。
    """
    steps: list[tuple[str, object]] = []

    if cfg.preprocess.drop_constant:
        steps.append(("drop_constant", DropConstant()))

    steps.append(
        ("drop_high_missing", DropHighMissing(cfg.preprocess.max_missing_frac))
    )

    if cfg.preprocess.corr_threshold is not None:
        steps.append(("drop_correlated", DropCorrelated(cfg.preprocess.corr_threshold)))

    if impute:
        steps.append(("impute", SimpleImputer(strategy="median")))
        steps.append(("scale", StandardScaler()))

    pipe = Pipeline(steps)
    # 讓每一步都回傳 DataFrame，這樣 SHAP 才能拿到真正的欄位名稱
    pipe.set_output(transform="pandas")
    return pipe


def describe_reduction(pre: Pipeline, n_input: int, n_output: int) -> str:
    """把「590 維砍到 N 維」講成一句人話，放進報告裡。"""
    parts = []
    for name, step in pre.steps:
        dropped = getattr(step, "dropped_", None)
        if dropped:
            parts.append(f"{name} -{len(dropped)}")
    detail = "，".join(parts) if parts else "無剔除"
    return f"{n_input} 維 -> {n_output} 維（{detail}）"
