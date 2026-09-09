"""模型：兩個 baseline 加一個 LightGBM，外掛機率校準。

為什麼一定要有 baseline：
    在 6.6% 的不平衡下，「全部猜 pass」就有 93.4% 的 accuracy。任何沒有
    baseline 對照的分數都是在騙自己。這裡放兩個 —— 多數類別（PR-AUC 的地板，
    約等於盛行率）和 L2 邏輯迴歸（線性可分的程度）。LightGBM 要贏過它們才算有價值。

為什麼一定要校準：
    第 3 章的成本計算是 `期望成本 = P(fail) × 損失`。如果 P(fail) 不是真實機率，
    只是一個「排序用的分數」，那個乘法就沒有意義，最佳門檻也算錯。
    樹模型的輸出通常過度自信，所以要校準。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .pipeline import build_preprocessor

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Two-parameter sigmoid calibration; single-class windows use a smoothed constant.

    Small samples and temporal distribution changes can still impair calibration.
    """

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        if len(np.unique(y)) < 2:
            self.constant_ = float((np.sum(y) + 1) / (len(y) + 2))
            return self
        self.lr_ = LogisticRegression(C=1e6, solver="lbfgs")
        self.lr_.fit(_logit(p_raw).reshape(-1, 1), y)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
        if hasattr(self, 'constant_'):
            return np.full(len(p_raw), self.constant_)
        return self.lr_.predict_proba(_logit(p_raw).reshape(-1, 1))[:, 1]


@dataclass
class FittedModel:
    name: str
    preprocessor: Pipeline
    estimator: object
    calibrator: PlattCalibrator | None = None
    n_features_in: int = 0
    n_features_out: int = 0
    best_iteration: int | None = None
    meta: dict = field(default_factory=dict)

    def _raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xt = self.preprocessor.transform(X)
        return self.estimator.predict_proba(Xt)[:, 1]

    def predict_proba(self, X: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        p = self._raw_proba(X)
        if calibrated and self.calibrator is not None:
            p = self.calibrator.transform(p)
        return p

    @property
    def feature_names(self) -> list[str]:
        return list(self.preprocessor[-1].get_feature_names_out())


def fit_majority(split, cfg) -> FittedModel:
    """Baseline 0：永遠猜多數類別。PR-AUC 的理論地板。"""
    pre = build_preprocessor(cfg, impute=True).fit(split.X_train, split.y_train)
    est = DummyClassifier(strategy="prior").fit(
        pre.transform(split.X_train), split.y_train
    )
    return FittedModel(
        name="majority",
        preprocessor=pre,
        estimator=est,
        n_features_in=split.X_train.shape[1],
        n_features_out=pre.transform(split.X_train.head(1)).shape[1],
    )


def fit_logreg(split, cfg) -> FittedModel:
    """Baseline 1：L2 邏輯迴歸，class_weight 平衡。看看線性到什麼程度。"""
    pre = build_preprocessor(cfg, impute=True).fit(split.X_train, split.y_train)
    Xtr = pre.transform(split.X_train)

    est = LogisticRegression(
        penalty="l2",
        C=0.05,
        class_weight="balanced",
        max_iter=5000,
        random_state=cfg.seed,
    ).fit(Xtr, split.y_train)

    model = FittedModel(
        name="logreg",
        preprocessor=pre,
        estimator=est,
        n_features_in=split.X_train.shape[1],
        n_features_out=Xtr.shape[1],
    )
    model.calibrator = PlattCalibrator().fit(
        model._raw_proba(split.X_val), split.y_val.to_numpy()
    )
    return model


def _lgbm_params(cfg, n_pos: int, n_neg: int, n_estimators: int) -> dict:
    p = cfg.model.lgbm
    return dict(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=p.learning_rate,
        num_leaves=p.num_leaves,
        min_child_samples=p.min_child_samples,
        subsample=p.subsample,
        subsample_freq=p.subsample_freq,
        colsample_bytree=p.colsample_bytree,
        reg_lambda=p.reg_lambda,
        scale_pos_weight=n_neg / max(n_pos, 1),
        random_state=cfg.seed,
        n_jobs=p.get('n_jobs', 4),
        verbose=-1,
    )


def _folds(Xtr, ytr, cfg, verbose: bool = False) -> list[tuple]:
    """產生擴張窗切分，並丟掉驗證段沒有 fail 的折（沒有訊號可比）。"""
    from sklearn.model_selection import TimeSeriesSplit

    out = []
    for k, (i_tr, i_va) in enumerate(
        TimeSeriesSplit(n_splits=cfg.model.cv.n_splits).split(Xtr), start=1
    ):
        n_pos_va = int(ytr.iloc[i_va].sum())
        if ytr.iloc[i_va].nunique() < 2 or ytr.iloc[i_tr].nunique() < 2:
            if verbose:
                print(f"    fold {k}: 訓練或驗證段只有單一類別，略過 AUC 選模")
            continue
        if verbose:
            print(
                f"    fold {k}: train={len(i_tr)} (fail {int(ytr.iloc[i_tr].sum())})"
                f"  val={len(i_va)} (fail {n_pos_va})"
            )
        out.append((i_tr, i_va))
    return out


def choose_n_estimators(Xtr, ytr, cfg, verbose: bool = False) -> tuple[int, dict]:
    """Choose tree count from the mean temporal CV curve.

    Each fold fits its own preprocessing, class weight, and feature bins; apply configured patience
    and minimum count.
    """
    import inspect

    import lightgbm as lgb

    c = cfg.model.cv
    folds = _folds(Xtr, ytr, cfg, verbose)
    floor = c.min_estimators
    if not folds:
        return floor, {"source": "no usable folds", "curve_len": 0}

    curves = []
    for i_tr, i_va in folds:
        pre = build_preprocessor(cfg, impute=False).fit(Xtr.iloc[i_tr], ytr.iloc[i_tr])
        n_pos = int(ytr.iloc[i_tr].sum())
        estimator = LGBMClassifier(**_lgbm_params(
            cfg, n_pos, len(i_tr) - n_pos, cfg.model.lgbm.n_estimators))
        hist = {}
        X_valid, y_valid = pre.transform(Xtr.iloc[i_va]), ytr.iloc[i_va]
        # LightGBM 4.7 起 `eval_set` 改為 `eval_X` / `eval_y`，舊名稱會發棄用警告；
        # 但 requirements 的下限是 4.5，那裡還沒有新名稱。因此按實際簽名選用 ——
        # 這不是防禦性程式碼，是 4.5~4.7 之間真實存在的 API 改名。
        eval_kwargs = (
            {"eval_X": X_valid, "eval_y": y_valid}
            if "eval_X" in inspect.signature(estimator.fit).parameters
            else {"eval_set": [(X_valid, y_valid)]}
        )
        # 不傳 early stopping callback：patience 要套在「各折平均後」的曲線上，
        # 每折各自早停再平均是另一種方法，會改變選出的棵數。
        estimator.fit(
            pre.transform(Xtr.iloc[i_tr]), ytr.iloc[i_tr],
            **eval_kwargs, eval_metric=c.metric,
            callbacks=[lgb.record_evaluation(hist)],
        )
        curves.append(hist['valid_0'][c.metric])
    curve = np.mean(curves, axis=0)
    # Apply patience to the mean curve, with each fold's own preprocessing/bins.
    peak, stop = 0, len(curve)
    for i in range(1, len(curve)):
        if curve[i] > curve[peak]:
            peak = i
        if i - peak >= c.early_stopping_rounds:
            stop = i + 1
            break
    # patience 從未觸發 = 峰值可能還在上限之外，選出的棵數會受上限影響。
    # 這是 n_estimators 上限開始生效的精確條件，必須讓它可見。
    ceiling_binding = stop == len(curve)
    curve = curve[:stop]
    best = int(np.argmax(curve)) + 1
    meta = {
        "source": "CV mean-curve argmax",
        "curve_len": len(curve),
        "best_metric": float(curve[best - 1]),
        "metric": c.metric,
        "n_folds": len(folds),
        "ceiling_binding": bool(ceiling_binding),
    }
    if ceiling_binding:
        print(
            f"    警告：內層 CV 的 patience 從未觸發（曲線用滿 {len(curve)} 輪 = "
            f"n_estimators 上限）。選出的棵數受上限影響，請調高 "
            f"config.yaml 的 model.lgbm.n_estimators 後重新量測。"
        )
    if verbose:
        print(
            f"    CV 平均曲線 {len(curve)} 輪，{c.metric} 峰值 {curve[best - 1]:.4f}"
            f" @ 第 {best} 棵"
        )
    if best < floor:
        meta["source"] = "min_estimators floor"
        return floor, meta
    return best, meta


def oof_raw_predictions(Xtr, ytr, cfg, n_estimators: int) -> np.ndarray:
    """Temporal out-of-fold scores using a predetermined tree count.

    Every fold fits preprocessing only on its own training rows. Unavailable early predictions
    remain NaN. Positive-slope calibration preserves score ordering.
    """
    from sklearn.model_selection import TimeSeriesSplit

    oof = np.full(len(ytr), np.nan)
    tscv = TimeSeriesSplit(n_splits=cfg.model.cv.n_splits)
    for i_tr, i_va in tscv.split(Xtr):
        y_fold = ytr.iloc[i_tr]
        if y_fold.sum() == 0:
            continue
        n_pos = int(y_fold.sum())
        pre = build_preprocessor(cfg, impute=False).fit(Xtr.iloc[i_tr], y_fold)
        est = LGBMClassifier(
            **_lgbm_params(cfg, n_pos, len(y_fold) - n_pos, n_estimators)
        ).fit(pre.transform(Xtr.iloc[i_tr]), y_fold)
        oof[i_va] = est.predict_proba(pre.transform(Xtr.iloc[i_va]))[:, 1]
    return oof


def fit_lgbm(split, cfg, verbose: bool = False) -> FittedModel:
    """主模型：LightGBM。NaN 不填補，交給模型自己學缺值往哪走。"""
    pre = build_preprocessor(cfg, impute=False).fit(split.X_train, split.y_train)
    Xtr = pre.transform(split.X_train)

    n_estimators, cv_meta = choose_n_estimators(split.X_train, split.y_train, cfg, verbose)
    floor_hit = cv_meta.get("source") == "min_estimators floor"
    if floor_hit:
        print(
            f"    警告：CV 平均曲線的峰值低於下限，改用 min_estimators={n_estimators}。"
            f"這是設定的最低棵數限制，需在報告揭露。"
        )

    n_pos = int(split.y_train.sum())
    n_neg = len(split.y_train) - n_pos
    est = LGBMClassifier(**_lgbm_params(cfg, n_pos, n_neg, n_estimators)).fit(
        Xtr, split.y_train
    )

    model = FittedModel(
        name="lgbm",
        preprocessor=pre,
        estimator=est,
        n_features_in=split.X_train.shape[1],
        n_features_out=Xtr.shape[1],
        best_iteration=n_estimators,
        meta={
            "scale_pos_weight": n_neg / max(n_pos, 1),
            "cv": cv_meta,
            "n_estimators_source": cv_meta.get("source"),
        },
    )

    # 校準資料 = train 的 OOF 預測 + val。兩者都是模型沒看過的。
    # Fixed in advance: using the globally CV-selected count for earlier OOF
    # predictions would indirectly reuse their labels during hyperparameter tuning.
    oof_trees = cfg.model.cv.get('calibration_estimators', 30)
    oof = oof_raw_predictions(split.X_train, split.y_train, cfg, oof_trees)
    ok = ~np.isnan(oof)
    p_cal = np.concatenate([oof[ok], model._raw_proba(split.X_val)])
    y_cal = np.concatenate([split.y_train.to_numpy()[ok], split.y_val.to_numpy()])

    model.calibrator = PlattCalibrator().fit(p_cal, y_cal)
    model.meta["calibration"] = {
        "n_samples": int(len(y_cal)),
        "n_positives": int(y_cal.sum()),
        "source": "train OOF + val",
        "oof_fixed_estimators": oof_trees,
    }
    if verbose:
        print(
            f"    校準樣本 {len(y_cal)} 筆（正樣本 {int(y_cal.sum())}）"
            f" = train OOF {int(ok.sum())} + val {len(split.y_val)}"
        )
    return model


REGISTRY = {
    "majority": fit_majority,
    "logreg": fit_logreg,
    "lgbm": fit_lgbm,
}


def fit_all(split, cfg, verbose: bool = False) -> dict[str, FittedModel]:
    models = {}
    for name, fn in REGISTRY.items():
        print(f"  訓練 {name} ...")
        models[name] = fn(split, cfg) if name != "lgbm" else fn(split, cfg, verbose)
    return models
