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
    """Platt scaling：在原始分數的 logit 上再套一層一維邏輯迴歸。

    只有兩個參數（斜率與截距），所以即使 val 只有三百多筆也不容易過擬合。
    這是刻意選的 —— isotonic regression 彈性大得多，在這個樣本量下會過擬合。
    """

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.lr_ = LogisticRegression(C=1e6, solver="lbfgs")
        self.lr_.fit(_logit(p_raw).reshape(-1, 1), y)
        return self

    def transform(self, p_raw: np.ndarray) -> np.ndarray:
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
        n_jobs=-1,
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
        if n_pos_va == 0 or ytr.iloc[i_tr].sum() == 0:
            if verbose:
                print(f"    fold {k}: 驗證段沒有 fail，跳過")
            continue
        if verbose:
            print(
                f"    fold {k}: train={len(i_tr)} (fail {int(ytr.iloc[i_tr].sum())})"
                f"  val={len(i_va)} (fail {n_pos_va})"
            )
        out.append((i_tr, i_va))
    return out


def choose_n_estimators(Xtr, ytr, cfg, verbose: bool = False) -> tuple[int, dict]:
    """用 train 內部的擴張窗 CV 決定樹的棵數。

    為什麼不用 val 早停：val 只有 11 個 fail，PR-AUC 在那個樣本量下純粹是噪音
    （第一版就停在第 1 棵樹）。而且 val 還要拿去做校準 —— 一份資料同時做兩件事
    就是兩次偷看。改成在 train 內部做時序 CV，val 就乾淨了。

    為什麼取「平均學習曲線的極大值」而不是「各折 best_iteration 的中位數」：
    每一折的 argmax 都是在 8~26 個正樣本上算出來的，抖得離譜（實測 [2, 1, 88]），
    中位數只會把噪音傳下去。先把各折的驗證曲線逐輪平均，再取極大值，
    等於用 3 倍的樣本數估同一件事，穩定得多。這是 lgb.cv 的標準用法。

    早停指標用 AUC 而非 PR-AUC：兩者方向一致，但 AUC 在小樣本上穩定得多。
    報告時仍然以 PR-AUC 為主指標。
    """
    import lightgbm as lgb

    c = cfg.model.cv
    folds = _folds(Xtr, ytr, cfg, verbose)
    floor = c.min_estimators
    if not folds:
        return floor, {"source": "no usable folds", "curve_len": 0}

    n_pos = int(ytr.sum())
    params = _lgbm_params(cfg, n_pos, len(ytr) - n_pos, cfg.model.lgbm.n_estimators)
    params.pop("n_estimators")
    params.update(metric=c.metric, verbosity=-1)

    hist = lgb.cv(
        params,
        lgb.Dataset(Xtr, label=ytr),
        num_boost_round=cfg.model.lgbm.n_estimators,
        folds=folds,
        callbacks=[lgb.early_stopping(c.early_stopping_rounds, verbose=False)],
        eval_train_metric=False,
    )
    key = next(k for k in hist if k.endswith("-mean"))
    curve = np.asarray(hist[key])
    best = int(np.argmax(curve)) + 1
    meta = {
        "source": "CV mean-curve argmax",
        "curve_len": len(curve),
        "best_metric": float(curve[best - 1]),
        "metric": c.metric,
        "n_folds": len(folds),
    }
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
    """用同一組 TimeSeriesSplit 產生 train 上的 out-of-fold 原始機率。

    為什麼需要這個：校準只能用「模型沒看過的資料」。原本只用 val（11 個 fail）
    去 fit Platt，兩個參數配 11 個正樣本，斜率被壓到幾乎為零 —— 校準後的機率
    全部擠在 0.028~0.054 之間（動態範圍剩 2 倍，原始是 47 倍），排序資訊被抹掉，
    導致最佳門檻擠在一條極窄的帶子裡，決策對門檻變得超級敏感。

    改用 OOF + val 一起校準，正樣本數從 11 拉到約 50 以上，斜率才估得準。
    第一折的訓練段沒有 OOF 預測（沒有任何模型沒看過它們），維持 NaN 並排除。
    """
    from sklearn.model_selection import TimeSeriesSplit

    oof = np.full(len(ytr), np.nan)
    tscv = TimeSeriesSplit(n_splits=cfg.model.cv.n_splits)
    for i_tr, i_va in tscv.split(Xtr):
        y_fold = ytr.iloc[i_tr]
        if y_fold.sum() == 0:
            continue
        n_pos = int(y_fold.sum())
        est = LGBMClassifier(
            **_lgbm_params(cfg, n_pos, len(y_fold) - n_pos, n_estimators)
        ).fit(Xtr.iloc[i_tr], y_fold)
        oof[i_va] = est.predict_proba(Xtr.iloc[i_va])[:, 1]
    return oof


def fit_lgbm(split, cfg, verbose: bool = False) -> FittedModel:
    """主模型：LightGBM。NaN 不填補，交給模型自己學缺值往哪走。"""
    pre = build_preprocessor(cfg, impute=False).fit(split.X_train, split.y_train)
    Xtr = pre.transform(split.X_train)

    n_estimators, cv_meta = choose_n_estimators(Xtr, split.y_train, cfg, verbose)
    floor_hit = cv_meta.get("source") == "min_estimators floor"
    if floor_hit:
        print(
            f"    警告：CV 平均曲線的峰值低於下限，改用 min_estimators={n_estimators}。"
            f"訊號很弱，這件事要寫進報告。"
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
    oof = oof_raw_predictions(Xtr, split.y_train, cfg, n_estimators)
    ok = ~np.isnan(oof)
    p_cal = np.concatenate([oof[ok], model._raw_proba(split.X_val)])
    y_cal = np.concatenate([split.y_train.to_numpy()[ok], split.y_val.to_numpy()])

    model.calibrator = PlattCalibrator().fit(p_cal, y_cal)
    model.meta["calibration"] = {
        "n_samples": int(len(y_cal)),
        "n_positives": int(y_cal.sum()),
        "source": "train OOF + val",
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
