"""洩漏防護測試。

這裡測的每一件事出錯都會讓整個專題的結論失效，而且都不會拋出例外 ——
只會讓分數變好看。前兩個測試是真的踩過才補上的。

    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from secom import config as cfg_mod
from secom import data as data_mod
from secom.pipeline import build_preprocessor


@pytest.fixture(scope="module")
def cfg():
    return cfg_mod.load()


@pytest.fixture(scope="module")
def df(cfg):
    path = cfg_mod.resolve_dir(cfg.data.processed_dir) / "secom.parquet"
    if not path.exists():
        pytest.skip("尚未建資料，請先執行 scripts/01_build_data.py")
    return pd.read_parquet(path)


# ── 1. 標籤欄不能混進特徵 ────────────────────────────────────────
# 曾經用 startswith("f") 篩特徵，而標籤欄叫 `fail` —— 它也以 f 開頭。


def test_label_column_is_not_a_feature(df):
    feats = data_mod.feature_cols(df)
    assert "fail" not in feats, "標籤欄 `fail` 混進特徵了 —— 這是直接把答案餵給模型"
    assert "ts" not in feats, "時間欄混進特徵了"


def test_feature_count_is_exactly_590(df):
    assert len(data_mod.feature_cols(df)) == 590


def test_feature_names_all_match_strict_pattern(df):
    for c in data_mod.feature_cols(df):
        assert data_mod.FEATURE_RE.match(c), f"{c} 不符合 f000-f589 的格式"


# ── 2. 時序切分不能有時間重疊 ───────────────────────────────────


def test_splits_do_not_overlap_in_time(df, cfg):
    s = data_mod.time_split(df, cfg)
    assert s.ts_train.max() <= s.ts_val.min(), "train 有資料晚於 val 的開始"
    assert s.ts_val.max() <= s.ts_test.min(), "val 有資料晚於 test 的開始"


def test_splits_partition_the_data_exactly(df, cfg):
    s = data_mod.time_split(df, cfg)
    assert len(s.y_train) + len(s.y_val) + len(s.y_test) == len(df)
    assert int(s.y_train.sum() + s.y_val.sum() + s.y_test.sum()) == int(df["fail"].sum())


def test_every_split_contains_at_least_one_failure(df, cfg):
    """任一段沒有 fail，該段的 PR-AUC 就沒有定義。"""
    s = data_mod.time_split(df, cfg)
    for name in ("train", "val", "test"):
        assert getattr(s, f"y_{name}").sum() > 0, f"{name} 沒有任何 fail"


# ── 3. 前處理的參數只能從 train 學 ──────────────────────────────


def test_preprocessor_column_choice_ignores_val_and_test(df, cfg):
    """把 val/test 換成垃圾，前處理選出的欄位必須完全不變。"""
    s = data_mod.time_split(df, cfg)
    pre_a = build_preprocessor(cfg, impute=False).fit(s.X_train, s.y_train)
    cols_a = list(pre_a.transform(s.X_train).columns)

    poisoned = s.X_test.copy()
    poisoned.iloc[:, :] = np.nan
    pre_b = build_preprocessor(cfg, impute=False).fit(s.X_train, s.y_train)
    _ = pre_b.transform(poisoned)
    cols_b = list(pre_b.transform(s.X_train).columns)

    assert cols_a == cols_b, "前處理的欄位選擇被 transform 的資料影響了"


def test_imputer_uses_train_medians_only(df, cfg):
    """填補值必須來自 train 的中位數，不能是 train+test 的。"""
    s = data_mod.time_split(df, cfg)
    pre = build_preprocessor(cfg, impute=True).fit(s.X_train, s.y_train)
    imputer = pre.named_steps["impute"]

    kept = list(pre.named_steps["drop_high_missing"].keep_)
    expected = s.X_train[kept].median().to_numpy()
    np.testing.assert_allclose(
        imputer.statistics_, expected, rtol=1e-9,
        err_msg="填補的中位數不等於 train 的中位數",
    )


def test_transform_is_row_independent(df, cfg):
    """單獨轉一列，結果必須跟整批一起轉的那一列相同。

    如果不成立，表示前處理裡有跨列的統計量（例如在 transform 時重算平均），
    上線後一次只來一批的推論結果就會跟離線評估不一致。
    """
    s = data_mod.time_split(df, cfg)
    pre = build_preprocessor(cfg, impute=True).fit(s.X_train, s.y_train)
    full = pre.transform(s.X_test)
    single = pre.transform(s.X_test.iloc[[7]])
    np.testing.assert_allclose(
        single.to_numpy()[0], full.to_numpy()[7], rtol=1e-9,
        err_msg="前處理不是逐列獨立的",
    )


# ── 4. 決策層的成本模型要自我一致 ───────────────────────────────


def test_cost_model_endpoints_are_the_trivial_policies():
    from secom import decision

    y = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0])
    p = np.array([0.1, 0.2, 0.9, 0.15, 0.7, 0.05, 0.3, 0.25, 0.6, 0.12])
    c_ins, c_esc = 10.0, 300.0

    none = decision.cost_at(y, p, 1.1, c_ins, c_esc)
    assert none["n_flagged"] == 0
    assert none["total_cost"] == pytest.approx(y.sum() * c_esc)

    every = decision.cost_at(y, p, 0.0, c_ins, c_esc)
    assert every["n_flagged"] == len(y)
    assert every["missed"] == 0
    assert every["total_cost"] == pytest.approx(len(y) * c_ins)


def test_optimal_threshold_respects_capacity_cap():
    from secom import decision

    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.07).astype(int)
    p = np.clip(0.05 + 0.25 * y + rng.normal(0, 0.08, 400), 0.001, 0.999)

    capped = decision.optimal_threshold(y, p, 10.0, 500.0, capacity_frac=0.15)
    assert capped["flag_rate"] <= 0.15 + 1e-9, "產能上限沒有被遵守"


# ── 5. 滾動回測不能偷看未來 ──────────────────────────────────────
# 回測現在是這個專案的核心結論來源，所以它本身必須被測。


def test_backtest_windows_are_strictly_chronological(df, cfg):
    """每一折的 訓練 < 校準 < 評估 必須在時間上嚴格遞增，且不重疊。"""
    from secom import backtest as bt_mod

    bt = bt_mod.rolling_origin(df, cfg, n_folds=4, policy="robust", verbose=False)
    assert not bt.empty, "回測沒有產生任何折"
    for _, r in bt.iterrows():
        assert r["train_end"] <= r["eval_from"], (
            f"fold {r['fold']}: 訓練段結束於 {r['train_end']}，"
            f"晚於評估段起點 {r['eval_from']} —— 偷看未來"
        )
        assert r["eval_from"] <= r["eval_to"]


def test_backtest_training_windows_grow_forward(df, cfg):
    """滾動原點：後面的折訓練資料必須更多，評估段必須更晚。"""
    from secom import backtest as bt_mod

    bt = bt_mod.rolling_origin(df, cfg, n_folds=4, policy="robust", verbose=False)
    assert bt["n_train"].is_monotonic_increasing, "訓練集沒有隨時間擴張"
    assert bt["eval_from"].is_monotonic_increasing, "評估段沒有往前推進"


def test_every_backtest_fold_has_failures_to_find(df, cfg):
    """評估段沒有 fail 的折會讓攔截率無定義，必須已被排除。"""
    from secom import backtest as bt_mod

    bt = bt_mod.rolling_origin(df, cfg, n_folds=4, policy="robust", verbose=False)
    assert (bt["n_fail_eval"] > 0).all()
    assert (bt["n_fail_cal"] > 0).all()


# ── 6. 穩健策略的保守性必須隨樣本量單調 ─────────────────────────


def test_conservative_base_rate_shrinks_toward_point_estimate():
    """樣本越多，上界越貼近點估計；樣本少時上界必須明顯更高。"""
    from secom.decision import conservative_base_rate

    small = conservative_base_rate(7, 235)
    large = conservative_base_rate(700, 23500)      # 同樣的 2.98%，100 倍樣本
    point = 7 / 235

    assert small > point, "上界必須高於點估計"
    assert large > point
    assert large < small, "樣本變多時上界應該收斂，實際卻沒有"
    assert abs(large - point) < abs(small - point) / 5


def test_robust_policy_never_inspects_less_than_plain_policy():
    """在相同資料上，保守版的加驗比例不應低於一般版。

    這是穩健策略的定義性質：它只會更保守，不會更冒險。
    """
    from secom import decision

    rng = np.random.default_rng(7)
    y = (rng.random(300) < 0.03).astype(int)
    p = np.clip(0.03 + 0.10 * y + rng.normal(0, 0.05, 300), 1e-3, 1 - 1e-3)

    plain = decision.optimal_flag_rate(y, p, 2000, 60000)
    robust = decision.robust_flag_rate(y, p, 2000, 60000)

    assert robust["escape_inflation"] >= 1.0
    assert robust["target_flag_rate"] >= plain["target_flag_rate"] - 1e-9, (
        f"保守策略驗得比一般策略少（{robust['target_flag_rate']:.0%} "
        f"< {plain['target_flag_rate']:.0%}）"
    )


# ── 7. SQL 時間窗特徵不能包含當前列 ──────────────────────────────


def test_sql_window_features_exclude_current_row(cfg, df):
    """偏離特徵的基線窗必須是 `20 PRECEDING AND 1 PRECEDING`。

    含了當前列就是用當下的值去算自己的基線，那是洩漏 ——
    而且是那種不會報錯、只會讓分數變好看的洩漏。
    """
    from secom import data as data_mod
    from secom import sql as sql_mod

    con = sql_mod.connect(cfg)
    cols = data_mod.feature_cols(df)[:2]
    tf = sql_mod.build_time_features(con, cols)

    c = cols[0]
    # 第一列前面沒有任何資料，偏離量必須是 NaN（沒有基線可比）
    assert pd.isna(tf[f"{c}_dev20"].iloc[0]), (
        "第一列有偏離值 —— 表示基線窗包含了當前列"
    )
    # 手算第 25 列：當前值 減去 前 20 列的平均
    raw = df[c].to_numpy()
    i = 25
    expected = raw[i] - np.nanmean(raw[i - 20:i])
    got = tf[f"{c}_dev20"].iloc[i]
    if not (np.isnan(expected) and pd.isna(got)):
        np.testing.assert_allclose(
            got, expected, rtol=1e-6,
            err_msg="偏離量不等於『當前值 - 前 20 列平均』",
        )
