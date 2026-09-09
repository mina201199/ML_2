"""漂移偵測與重訓觸發規則的測試。

這個模組是這一版新增的。它必須有測試，否則就跟原本那個「設定檔永遠不啟用、
也沒人驗證過」的 DropCorrelated 是同一個問題。
"""

import numpy as np
import pandas as pd
import pytest

from secom import config, drift


@pytest.fixture(scope="module")
def cfg():
    return config.load()


def test_breakeven_prevalence_is_the_cost_ratio_inverse(cfg):
    """p* = c_inspect / c_escape。這條線是整個告警規則的錨點。"""
    assert drift.breakeven_prevalence(cfg) == pytest.approx(
        cfg.cost.c_inspect / cfg.cost.c_escape
    )
    assert drift.breakeven_prevalence(cfg) == pytest.approx(1 / 30)


def test_prevalence_regime_reports_undecided_when_interval_straddles_breakeven():
    """小窗 + 稀有事件下「無法判定」必須是允許的答案，不能硬歸類。"""
    breakeven = 1 / 30                               # 3.33%

    # 6/235 = 2.55%，Clopper-Pearson 區間會跨過 3.33%
    straddling = drift.prevalence_regime(6, 235, breakeven)
    assert straddling["regime"] == "undecided"
    assert straddling["ci"][0] < breakeven < straddling["ci"][1]

    # 大量高盛行率：整段區間都在線上方
    clearly_high = drift.prevalence_regime(200, 1000, breakeven)
    assert clearly_high["regime"] == "inspect"
    assert clearly_high["ci"][0] > breakeven

    # 大量低盛行率：整段區間都在線下方
    clearly_low = drift.prevalence_regime(2, 2000, breakeven)
    assert clearly_low["regime"] == "no_inspect"
    assert clearly_low["ci"][1] < breakeven


def test_prevalence_regime_handles_degenerate_windows():
    breakeven = 1 / 30
    assert drift.prevalence_regime(0, 0, breakeven)["regime"] == "undecided"
    zero_fail = drift.prevalence_regime(0, 50, breakeven)
    assert zero_fail["prevalence"] == 0.0
    assert zero_fail["ci"][0] == 0.0
    all_fail = drift.prevalence_regime(50, 50, breakeven)
    assert all_fail["ci"][1] == 1.0
    assert all_fail["regime"] == "inspect"


def test_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2000)
    assert drift.population_stability_index(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_flags_a_real_shift_and_stays_quiet_for_noise():
    rng = np.random.default_rng(1)
    reference = rng.normal(0, 1, 4000)
    same = rng.normal(0, 1, 4000)                    # 同分布，只是不同抽樣
    shifted = rng.normal(2.5, 1, 4000)               # 平移 2.5 個標準差

    assert drift.population_stability_index(reference, same) < drift.PSI_MINOR
    assert drift.population_stability_index(reference, shifted) > drift.PSI_MAJOR


def test_psi_bin_edges_come_only_from_the_reference_window():
    """分桶邊界若用合併資料算，當前窗的資訊會洩漏進基線，PSI 會被低估。

    做法：固定參考窗，把當前窗換成一個完全落在參考窗範圍外的分布。
    只用參考窗分桶時，當前窗會全部掉進最後一桶，PSI 必須很大。
    """
    reference = np.linspace(0, 1, 1000)
    outside = np.linspace(10, 11, 1000)
    assert drift.population_stability_index(reference, outside) > 2.0


def test_psi_treats_missingness_as_its_own_bucket():
    """製程資料裡「沒量到」本身帶訊息，所以缺值比例的變化必須被 PSI 看見。"""
    rng = np.random.default_rng(2)
    reference = rng.normal(size=2000)
    current = reference.copy()
    current[:1000] = np.nan                          # 半數變成缺值，數值分布不變

    psi = drift.population_stability_index(reference, current)
    assert psi > drift.PSI_MAJOR, f"缺值比例劇變沒有被偵測到（PSI={psi}）"


def test_psi_returns_nan_when_reference_has_no_usable_values():
    assert np.isnan(drift.population_stability_index([np.nan] * 10, [1.0] * 10))
    assert np.isnan(drift.population_stability_index([5.0] * 10, [1.0] * 10))


def test_feature_drift_is_sorted_worst_first():
    rng = np.random.default_rng(3)
    reference = pd.DataFrame({
        "f000": rng.normal(size=1500),
        "f001": rng.normal(size=1500),
    })
    current = pd.DataFrame({
        "f000": rng.normal(size=1500),               # 沒漂移
        "f001": rng.normal(4, 1, 1500),              # 大幅漂移
    })
    table = drift.feature_drift(reference, current, ["f000", "f001"])
    assert list(table["feature"]) == ["f001", "f000"]
    assert table["psi"].is_monotonic_decreasing


def test_retraining_trigger_fires_when_the_optimal_action_flips(cfg):
    """成本層面的訊號：盛行率跨過損益兩平線，最佳動作就換了一個。"""
    rng = np.random.default_rng(5)
    n = 1200
    features = {f"f{i:03d}": rng.normal(size=n) for i in range(3)}

    high = pd.DataFrame({**features, "fail": np.r_[np.ones(240), np.zeros(n - 240)]})
    low = pd.DataFrame({**features, "fail": np.r_[np.ones(2), np.zeros(n - 2)]})

    result = drift.retraining_trigger(high, low, cfg, cols=list(features))
    assert result["retrain"] is True
    assert result["action_flipped"] is True
    assert result["reference_regime"]["regime"] == "inspect"
    assert result["current_regime"]["regime"] == "no_inspect"
    assert any("最佳動作翻轉" in r for r in result["reasons"])


def test_retraining_trigger_stays_quiet_when_nothing_moved(cfg):
    rng = np.random.default_rng(6)
    n = 1200
    frame = pd.DataFrame({
        **{f"f{i:03d}": rng.normal(size=n) for i in range(5)},
        "fail": np.r_[np.ones(120), np.zeros(n - 120)],
    })
    other = pd.DataFrame({
        **{f"f{i:03d}": rng.normal(size=n) for i in range(5)},
        "fail": np.r_[np.ones(120), np.zeros(n - 120)],
    })
    result = drift.retraining_trigger(frame, other, cfg,
                                      cols=[f"f{i:03d}" for i in range(5)])
    assert result["retrain"] is False
    assert result["reasons"] == []


def test_retraining_trigger_fires_on_widespread_input_drift(cfg):
    """輸入層面的訊號：即使盛行率沒變，大量特徵漂移也要告警。"""
    rng = np.random.default_rng(7)
    n = 1500
    cols = [f"f{i:03d}" for i in range(10)]
    labels = np.r_[np.ones(150), np.zeros(n - 150)]

    reference = pd.DataFrame({**{c: rng.normal(size=n) for c in cols}, "fail": labels})
    # 十個特徵全部平移，盛行率完全不動
    current = pd.DataFrame({**{c: rng.normal(5, 1, n) for c in cols}, "fail": labels})

    result = drift.retraining_trigger(reference, current, cfg, cols=cols)
    assert result["retrain"] is True
    assert result["action_flipped"] is False, "這個案例的盛行率不該變"
    assert result["input_drifted"] is True
    assert result["psi_features_major"] == 10
    assert any("輸入漂移" in r for r in result["reasons"])


def test_auc_is_deliberately_not_a_trigger():
    """明確記錄一個設計決定：AUC 不能當觸發器。

    在 33 個正樣本的量級上，可偵測的最小 AUC 約 0.64（見 stats.min_detectable_auc），
    所以 AUC 的抖動大於任何真實變化，拿它當觸發器只會製造假警報。
    這個測試把這個決定釘在程式碼裡，避免有人日後「順手」加進去。
    """
    from secom import stats

    source = drift.retraining_trigger.__doc__
    assert "AUC" in source and "假警報" in source, "設計決定的說明被移除了"

    detectable = stats.min_detectable_auc(33, 673)["min_detectable_auc"]
    assert detectable > 0.6, (
        f"可偵測最小 AUC 變成 {detectable:.3f} —— 若樣本量已經足以偵測小效果，"
        "就該重新考慮把 AUC 納入觸發條件"
    )
