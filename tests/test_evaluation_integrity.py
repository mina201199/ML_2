"""Regression checks for evaluation boundaries and deployable decisions."""
import copy

import numpy as np
import pandas as pd

from secom import backtest, config, decision, models


def test_sensitivity_selects_on_validation_not_evaluation_labels():
    y_val, p_val = [0, 0, 1, 1], [.1, .2, .8, .9]
    p_test = [.1, .2, .8, .9]
    a = decision.sensitivity([1, 1, 0, 0], p_test,
                             y_select=y_val, p_select=p_val, ratios=[3])
    b = decision.sensitivity([0, 0, 1, 1], p_test,
                             y_select=y_val, p_select=p_val, ratios=[3])
    assert a.loc[3, 't*'] == b.loc[3, 't*'] == .8
    assert a.loc[3, 'cost_模型'] == 8
    assert b.loc[3, 'cost_模型'] == 2


def test_quantile_ties_never_exceed_requested_capacity():
    p = np.full(10, .2)
    t = decision.threshold_for_flag_rate(p, .2)
    assert (p >= t).sum() <= 2


def test_zero_failures_still_trigger_conservative_inspection():
    result = decision.robust_flag_rate(np.zeros(10), np.linspace(.1, .9, 10), 1, 30)
    assert result['target_flag_rate'] == 1


def test_oof_preprocessing_ignores_later_feature_values(monkeypatch):
    cfg = copy.deepcopy(config.load())
    cfg['model']['lgbm']['min_child_samples'] = 2
    cfg['model']['lgbm']['n_jobs'] = 1
    rng = np.random.default_rng(8)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=['f000','f001','f002','f003'])
    X.loc[:39, 'f003'] = np.nan
    y = pd.Series(np.tile([0, 0, 0, 1], 20))
    poison = X.copy()
    poison.loc[40:, 'f003'] = np.nan
    from secom.pipeline import DropHighMissing
    seen = []
    original = DropHighMissing.fit
    def record_fit(self, frame, labels=None):
        seen.append(len(frame))
        return original(self, frame, labels)
    monkeypatch.setattr(DropHighMissing, 'fit', record_fit)
    a = models.oof_raw_predictions(X, y, cfg, 8)
    assert seen == [20, 40, 60]
    b = models.oof_raw_predictions(poison, y, cfg, 8)
    np.testing.assert_allclose(a[20:40], b[20:40])


def test_zero_failure_evaluation_windows_are_retained():
    cfg = config.load()
    rng = np.random.default_rng(2)
    df = pd.DataFrame({'f000': rng.normal(size=200),
                       'fail': np.r_[np.tile([0,1], 40), np.zeros(120)],
                       'ts': pd.date_range('2020-01-01', periods=200, freq='h')})
    bt = backtest.rolling_origin(df, cfg, n_folds=2, model_name='majority', verbose=False)
    assert len(bt) == 2
    assert bt.n_eval.sum() == 90
    assert bt.n_fail_eval.sum() == 0
    assert np.isfinite(bt.cost_model).all()


def test_sql_augmentation_ignores_future_values_and_labels():
    from secom.sql import augment_time_features
    df = pd.DataFrame({'ts': pd.date_range('2020-01-01', periods=30, freq='h'),
                       'fail': np.tile([0, 1], 15), 'f000': np.arange(30, dtype=float)})
    altered = df.copy()
    altered.loc[20:, ['f000', 'fail']] = 999
    a, b = augment_time_features(df), augment_time_features(altered)
    assert a.shape[1] == 5
    pd.testing.assert_frame_equal(a.iloc[:20], b.iloc[:20])
    assert a['f900'].iloc[20] == 10.5


def test_random_baseline_uses_integer_capacity():
    table = decision.policy_table([1]*7, np.linspace(.1,.9,7), 1.1, 1, 30,
                                   y_select=[1]*7, capacity_frac=.2)
    row = table.loc[table.index.str.startswith('隨機')].iloc[0]
    assert row.n_flagged == 1
    assert row.total_cost == 181


def test_summary_renders_undefined_metrics(tmp_path, monkeypatch):
    from secom import provenance, reporting
    cfg = config.load()
    policy_row = dict(policy='模型導向', flag_rate=0, catch_rate=None,
                      missed=0, cost_per_lot=0, 節省比例=None)
    source = {'cases': {'unconstrained': {'test_policies': [policy_row]}}}
    for case in source['cases'].values():
        for row in case['test_policies']:
            row['catch_rate'] = None
            row['節省比例'] = None
    monkeypatch.setattr(config, 'ROOT', tmp_path)
    monkeypatch.setattr(config, 'load', lambda: cfg)
    provenance.write_json('reports/metrics/decision.json', source)
    reporting.write_summary(include_backtest=False)
    assert (tmp_path/'reports/executive_summary.md').exists()


def test_calibration_does_not_change_rank_based_decisions():
    """Platt 是單調變換，所以分位數策略的決策必須完全不變。

    報告主張「校準只影響絕對門檻策略，不影響分位數與保守分位數策略」。這是一個
    可以驗證的主張，不該只寫在文字裡 —— 尤其因為 models.py 的 docstring 用
    「期望成本 = P(fail) × 損失」論證校準的必要性，讀者很容易誤以為三個策略
    都吃機率值。
    """
    from secom.models import PlattCalibrator

    rng = np.random.default_rng(11)
    n = 400
    raw = np.clip(rng.beta(2, 20, n), 1e-4, 1 - 1e-4)
    y = (rng.random(n) < raw * 3).astype(int)

    calibrator = PlattCalibrator().fit(raw, y)
    assert calibrator.lr_.coef_[0, 0] > 0, "斜率為負時 Platt 會反轉排序，前提不成立"
    calibrated = calibrator.transform(raw)

    # 同一個目標加驗比例下，被標記的那一組批次必須逐筆相同
    for rate in (0.05, 0.2, 0.5):
        t_raw = decision.threshold_for_flag_rate(raw, rate)
        t_cal = decision.threshold_for_flag_rate(calibrated, rate)
        np.testing.assert_array_equal(
            raw >= t_raw, calibrated >= t_cal,
            err_msg=f"加驗比例 {rate:.0%} 下，校準改變了被標記的批次",
        )

    # 對照組：校準確實改變了機率的尺度，所以「絕對門檻會受影響」才是真的
    assert not np.allclose(raw, calibrated, atol=1e-3), (
        "校準沒有改變機率尺度，那麼「只有絕對門檻策略受影響」這個對照就不成立"
    )
