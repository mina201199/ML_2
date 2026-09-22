"""不確定性量化的測試。

為什麼補這個檔案：README 用粗體列出四個量化證據，其中三個出自
`secom/stats.py` —— bootstrap 區間、置換檢定 p 值、以及「需要多少不良品」的
檢定力推算。而在這個檔案存在之前，`stats.py` 的五個函式只有
`min_detectable_auc` 被 `test_drift.py` 順手用到一次。

也就是說，洩漏防護測試守住了資料處理那一層，**結論那一層是裸的**。
整份報告最關鍵的兩個數字（相對節省 −23.5% 到 +15.9%、p = 0.52）
沒有任何測試在看著它們。

這裡的測試刻意全部走解析解或退化情境，不依賴模型、不依賴 `data/`：
`stats._cells` 每折只讀 caught / missed / n_flagged / n_eval 四個鍵，
所以合成幾個 dict 就能把公式釘死。
"""

import pytest

from secom import config, stats


def make_cfg(c_inspect: float = 2000.0, c_escape: float = 60000.0):
    """成本刻意寫死在測試裡，不讀 config.yaml。

    這些測試驗的是公式，不是專案當下的成本假設。若改 config.yaml 會讓這裡變紅，
    那就是把「假設」和「實作」綁在一起了 —— 成本假設本來就該能自由調整。
    """
    return config.Config({"cost": {"c_inspect": c_inspect, "c_escape": c_escape}})


def make_fold(n_eval: int, n_fail: int, n_flagged: int, caught: int, fold: int = 1):
    """一折評估窗的最小表示。missed 由 n_fail 推出，保證 2x2 自洽。"""
    return {
        "fold": fold,
        "n_eval": n_eval,
        "n_flagged": n_flagged,
        "caught": caught,
        "missed": n_fail - caught,
    }


# ── 1. 2x2 還原：格數不自洽必須立刻爆 ──────────────────────────────
# 這是所有下游計算的地基。caught 比 n_flagged 還多之類的輸入如果靜默通過，
# 後面的成本、區間、p 值全都是在算垃圾，而且看起來完全正常。


def test_inconsistent_fold_counts_raise_instead_of_computing_garbage():
    """caught > n_flagged 是不可能的（被攔截的批次一定在被標記的那一組裡）。"""
    bad = [make_fold(n_eval=100, n_fail=10, n_flagged=3, caught=5)]
    with pytest.raises(ValueError, match="不自洽"):
        stats.bootstrap_cost_ci(bad, make_cfg(), "inspect_all", n_boot=10)


# ── 2. 置換檢定 ────────────────────────────────────────────────────


def test_permutation_p_value_can_never_be_exactly_zero():
    """p 用 (1 + 計數) / (1 + n_perm) 修正，所以下界是 1/(1+n_perm) 而不是 0。

    完美排序（配額 10 筆全中 10 筆不良品）是這個檢定能看到的最強訊號。
    即使如此 p 也不該是 0 —— 有限次重抽沒有資格宣稱機率為零。
    """
    perfect = [make_fold(n_eval=100, n_fail=10, n_flagged=10, caught=10)]
    n_perm = 200
    out = stats.permutation_test_vs_random(perfect, make_cfg(), n_perm=n_perm, seed=42)

    assert out["p_value_one_sided"] > 0, "p = 0 是有限次重抽不該給出的結論"
    assert out["p_value_one_sided"] == pytest.approx(1.0 / (1 + n_perm))
    assert out["p_value_one_sided"] < 0.01, "完美排序應該要被判定為顯著"


def test_permutation_does_not_claim_significance_for_an_uninformative_ranking():
    """攔截數剛好等於超幾何期望值時，p 必須落在 0.5 附近。

    這正是本專題實測的情境（p = 0.52）。如果這個退化案例都能給出小 p 值，
    那份「與同配額隨機抽驗無法區分」的結論就沒有意義了。
    """
    # N=200、K=20 筆不良品、抽 40 筆 → 期望攔截 40 × 20/200 = 4 筆
    null_like = [make_fold(n_eval=200, n_fail=20, n_flagged=40, caught=4)]
    out = stats.permutation_test_vs_random(null_like, make_cfg(), n_perm=20000, seed=42)

    assert 0.45 < out["p_value_one_sided"] < 0.75, (
        f"無資訊排序的 p = {out['p_value_one_sided']:.3f}，應該在 0.5 附近"
    )


def test_permutation_null_mean_matches_the_hypergeometric_expectation():
    """虛無分布的平均成本要對得上解析期望值。

    虛無假設下攔截數服從 Hypergeometric(N, K, n_flagged)，期望值 n_flagged·K/N。
    這條斷言確認「不真的去打亂資料、直接抽超幾何」這個捷徑是對的 ——
    抽錯分布的話 p 值會整體平移，而且不會有任何跡象。
    """
    n_eval, n_fail, n_flagged = 200, 20, 40
    c_ins, c_esc = 2000.0, 60000.0
    folds = [make_fold(n_eval, n_fail, n_flagged, caught=4)]
    out = stats.permutation_test_vs_random(folds, make_cfg(c_ins, c_esc),
                                           n_perm=20000, seed=42)

    expected_caught = n_flagged * n_fail / n_eval
    analytic = (n_flagged * c_ins + (n_fail - expected_caught) * c_esc) / n_eval
    assert out["null_cost_mean"] == pytest.approx(analytic, rel=0.01)


def test_permutation_is_reproducible_under_a_fixed_seed():
    """同一個 seed 必須給出同一個 p。報告裡的 0.52 要能被重現。"""
    folds = [make_fold(n_eval=200, n_fail=20, n_flagged=40, caught=6)]
    a = stats.permutation_test_vs_random(folds, make_cfg(), n_perm=2000, seed=7)
    b = stats.permutation_test_vs_random(folds, make_cfg(), n_perm=2000, seed=7)
    assert a["p_value_one_sided"] == b["p_value_one_sided"]


# ── 3. bootstrap 成本區間 ──────────────────────────────────────────


def test_bootstrap_reports_a_tie_as_a_tie_not_as_a_win():
    """模型退化成「全檢」時，與 inspect_all 的差必須恆為 0，而且不算贏。

    這條是頭條結論的支點：`majority/robust` 的 2,000 就是全檢換了個名字，
    它跟零模型策略打平。`share_of_resamples_model_cheaper` 用嚴格大於，
    所以平手回報 0.0 —— 如果哪天這裡變成 0.5 或 1.0，
    README 那句「它是平手而不是勝出」就不再成立。
    """
    always_inspect = [make_fold(n_eval=100, n_fail=10, n_flagged=100, caught=10)]
    out = stats.bootstrap_cost_ci(always_inspect, make_cfg(), "inspect_all",
                                  n_boot=500, seed=42)

    assert out["cost_difference_ci"] == [0.0, 0.0]
    assert out["share_of_resamples_model_cheaper"] == 0.0, "平手不能被算成勝出"


def test_bootstrap_interval_stays_above_zero_for_a_uniformly_cheaper_model():
    """反向對照：真的每次重抽都更便宜時，區間要整段在 0 以上。

    沒有這一條，上面那個「平手回報 0」的測試可以被一個永遠回傳 0 的爛實作滿足。
    """
    # 配額 10 筆全中，對照「完全不檢」：模型 200／筆 vs 基準 6,000／筆
    strong = [make_fold(n_eval=100, n_fail=10, n_flagged=10, caught=10)]
    out = stats.bootstrap_cost_ci(strong, make_cfg(), "inspect_none",
                                  n_boot=2000, seed=42)

    lo, hi = out["cost_difference_ci"]
    assert lo > 0 and hi > lo, f"區間 [{lo}, {hi}] 應該整段在 0 以上"
    assert out["share_of_resamples_model_cheaper"] > 0.99


def test_bootstrap_uses_the_reference_action_it_was_given():
    """基準策略是傳進來的那一個，不會在重抽裡被偷偷換成最便宜的那個。

    每次重抽都重新挑最便宜的基準，會把「挑選」的樂觀偏誤混進區間 ——
    區間會系統性偏向「模型沒有輸那麼多」。這裡用同一組折、只換基準，
    確認兩個答案不同（也就是參數真的有被用到）。
    """
    folds = [make_fold(n_eval=100, n_fail=10, n_flagged=10, caught=10)]
    none = stats.bootstrap_cost_ci(folds, make_cfg(), "inspect_none", n_boot=500, seed=42)
    every = stats.bootstrap_cost_ci(folds, make_cfg(), "inspect_all", n_boot=500, seed=42)

    assert none["reference_action"] == "inspect_none"
    assert every["reference_cost_ci"] == [2000.0, 2000.0], "全檢成本與盛行率無關"
    assert none["reference_cost_ci"][1] > every["reference_cost_ci"][1]

    with pytest.raises(ValueError, match="未知的零模型策略"):
        stats.bootstrap_cost_ci(folds, make_cfg(), "inspect_the_cheap_ones", n_boot=10)


def test_bootstrap_capacity_quota_is_an_integer_number_of_lots():
    """產能上限是「幾筆」而不是「幾成」：rate = floor(n · frac) / n。

    用零 fail 的窗讓盛行率恆為 0，基準成本就退化成 rate · c_inspect，
    區間塌成一個點，可以精確比對。94 筆、上限 20% → 18 筆，不是 18.8 筆。
    保留零 fail 窗本身也是評估協議寫明的行為。
    """
    zero_fail = [make_fold(n_eval=94, n_fail=0, n_flagged=30, caught=0)]
    out = stats.bootstrap_cost_ci(zero_fail, make_cfg(), "random_at_capacity",
                                  capacity_frac=0.2, n_boot=200, seed=42)

    expected = 18 / 94 * 2000.0
    assert out["reference_cost_ci"] == [pytest.approx(expected), pytest.approx(expected)]
    assert out["reference_cost_ci"][0] < 0.2 * 2000.0, "配額沒有被向下取整到整數筆"


# ── 4. 檢定力：這個設計本來就看得見多大的訊號 ──────────────────────


def test_min_detectable_auc_falls_as_positives_accumulate():
    """樣本量越大、看得見的效果越小。單調性壞掉就代表公式打錯了。"""
    values = [stats.min_detectable_auc(n, n * 20)["min_detectable_auc"]
              for n in (33, 69, 275)]
    assert values == sorted(values, reverse=True)
    assert all(v > 0.5 for v in values), "可偵測的最小 AUC 必須在 0.5 以上"


def test_min_detectable_auc_matches_the_reported_0_644():
    """釘住 README 的 0.644：33 筆不良品、673 筆正常品。"""
    out = stats.min_detectable_auc(33, 673)
    assert out["min_detectable_auc"] == pytest.approx(0.644, abs=0.001)


def test_min_detectable_auc_declines_to_answer_for_an_empty_arm():
    """其中一邊沒有樣本時回 None，不回一個看起來很正常的數字。"""
    assert stats.min_detectable_auc(0, 100)["min_detectable_auc"] is None
    assert stats.min_detectable_auc(10, 0)["min_detectable_auc"] is None


def test_required_positives_round_trips_through_min_detectable_auc():
    """兩個方向必須互為反函數，並且對得上 README 寫的 69 / 1,468 與 275 / 5,872。

    這是整份「所以你需要多少資料」的答案。兩支函式各自算一遍再代回去，
    任何一邊的公式被動過都會在這裡露出來。
    """
    prevalence = 33 / 706          # 回測四個評估窗的合併盛行率
    for target, positives, lots in ((0.60, 69, 1468), (0.55, 275, 5872)):
        out = stats.required_positives_for_auc(target, prevalence)
        assert (out["required_positives"], out["required_lots"]) == (positives, lots)

        back = stats.min_detectable_auc(positives, lots - positives)
        assert back["min_detectable_auc"] == pytest.approx(target, abs=0.001)


def test_required_positives_declines_to_answer_for_a_non_effect():
    """target_auc <= 0.5 沒有「需要幾筆」這種答案，回 None 而不是一個大數字。"""
    for target in (0.50, 0.45):
        assert stats.required_positives_for_auc(target, 0.05)["required_positives"] is None
    assert stats.required_positives_for_auc(0.6, 0.0)["required_positives"] is None


def test_cost_resolution_is_one_failure_worth_of_cost_per_lot():
    """成本指標的解析度 = c_escape / 筆數，也就是「多攔一筆」值多少。

    706 筆、漏放 60,000 → 85 元/筆。報告用它來說明為什麼百分比的小數位無意義。
    """
    out = stats.cost_resolution(706, 60000.0)
    assert out["twd_per_lot_per_failure"] == pytest.approx(60000.0 / 706)
    assert stats.cost_resolution(0, 60000.0)["twd_per_lot_per_failure"] is None
