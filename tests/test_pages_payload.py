"""docs/index.html 的資料區塊必須與 reports/metrics/ 一致。

這一頁是 GitHub Pages 服務的靜態試算頁，任何人點連結就能用。它的數字內嵌在頁面
裡，所以有一個特定的失效模式：**實驗重跑了，但沒有重跑 scripts/06_build_pages.py**，
於是網頁上的數字與報告不一致，而且不會有任何錯誤訊息。

這個測試就是為了讓那件事變成一個會失敗的測試，而不是一個沒人發現的矛盾。
"""

import json
import re

import pytest

from secom import config, decision
from secom.provenance import _clean

PAGE = "docs/index.html"
PAYLOAD_RE = re.compile(
    r'<script id="payload" type="application/json">(.*?)</script>', re.S
)


def _read_json(rel):
    path = config.resolve(rel)
    if not path.exists():
        pytest.skip(f"尚未產生 {rel}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def payload():
    path = config.resolve(PAGE)
    if not path.exists():
        pytest.skip("尚未產生 docs/index.html")
    m = PAYLOAD_RE.search(path.read_text(encoding="utf-8"))
    assert m, "找不到 payload 區塊 —— scripts/06_build_pages.py 的邊界字串可能被改動"
    raw = m.group(1)
    assert raw.strip() not in ("", "{}"), (
        "payload 還是空的，請執行 python scripts/06_build_pages.py"
    )
    return json.loads(raw)


def test_payload_has_no_unescaped_script_terminator(payload):
    """JSON 內容不得含未轉義的 '<'，否則瀏覽器會提早結束 script 區塊。"""
    raw = PAYLOAD_RE.search(
        config.resolve(PAGE).read_text(encoding="utf-8")
    ).group(1)
    assert "<" not in raw, "payload 含未轉義的 '<'，頁面會被截斷"


def test_holdout_matches_scored_artifact(payload):
    """驗證／測試窗的標籤與機率必須逐筆等於 02_train 的產出。"""
    holdout = _read_json("reports/metrics/scored_holdout.json")
    for name in ("val", "test"):
        assert payload[name]["y"] == holdout[name]["y"], f"{name} 標籤不一致"
        assert payload[name]["p"] == holdout[name]["p"], f"{name} 機率不一致"
        assert len(payload[name]["y"]) == payload["windows"][name]["n"]
        assert sum(payload[name]["y"]) == payload["windows"][name]["n_fail"]


def test_escape_inflation_matches_python(payload):
    """內嵌的放大倍數必須等於 decision.conservative_base_rate 算出來的值。

    前端刻意不實作 Clopper-Pearson 的 beta 反函數，改用內嵌常數 ——
    所以那個常數必須被釘住，否則前端的保守策略會靜悄悄地算錯。
    """
    n = payload["windows"]["val"]["n"]
    n_fail = payload["windows"]["val"]["n_fail"]
    expected = decision.conservative_base_rate(n_fail, n, 0.95) / (n_fail / n)
    assert payload["escape_inflation"] == pytest.approx(expected, rel=1e-12)


def test_backtest_numbers_match_the_report(payload):
    """回測表、零模型對照與不確定性都必須等於 backtest.json。"""
    report = _read_json("reports/metrics/backtest.json")

    assert payload["dominance"] == _clean(report["dominance"])
    assert payload["cost_assumptions"] == _clean(report["cost_assumptions"])
    assert payload["display_key"] == report["display_key"]

    assert {r["key"] for r in payload["results"]} == set(report["results"])
    for row in payload["results"]:
        s = report["results"][row["key"]]["summary"]
        assert row["cost_per_lot"] == pytest.approx(s["cost_per_lot"])
        assert row["total_missed"] == s["total_missed"]

    for regime, ref in payload["reference_policies"].items():
        src = report["reference_policies"][regime]
        assert ref["costs"] == _clean(src["costs"])
        assert ref["feasible"] == src["feasible"]
        assert ref["best_action"] == src["best_action"]

    u = report["uncertainty"][report["display_key"]]
    assert payload["uncertainty"]["min_detectable_auc"] == pytest.approx(
        u["power"]["min_detectable_auc"]
    )
    assert payload["uncertainty"]["permutation_p"] == pytest.approx(
        u["permutation_vs_random"]["p_value_one_sided"]
    )


def test_beats_flags_agree_with_dominance(payload):
    """「跨過門檻」的旗標必須來自 dominance，不是另外算一次。

    頁面上這一欄是讀者判斷模型有無價值的依據，所以它與主結論必須同源。
    """
    beaten = set()
    for dom in payload["dominance"].values():
        beaten |= set(dom["configurations_beating_model_free"])
    flagged = {r["key"] for r in payload["results"] if r["beats"]}
    assert flagged == beaten

    n_beat = sum(d["n_beating_model_free"] for d in payload["dominance"].values())
    assert len(flagged) == n_beat


def test_page_states_the_same_verdict_as_the_report(payload):
    """頁面的結論句由 dominance 算出，這裡確認分母是全部配置數。"""
    n_cfg = sum(d["n_configurations"] for d in payload["dominance"].values())
    assert n_cfg == len(payload["results"]), (
        "configurations 總數與 results 列數不一致，頁面的「N / M」會是錯的"
    )


def test_action_labels_cover_every_reference_policy(payload):
    """每個零模型策略都要有中文標籤，否則頁面會顯示原始英文鍵值。"""
    actions = {
        a for ref in payload["reference_policies"].values() for a in ref["costs"]
    }
    actions |= {
        d["best_model_free_action"] for d in payload["dominance"].values()
    }
    missing = actions - set(payload["action_labels"])
    assert not missing, f"缺少標籤：{sorted(missing)}"
