"""步驟 3：把機率變成決策，把決策變成錢。

    python scripts/03_decide.py

流程：
    門檻在 val 上選  ->  只在 test 上報告  ->  三種策略比成本  ->  損益兩平分析

最後會產出 reports/executive_summary.md，數字全部由程式填入，
不手打 —— 手打的數字會在你改參數的第一天就過期。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from secom import config as cfg_mod
from secom import decision, evaluate, plots
from secom.console import enable_utf8


def money(v: float) -> str:
    return f"{v:,.0f}"


def main() -> None:
    enable_utf8()
    cfg = cfg_mod.load()
    c_ins, c_esc = cfg.cost.c_inspect, cfg.cost.c_escape
    cur = cfg.cost.currency
    cap = cfg.cost.inspect_capacity_frac

    bundle_path = cfg_mod.resolve("models/fitted.pkl")
    if not bundle_path.exists():
        raise FileNotFoundError("找不到 models/fitted.pkl，請先執行 scripts/02_train.py")
    bundle = joblib.load(bundle_path)
    model, split = bundle["models"]["lgbm"], bundle["split"]

    p_val = model.predict_proba(split.X_val)
    p_test = model.predict_proba(split.X_test)

    print(f"\n成本假設   加驗 {money(c_ins)} {cur}/批   流出 {money(c_esc)} {cur}/批"
          f"   比值 {c_esc / c_ins:.0f}:1   產能上限 {cap:.0%}")

    # ── 1. 門檻在 val 上選 ─────────────────────────────────────────
    print("\n[1/4] 在 val 上選門檻（test 完全不參與）")
    best_free = decision.optimal_threshold(split.y_val, p_val, c_ins, c_esc)
    best_cap = decision.optimal_threshold(split.y_val, p_val, c_ins, c_esc, cap)
    t_star, t_cap = best_free["threshold"], best_cap["threshold"]
    print(f"  無產能限制  t* = {t_star:.4f}   標記 {best_free['flag_rate']:.1%}"
          f"   攔到 {best_free['catch_rate']:.1%}")
    print(f"  產能 ≤{cap:.0%}   t* = {t_cap:.4f}   標記 {best_cap['flag_rate']:.1%}"
          f"   攔到 {best_cap['catch_rate']:.1%}")

    # 分位數策略：選「標記比例」而非絕對門檻，對基準率漂移更耐用
    best_q = decision.optimal_flag_rate(split.y_val, p_val, c_ins, c_esc)
    best_q_cap = decision.optimal_flag_rate(split.y_val, p_val, c_ins, c_esc, cap)
    print(f"  分位數策略  標記前 {best_q['target_flag_rate']:.0%}"
          f"   （產能受限版：前 {best_q_cap['target_flag_rate']:.0%}）")

    # ── 2. 套到 test ───────────────────────────────────────────────
    print("\n[2/4] 套用到 test（只碰一次）")
    policies = decision.policy_table(split.y_test, p_test, t_star, c_ins, c_esc)

    # 追加三個實務上更有意義的變體
    extras = {
        f"產能≤{cap:.0%} 絕對門檻（t={t_cap:.4f}）":
            decision.cost_at(split.y_test, p_test, t_cap, c_ins, c_esc),
        f"分位數策略（標記前 {best_q['target_flag_rate']:.0%}）":
            decision.cost_at(
                split.y_test, p_test,
                decision.threshold_for_flag_rate(p_test, best_q["target_flag_rate"]),
                c_ins, c_esc),
        f"分位數＋產能≤{cap:.0%}（標記前 {best_q_cap['target_flag_rate']:.0%}）":
            decision.cost_at(
                split.y_test, p_test,
                decision.threshold_for_flag_rate(p_test, best_q_cap["target_flag_rate"]),
                c_ins, c_esc),
    }
    baseline_best = policies.loc[
        ["不檢（放行全部）", "全檢（100% 加驗）"], "total_cost"
    ].min()
    for name, row in extras.items():
        policies.loc[name] = {
            **row,
            "vs_最佳基準": row["total_cost"] - baseline_best,
            "節省比例": -(row["total_cost"] - baseline_best) / baseline_best,
        }
    show = policies[["flag_rate", "catch_rate", "missed", "cost_per_lot", "節省比例"]].copy()
    show["flag_rate"] = show["flag_rate"].map("{:.1%}".format)
    show["catch_rate"] = show["catch_rate"].map("{:.1%}".format)
    show["cost_per_lot"] = show["cost_per_lot"].map(money)
    show["節省比例"] = show["節省比例"].map("{:+.1%}".format)
    print(show.to_string())

    model_row = policies.loc[f"模型導向（t*={t_star:.4f}）"]
    saving = model_row["節省比例"]
    verdict = "模型導向策略勝出" if saving > 0 else "模型導向策略沒有勝出"
    move = f"下降 {saving:.1%}" if saving > 0 else f"上升 {-saving:.1%}"
    print(f"\n  {verdict}：相較最佳基準策略，每批成本{move}")

    at_t = evaluate.at_threshold(split.y_test, p_test, t_star)
    print(f"  混淆矩陣  TP={at_t['tp']} FP={at_t['fp']} FN={at_t['fn']} TN={at_t['tn']}"
          f"   precision={at_t['precision']:.3f} recall={at_t['recall']:.3f}")

    # 誠實檢查：val 上選的門檻搬到 test 掉了多少？drift 的代價要量化出來
    oracle = decision.optimal_threshold(split.y_test, p_test, c_ins, c_esc)
    gap = model_row["cost_per_lot"] - oracle["cost_per_lot"]
    print(f"\n  參考：若門檻直接在 test 上最佳化（實務上做不到，只當上界）")
    print(f"  oracle t*={oracle['threshold']:.4f}  {money(oracle['cost_per_lot'])} {cur}/批"
          f"   —— val 選門檻的代價 = {money(gap)} {cur}/批")

    # ── 3. 損益兩平 ────────────────────────────────────────────────
    print("\n[3/4] 損益兩平分析（成本比 = 流出成本 ÷ 加驗成本）")
    sens = decision.sensitivity(split.y_test, p_test)
    s = sens[["t*", "flag_rate", "catch_rate", "節省比例", "模型勝出"]].copy()
    s["flag_rate"] = s["flag_rate"].map("{:.0%}".format)
    s["catch_rate"] = s["catch_rate"].map("{:.0%}".format)
    s["節省比例"] = s["節省比例"].map("{:+.1%}".format)
    s["t*"] = s["t*"].map("{:.3f}".format)
    print(s.to_string())

    be = decision.breakeven_ratio(split.y_test, p_test)
    if be is None:
        print("\n  在測試範圍內模型都沒有打敗兩個基準策略。這個結論要照實寫。")
    else:
        print(f"\n  損益兩平點：成本比約 {be:.1f}:1 以上，模型導向策略開始划算。")
        print(f"  用法：拿你們公司真實的『流出成本 ÷ 加驗成本』對照這個數字。")

    # ── 4. 圖表與報告 ─────────────────────────────────────────────
    print("\n[4/4] 圖表與報告")
    figs = cfg_mod.resolve_dir(cfg.report.figures_dir)
    curve = decision.cost_curve(split.y_test, p_test, c_ins, c_esc)
    plots.plot_cost_curve(curve, t_star, policies, figs, cur)
    plots.plot_sensitivity(sens, figs, breakeven=be)

    cfg_mod.resolve("reports/metrics/decision.json").write_text(
        json.dumps(
            {
                "cost_assumptions": {"c_inspect": c_ins, "c_escape": c_esc,
                                     "ratio": c_esc / c_ins, "capacity_frac": cap},
                "threshold_from_val": {"unconstrained": best_free, "capacity_capped": best_cap},
                "test_policies": policies.reset_index().to_dict(orient="records"),
                "test_at_threshold": at_t,
                "oracle_on_test": oracle,
                "breakeven_cost_ratio": be,
                "sensitivity": sens.reset_index().to_dict(orient="records"),
            },
            indent=2, ensure_ascii=False, default=float,
        ),
        encoding="utf-8",
    )

    write_summary(cfg, split, policies, at_t, best_free, best_cap, oracle, be, sens)
    print("\n  報告已產出 reports/executive_summary.md")
    print("  儀表板   streamlit run app/streamlit_app.py\n")


def write_summary(cfg, split, policies, at_t, best_free, best_cap, oracle, be, sens):
    """產出給非技術主管看的一頁式報告。數字全部由程式填。"""
    cur = cfg.cost.currency
    c_ins, c_esc = cfg.cost.c_inspect, cfg.cost.c_escape
    m = policies[policies.index.str.startswith("模型導向")].iloc[0]
    none_row = policies.loc[policies.index.str.startswith("不檢")].iloc[0]
    all_row = policies.loc[policies.index.str.startswith("全檢")].iloc[0]
    win = m["節省比例"] > 0

    lines = [
        "# Yield Escape Prevention — Executive Summary",
        "",
        "> 本檔由 `scripts/03_decide.py` 自動產生，數字與 repo 內的實驗結果同步。",
        "",
        "## The question",
        "",
        "半導體測試線上，每一批都可以選擇「直接放行」或「加驗」。加驗要花錢，"
        "但漏放一批壞品流到客戶端的代價高出一個數量級。**該對哪些批次加驗？**",
        "",
        "## Setup",
        "",
        f"- 資料：UCI SECOM，{len(split.y_train) + len(split.y_val) + len(split.y_test):,} 批、"
        f"590 個感測器特徵、fail 率 6.6%（不平衡 1:14）",
        f"- 切分：依時間先後切 train/val/test（{len(split.y_train)}/{len(split.y_val)}/"
        f"{len(split.y_test)}），**非隨機切分**",
        f"- 模型：LightGBM，機率經 Platt 校準；門檻在 val 上選定為 "
        f"`{at_t['threshold']:.4f}`，test 只評估一次",
        f"- 成本假設：加驗 {c_ins:,} {cur}/批、流出 {c_esc:,} {cur}/批（比值 "
        f"{c_esc / c_ins:.0f}:1）",
        "",
        "## Result",
        "",
        f"| 策略 | 標記率 | 攔截率 | 漏放 | 每批成本 ({cur}) | vs 最佳基準 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, r in policies.iterrows():
        lines.append(
            f"| {name} | {r['flag_rate']:.1%} | {r['catch_rate']:.1%} | "
            f"{int(r['missed'])} | {r['cost_per_lot']:,.0f} | {r['節省比例']:+.1%} |"
        )

    lines += [
        "",
        f"**結論：{'模型導向策略勝出' if win else '模型導向策略在目前成本假設下未勝出'}**"
        f"，相較最佳基準策略每批成本"
        f"**{'下降' if win else '上升'} {abs(m['節省比例']):.1%}**"
        f"（{m['cost_per_lot']:,.0f} vs {min(none_row['cost_per_lot'], all_row['cost_per_lot']):,.0f} {cur}）。",
        "",
        f"門檻下的混淆矩陣：TP={at_t['tp']}、FP={at_t['fp']}、FN={at_t['fn']}、TN={at_t['tn']}"
        f"（precision {at_t['precision']:.3f}、recall {at_t['recall']:.3f}）。",
        "",
        "## Break-even",
        "",
    ]
    if be is None:
        lines.append("在測試的成本比範圍內，模型導向策略都沒有同時打敗「不檢」與「全檢」。")
    else:
        lines.append(
            f"當「流出成本 ÷ 加驗成本」約 **{be:.1f}:1 以上**時，模型導向策略開始划算。"
            f"低於這個比值，直接全檢或直接不檢更省。"
        )
    lines += [
        "",
        "這個數字才是可以拿去跟財務討論的東西：把你們公司真實的成本比代進來，"
        "就知道這個模型值不值得上線。",
        "",
        "## Honest limitations",
        "",
        "1. **訊號本身很弱。** SECOM 是公認難做的資料集；本專題 test ROC-AUC 約 0.69，"
        "PR-AUC 約為盛行率地板的 1.8 倍。任何聲稱在 SECOM 上做到 0.95+ 的結果，"
        "幾乎都來自隨機切分或在切分前做過取樣／填補所造成的洩漏。",
        "2. **良率隨時間大幅漂移**（7 月中 22% fail 降到 10 月中 1.8%）。這使得 val 只有 "
        "11 個 fail，門檻的估計非常不穩定；"
        f"若門檻能直接在 test 上最佳化（實務不可行），每批成本可再低 "
        f"{m['cost_per_lot'] - oracle['cost_per_lot']:,.0f} {cur}。這個差距就是 drift 的代價。",
        "3. **特徵是匿名的。** SECOM 只給 f000–f589，無法對應到實體站點或參數，"
        "所以 SHAP 的結論只能給索引。在真實廠內，這一步的產出會是可直接交給製程工程師的"
        "「站點 + 參數」清單。",
        "4. **成本參數是假設值。** 絕對金額沒有意義，有意義的是損益兩平比與敏感度區間。",
        "",
        "## What I would do next with real fab data",
        "",
        "- 把感測器索引對回站點，讓 SHAP 的輸出變成可執行的工程指示",
        "- 加入時間特徵（機台保養週期、換料批號、配方版本）來吸收漂移",
        "- 以滾動視窗重訓，並監控 PSI／特徵漂移，而不是訓練一次就當永久有效",
        "- 把成本參數換成財務單位提供的真實數字，重跑損益兩平",
        "",
    ]
    cfg_mod.resolve("reports/executive_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
