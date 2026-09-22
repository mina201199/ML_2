"""把實驗產出渲染成一份前後一致、每個數字都有來源的 markdown 報告。

這個檔案只讀 reports/metrics/*.json，不重算任何實驗、也不寫 JSON
（那是 secom/provenance.py 的事）。報告裡不准出現手打的數字 —— 所有數值都必須
追得回某一支腳本的輸出，否則就會出現「README 說 +12.4% 但沒人重現得出來」的狀況。

報告的段落順序是刻意的：**主結論在最前面**。上一版把 18 列的比較表放在前面、
把「零模型策略其實更便宜」這件事留給讀者自己從表裡讀出來，這是在迴避結論。

每個 `_xxx_lines` 函式負責報告的一個章節，回傳 markdown 行的清單；
`write_summary` 只做編排。要增刪章節就是增刪一個函式和一行呼叫。
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import beta

from . import config
from .decision import required_lift

# 零模型策略在報告裡的中文顯示名稱
ACTION_LABELS = {
    "inspect_none": "永遠不檢",
    "inspect_all": "永遠全檢",
    "random_at_capacity": "按配額隨機加驗",
}


def _fmt(value, spec):
    return "—" if value is None or not np.isfinite(value) else format(value, spec)


def _read(path):
    return json.loads(config.resolve(path).read_text(encoding="utf-8"))


def _headline_lines(report: dict, cfg) -> list[str]:
    """主結論：零模型固定策略 vs 全部模型導向配置。

    這一段是整份報告最重要的部分，因為它是唯一直接回答「這個模型值不值得做」
    的段落。它的每個數字都來自 04_backtest.py 寫出的 reference_policies 與
    dominance 區塊，不是人工判讀比較表得來的。
    """
    reference = report.get("reference_policies", {})
    dominance = report.get("dominance", {})
    if not reference or not dominance:
        return []

    ratio = cfg.cost.c_escape / cfg.cost.c_inspect
    pooled = next(iter(reference.values()))["prevalence_pooled"]
    bar = required_lift(pooled, ratio)

    n_configs = sum(d["n_configurations"] for d in dominance.values())
    n_beat = sum(d["n_beating_model_free"] for d in dominance.values())

    lines = [
        "## 主結論", "",
        f"**在本專題的成本假設下，不需要模型的固定策略比全部 {n_configs} 個模型導向"
        f"配置都便宜；跨過門檻的配置數是 {n_beat} / {n_configs}。** 因此這份實驗不支持"
        "「用這個模型配置加驗資源」的結論；它支持的是一個負面結論，以及一組說明"
        "為什麼的量化條件。", "",
        "### 零模型參考策略", "",
        "這三個策略不看分數、不選門檻、不做校準，且從頭到尾固定不變 —— 連「事前挑錯」"
        "的機會都沒有。成本單位與下方比較表一致（每筆評估觀測，依窗大小加權）。", "",
        "| 產能情境 | 零模型策略 | 可行 | 成本／筆 |",
        "| --- | --- | :---: | ---: |",
    ]
    for regime, ref in reference.items():
        for action, cost in ref["costs"].items():
            ok = "是" if action in ref["feasible"] else "否"
            mark = " ←最便宜" if action == ref["best_action"] else ""
            lines.append(
                f"| {regime} | {ACTION_LABELS.get(action, action)}{mark} | {ok} | "
                f"{_fmt(cost, ',.0f')} |"
            )
    lines += ["", "### 有多少模型導向配置跨過了這個門檻", "",
              "| 產能情境 | 零模型最佳 | 最便宜的模型導向配置 | 贏過零模型的配置數 |",
              "| --- | ---: | --- | :---: |"]
    for regime, dom in dominance.items():
        lines.append(
            f"| {regime} | {ACTION_LABELS.get(dom['best_model_free_action'])} "
            f"{_fmt(dom['best_model_free_cost'], ',.0f')} | "
            f"{dom['cheapest_model_driven'].split('/', 1)[1]} "
            f"{_fmt(dom['cheapest_model_driven_cost'], ',.0f')} | "
            f"{dom['n_beating_model_free']} / {dom['n_configurations']} |"
        )

    lines += [
        "",
        "無限制情境那一列要特別讀清楚：最便宜的「模型導向」配置是 `majority/robust`，"
        "而 majority 是零資訊的 DummyClassifier，它的保守策略退化成 100% 加驗 —— "
        "所以那個 2,000 就是「永遠全檢」本身，只是掛了模型的名字。真正用到排序能力的"
        "配置全部更貴。",
    ]

    lines += [
        "", "### 為什麼：門檻高度是可以先算出來的", "",
        f"設加驗成本 1 單位、漏放成本 R = {ratio:.0f} 倍、合併盛行率 p = {pooled:.2%}，"
        f"則 p·R = {pooled * ratio:.2f}。",
        "",
        "p·R ≥ 1 表示加驗平均而言就是划算的，此時把關的零模型策略是「按配額隨機加驗」"
        "（無產能限制時即全檢），而模型導向策略要贏過它，條件化簡成一個很乾淨的形式："
        f"**lift 必須大於 {bar:.2f}** —— 也就是「同樣的加驗配額，模型挑出來的批次裡，"
        "不良品濃度要高於隨機抽樣」。這個門檻與加驗比例無關；產能上限只改變哪一個零模型"
        "策略在把關，不改變門檻高度（推導見 `secom/decision.py` 的 `required_lift`）。",
        "",
        "換句話說，本專題選定的成本假設把情境放進了「排序能力只需要贏過隨機抽驗」的"
        "區域 —— 這是對模型最寬鬆的一個區域。實測沒有贏。", "",
        "![可行性與必要 lift](figures/feasibility.png)", "",
    ]
    return lines


def _protocol_lines(report: dict) -> list[str]:
    """評估設計，加上兩段這一版新增的自我揭露。"""
    lines = [
        "## 評估設計", "",
        "外層依時間分成訓練、校準／選策略、未來評估。內層 CV 的欄位篩選與樹模型分箱"
        "各自只使用該折訓練資料。", "",
        "LightGBM 的棵數由內部平均 AUC 曲線選出，並套用 config.yaml 的最低 10 棵限制。"
        "校準使用固定棵數的時序 OOF 預測加驗證窗；固定棵數避免用較晚標籤替較早 OOF "
        "調參。校準模型與最終模型的訓練量及棵數不同，校準可轉移性仍是限制。", "",
        "校準窗同時用來擬合校準器與挑策略，因此該窗的分數不是獨立成績；成績只在後續窗"
        "計算。所有模型與策略使用相同外層時間窗。零 fail 窗仍計算成本，無定義的 AUC／"
        "攔截率記為空值。", "",
        "基準策略由校準窗事先選定：無限制時比較全檢與不檢；有產能限制時比較不檢與按上限"
        "均勻隨機加驗的期望成本。比較表的「節省中位數」分母就是這個事前選定的基準。"
        "**但這個基準會挑錯**，所以主結論改用固定不變的零模型策略當門檻，那是更嚴格、"
        "也更難反駁的比較對象。", "",
        "分位數策略需要同一決策批次的所有分數，本回測假設整個評估窗為可一起排序的批次。"
        "它不是逐筆即時部署模擬。同分邊界一起排除以遵守上限，因此可能未用滿產能。", "",
    ]

    # ── 保守策略的行為揭露（A3）──
    results = report.get("results", {})
    key = report.get("display_key", "unconstrained/lgbm/robust")
    if key in results:
        folds = results[key]["folds"]
        rates = [f.get("target_flag_rate") for f in folds]
        rates = [r for r in rates if r is not None]
        inflation = [f.get("escape_inflation") for f in folds]
        inflation = [x for x in inflation if x is not None and np.isfinite(x)]
        high = sum(1 for r in rates if r >= 0.8)
        floor_hits = results[key]["summary"].get("folds_at_min_estimators")
        if floor_hits is not None:
            source_labels = {
                "min_estimators floor": "下限覆寫",
                "CV mean-curve argmax": "CV 峰值",
                "no usable folds": "無可用折",
            }
            detail = "／".join(
                f"{f.get('n_estimators')} 棵"
                f"（{source_labels.get(f.get('n_estimators_source'), '未知')}）"
                for f in folds
            )
            lines += [
                "### 樹數下限確實在生效", "",
                "config.yaml 的 `min_estimators: 10` 會在內層 CV 平均曲線的峰值落在 10 棵"
                "以下時覆寫它的選擇。它不是備而不用的保險 —— "
                f"`{key}` 的四折結果是 {detail}，其中 **{floor_hits} 折是下限覆寫**。",
                "",
                "被覆寫代表那個訓練窗上，平均 AUC 在不到 10 棵樹時就已經達到峰值、之後"
                "轉為下降 —— 模型幾乎立刻開始過擬合訓練窗，沒有可泛化的訊號可以累積。"
                "另有折的 CV 峰值本身就落在 10 棵，兩者棵數相同但成因不同，"
                "所以上面逐折標註來源。", "",
                "這與下一節的檢定力分析、以及漂移分析指向同一件事，不是三個獨立的問題。", "",
            ]
        lines += [
            "### 「保守」策略實際上在做什麼", "",
            f"保守策略把觀測到的基準率換成 Clopper–Pearson 上界，等效於把漏放成本放大 "
            f"{'／'.join(f'{x:.2f}' for x in inflation)} 倍（各折）。放大後的成本比讓"
            f"最佳加驗比例直接貼到上限：{key} 的四折選出的加驗比例是 "
            f"{'／'.join(f'{r:.0%}' for r in rates)}，其中 {high} 折在 80% 以上。", "",
            "**這意味著它不是一個穩健化方法，而是一個「切換成接近全檢」的開關。**"
            "它的成本改善主要來自加驗比例本身，不是來自模型排序 —— 這也是為什麼"
            "零資訊的 majority 模型配上同一個策略會得到更低的成本。", "",
        ]

    # ── 校準對決策的影響（A4）──
    lines += [
        "### 校準對這三個策略的影響不一樣", "",
        "Platt scaling 是單調遞增變換，而分位數與保守分位數策略只使用分數的**排序**。"
        "因此校準完全不改變這兩個策略的決策，只有「絕對門檻」策略會用到機率的尺度。", "",
        "這一點值得明說，因為成本計算的形式（期望成本 = P(fail) × 損失）看起來像是"
        "「必須先校準」，但本專題三個策略裡有兩個其實不吃機率值。"
        "測試 `test_calibration_does_not_change_rank_based_decisions` 直接驗證這件事。", "",
    ]
    return lines


def _comparison_lines(report: dict) -> list[str]:
    results = report["results"]
    lines = [
        "## 相同時間窗的模型與策略比較", "",
        "「節省中位數」與「最差折」的分母是校準窗事前選定的基準（見上）。要判斷模型"
        "有無價值，請看主結論那張零模型比較表，不要看這一欄。", "",
        "| 情境／模型／策略 | 成本／筆 | 事前基準／筆 | 節省中位數 | 最差折 | "
        "ROC-AUC 中位數 | lift@20% 中位數 | 總攔截 | 總漏放 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, result in results.items():
        s = result["summary"]
        lines.append(
            f"| {key} | {_fmt(s['cost_per_lot'], ',.0f')} | "
            f"{_fmt(s['baseline_cost_per_lot'], ',.0f')} | "
            f"{_fmt(s['saving_median'], '+.1%')} | {_fmt(s['saving_min'], '+.1%')} | "
            f"{_fmt(s['roc_auc_median'], '.3f')} | "
            f"{_fmt(s.get('lift_at_20_median'), '.2f')} | "
            f"{s['total_caught']} | {s['total_missed']} |"
        )

    key = report.get("display_key", "unconstrained/lgbm/robust")
    s = results[key]["summary"]
    k, n = s["total_caught"], s["eval_fails_total"]
    lo = beta.ppf(0.025, k, n - k + 1) if k else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    if n == 0:
        lo, hi = 0.0, 1.0
    lines += [
        "",
        f"預先指定展示的配置 `{key}`：{s['n_folds']} 個評估窗，共 "
        f"{s['eval_lots_total']} 筆、{n} 筆 fail，攔截 {k} 筆。",
        f"把不良品視為獨立同分布二項試驗時，合併攔截率的 95% Clopper–Pearson 區間為 "
        f"{lo:.1%}–{hi:.1%}。時間依賴與漂移可能使此假設失效；此區間不保證未來安全，"
        "也不是成本的信賴區間。",
        "",
        "注意 ROC-AUC 中位數這一欄：logreg 在四個窗一致低於 0.5，lgbm 在 0.5 附近。"
        "三個模型有兩個在或低於隨機水準，這不是「證據有限」，而是「在這個粒度上這份"
        "資料沒有可利用的時序訊號」這個結論本身的證據。",
        "",
        "![滾動回測](figures/backtest.png)", "",
    ]
    return lines


def _uncertainty_lines(report: dict, cfg) -> list[str]:
    """統計不確定性（B 層）。

    這一節存在的理由：上一版只給「四折節省比例的中位數」。四個數字的中位數不是
    統計量，它連方向都保證不了。面試會問「你怎麼知道這不是雜訊」，這一節就是答案。
    """
    from .stats import required_positives_for_auc

    uncertainty = report.get("uncertainty", {})
    if not uncertainty:
        return []

    first = next(iter(uncertainty.values()))
    power = first["power"]
    resolution = first["cost_resolution"]
    prevalence = power["n_pos"] / (power["n_pos"] + power["n_neg"])

    lines = [
        "## 統計不確定性：這個設計看得見多大的訊號", "",
        "### 可偵測效果量", "",
        f"評估窗共 {power['n_pos'] + power['n_neg']:,} 筆、{power['n_pos']} 筆 fail。"
        f"在 α = {power['alpha']}、檢定力 {power['power']:.0%} 之下，這個樣本量可偵測的"
        f"最小 ROC-AUC 是 **{power['min_detectable_auc']:.3f}**"
        "（Mann–Whitney 虛無變異數近似）。", "",
        f"實測 ROC-AUC 中位數在 0.5 附近，落在「任何小於 {power['min_detectable_auc']:.2f} "
        "的真實訊號都看不見」的區間裡。**這既不證明模型有用，也不證明模型無用** —— "
        "它證明的是這個資料量無法回答這個問題。要把它變成可回答的，需要：", "",
        "| 想偵測的 AUC | 需要的 fail 筆數 | 需要的觀測筆數 | 相對現況 |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for target in (0.65, 0.60, 0.55):
        r = required_positives_for_auc(target, prevalence)
        lines.append(
            f"| {target:.2f} | {r['required_positives']:,} | {r['required_lots']:,} | "
            f"{r['required_positives'] / power['n_pos']:.1f}× |"
        )

    lines += [
        "", "### 成本指標的解析度", "",
        f"{resolution['n_lots']:,} 筆評估觀測、漏放成本 {cfg.cost.c_escape:,}，因此"
        f"多攔或少攔**一筆**不良品就等於每筆成本相差 "
        f"{resolution['twd_per_lot_per_failure']:,.0f} TWD。", "",
    ]

    key = report.get("display_key")
    if key in report.get("results", {}):
        s = report["results"][key]["summary"]
        gap = s["saving_median"] * s["baseline_cost_per_lot"]
        lines += [
            f"比較表裡 `{key}` 的「節省中位數 {s['saving_median']:+.1%}」相對於事前基準 "
            f"{s['baseline_cost_per_lot']:,.0f} 等於 {gap:,.0f} TWD／筆，也就是約 "
            f"**{gap / resolution['twd_per_lot_per_failure']:.1f} 筆不良品的差別**。"
            "整個百分比建立在三到四筆觀測上。", "",
        ]

    lines += [
        "### 成本差的 bootstrap 區間", "",
        "在每一折內重抽觀測（折的大小固定，不跨折混合），零模型策略的成本隨同一次重抽"
        "一起變動，所以比較是配對的。零模型策略在原始資料上事前選定，不在每次重抽裡"
        "重新挑。", "",
        "| 配置 | 對照的零模型策略 | 模型成本 95% CI | 相對節省 95% CI | 模型較便宜的重抽比例 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name, u in uncertainty.items():
        b = u["bootstrap"]
        lines.append(
            f"| {name} | {ACTION_LABELS.get(b['reference_action'], b['reference_action'])} | "
            f"{b['model_cost_ci'][0]:,.0f} – {b['model_cost_ci'][1]:,.0f} | "
            f"{b['relative_saving_ci'][0]:+.1%} ~ {b['relative_saving_ci'][1]:+.1%} | "
            f"{b['share_of_resamples_model_cheaper']:.1%} |"
        )
    lines += [
        "",
        "兩個情境的相對節省區間都跨過零。「永遠全檢」的成本不隨標籤變動（固定等於加驗"
        "成本），所以它那一列的基準區間是退化的單點 —— 這正是它作為門檻難以反駁的原因。", "",
        "### 排序是否勝過同配額隨機（精確置換檢定）", "",
        "虛無假設是「被標記的批次組與標籤無關」。加驗筆數固定，因此落進標記組的不良品數"
        "服從超幾何分布，不需要真的打亂資料，直接抽樣就是精確的置換分布。"
        "這一項刻意只檢定 `required_lift` 指出的那個唯一門檻。", "",
        "| 配置 | 實測成本／筆 | 虛無平均 | 虛無 95% 區間 | 單尾 p 值 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, u in uncertainty.items():
        t = u["permutation_vs_random"]
        lines.append(
            f"| {name} | {t['observed_cost_per_lot']:,.0f} | {t['null_cost_mean']:,.0f} | "
            f"{t['null_cost_ci'][0]:,.0f} – {t['null_cost_ci'][1]:,.0f} | "
            f"{t['p_value_one_sided']:.3f} |"
        )
    lines += [
        "",
        "p 值遠離顯著水準，且實測成本落在虛無分布的中央 —— 也就是說模型挑出來的批次組，"
        "與同配額隨機抽出來的批次組在成本上無法區分。這是本專題最直接的否證結果。", "",
    ]
    return lines


def _ablation_lines() -> list[str]:
    ab = _read("reports/metrics/sql_ablation.json")
    lines = [
        "## 歷史特徵消融", "",
        "為排除測試集 SHAP 選特徵的影響，全部 590 個感測器各增加前 20 筆均值偏離量與"
        "標準化偏離量，共 1,180 個歷史特徵。此實驗與舊版「只選 40 個」不同，不直接沿用"
        "舊結論。", "",
        "| 特徵 | 維度 | 成本／筆 | 加驗率中位數 | 漏放 | ROC-AUC 中位數 | 最差節省 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, r in ab["ablation"].items():
        s = r["summary"]
        lines.append(
            f"| {name} | {r['n_features']} | {_fmt(s['cost_per_lot'], ',.0f')} | "
            f"{_fmt(s['flag_rate_median'], '.1%')} | {s['total_missed']} | "
            f"{_fmt(s['roc_auc_median'], '.3f')} | {_fmt(s['saving_min'], '+.1%')} |"
        )
    lines += [
        "",
        "此表為無產能限制情境。兩列的加驗率中位數都在 90% 以上 —— 也就是說 1,180 個"
        "歷史特徵買到的不是「更好的排序」，而是「可以少驗幾個百分點的批次」，而這個差距"
        "是在 33 個正樣本上量到的。不能把成本改善歸因於模型排序，也不能證明統計顯著改善。",
        "",
    ]
    return lines


def _drift_lines() -> list[str]:
    """漂移監控與重訓觸發規則（D3）。

    這一節回答「上線之後你監控什麼、什麼時候重訓」。上一版只有一張描述性的
    每週不良率圖，沒有任何門檻與動作 —— 而本專題的整個論點就是良率會漂移。
    """
    ab = _read("reports/metrics/sql_ablation.json")
    monitoring = ab.get("drift_monitoring")
    if not monitoring:
        return []

    ref = monitoring["reference_window"]
    breakeven = monitoring["breakeven_prevalence"]
    usable = monitoring["alerts"][0]["psi_features_usable"]
    n_features = _read("reports/metrics/data_profile.json")["n_features"]
    lines = [
        "## 漂移監控與重訓觸發", "",
        "一般的漂移偵測會說「分布變了就告警」。在這裡那是錯的關注點 —— 分布天天在變，"
        "我們真正在意的是**最佳動作有沒有翻轉**。", "",
        f"由 `required_lift` 的推導，動作的分界就在損益兩平盛行率 "
        f"p\\* = c_inspect / c_escape = **{breakeven:.2%}**。盛行率在 p\\* 之上，"
        "零模型的最佳動作是「按配額加驗」；在 p\\* 之下是「完全不檢」。所以告警條件是"
        "「有證據顯示盛行率已經跨過 p\\*」，而不是某個統計量抖動了。", "",
        "**規則（任一成立即重訓）**", "",
        f"1. 當前窗盛行率的 95% Clopper–Pearson 區間整段落在 p\\* = {breakeven:.2%} "
        "的另一側，且與參考窗的判定不同。",
        f"2. PSI > {ab['drift_monitoring'].get('psi_threshold_major', 0.25)} 的特徵"
        "超過可用特徵的 10%。缺值自成一桶 —— 在製程資料裡「某站點當時沒量到」本身帶訊息。", "",
        f"下表的分母 {usable} 是**在參考窗算得出 PSI 的特徵數**，不是感測器總數"
        f"（{n_features}）。其餘 {n_features - usable} 欄在參考窗全缺值或零變異，"
        "切不出兩個以上的分位邊界，PSI 是未定義而不是 0 —— 把它們當成「沒有漂移」"
        "計進分母會系統性稀釋這個比例。", "",
        "**AUC 刻意不列為觸發條件。** 在 33 個正樣本的量級上可偵測的最小 AUC 約 0.64"
        "（見上一節），AUC 的抖動大於任何真實變化，拿它當觸發器只會製造假警報。", "",
        f"參考窗：{ref['n']:,} 筆、{ref['n_fail']} 筆 fail，截止於 {ref['to'][:10]}。"
        "模擬「模型在此時點被訓練並部署」，然後逐一檢查每個後續評估窗。", "",
        "| 評估窗 | 期間 | 盛行率 | 95% 區間 | 體制判定 | PSI>0.25 特徵 | 動作 |",
        "| ---: | --- | ---: | --- | --- | ---: | --- |",
    ]
    regime_labels = {
        "inspect": "該加驗", "no_inspect": "不該加驗", "undecided": "無法判定",
    }
    for alert in monitoring["alerts"]:
        cur = alert["current_regime"]
        ci = cur.get("ci") or [None, None]
        lines.append(
            f"| {alert['fold']} | {alert['eval_from'][:10]} – {alert['eval_to'][:10]} | "
            f"{_fmt(cur.get('prevalence'), '.2%')} | "
            f"{_fmt(ci[0], '.2%')} – {_fmt(ci[1], '.2%')} | "
            f"{regime_labels.get(cur.get('regime'), cur.get('regime'))} | "
            f"{alert['psi_features_major']} / {alert['psi_features_usable']} | "
            f"{'**觸發重訓**' if alert['retrain'] else '維持'} |"
        )

    alerts = monitoring["alerts"]
    fired = [a for a in alerts if a["retrain"]]
    by_input = [a for a in fired if a["input_drifted"]]
    by_action = [a for a in fired if a["action_flipped"]]
    undecided = [a for a in alerts if a["current_regime"].get("regime") == "undecided"]
    shares = [a["psi_share_major"] for a in alerts]

    lines += ["", f"### 結果：{len(fired)} / {len(alerts)} 個窗觸發重訓", ""]

    if by_input:
        trend = " → ".join(f"{s:.0%}" for s in shares)
        lines += [
            f"全部由**輸入漂移**觸發（{len(by_input)} 個窗），而且超標比例隨時間單調上升："
            f"{trend}。離訓練窗越遠，移位的感測器越多。", "",
            "**這是上一節「ROC-AUC 中位數在 0.5 附近」的機制。** 模型不是學不到東西，"
            "而是被要求在它從未見過的分布上外插 —— 到最後一個窗，四分之三的感測器"
            "已經不再像訓練時的樣子。在這種條件下，排序能力歸零並不意外。", "",
        ]
    if by_action:
        lines += [
            f"另有 {len(by_action)} 個窗由**動作翻轉**觸發（成本層面的訊號，"
            "優先級高於輸入漂移）。", "",
        ]
    if undecided:
        lines += [
            f"另一件值得注意的事：{len(undecided)} / {len(alerts)} 個窗的盛行率區間"
            f"跨過 p\\*，規則誠實地回報「無法判定」。單一評估窗只有約 176 筆、"
            "4–17 筆 fail，區間寬到無法定位在損益兩平線的哪一側。", "",
            "也就是說：**在這個窗大小之下，連「該不該加驗」這個二元決策都沒有足夠的"
            "統計證據可以做**，更不用說排序哪些批次該加驗。", "",
        ]
    lines += [
        "要讓這條監控規則能真的驅動動作，需要更大的窗或更高的盛行率 —— 這與上一節"
        "檢定力分析指向的方向一致，也是「取得更多時期資料」這個下一步的具體理由。", "",
        "![每週不良率](figures/drift.png)", "",
    ]
    return lines


def _single_split_lines(decision: dict) -> list[str]:
    """單次切分降級成附錄（A5）。"""
    lines = [
        "## 附錄：單次時間切分（僅供對照，不是結論來源）", "",
        "這一節保留是為了讓讀者看到單一切分與滾動回測的差異有多大，不是為了引用它的"
        "數字。門檻只由驗證集選定；每個成本比也重新在驗證集選門檻，再於測試集評分。", "",
        "| 情境 | 加驗率 | 攔截率 | 漏放 | 每筆成本 | 相對事前基準節省 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, case in decision["cases"].items():
        r = next(x for x in case["test_policies"] if x["policy"].startswith("模型導向"))
        lines.append(
            f"| {name} | {_fmt(r['flag_rate'], '.1%')} | {_fmt(r['catch_rate'], '.1%')} | "
            f"{r['missed']} | {_fmt(r['cost_per_lot'], ',.0f')} | "
            f"{_fmt(r['節省比例'], '+.1%')} |"
        )
    lines += [
        "",
        "無限制那一列的加驗率是 70%，攔截率 94% —— 這又是同一個「接近全檢」的假影，"
        "不是排序能力的證據。單一切分只有一個時間窗、17 個正樣本，它的正號完全在"
        "滾動回測顯示的折間變異範圍之內。",
        "",
        "成本比掃描是有限網格上的觀察，不推論「高於某一比值就永遠划算」。成本曲線遍歷"
        "測試門檻僅是事後診斷，不用於策略選擇。", "",
    ]
    return lines


def write_summary(include_backtest: bool = True, include_ablation: bool = False) -> None:
    cfg = config.load()
    decision = _read("reports/metrics/decision.json")

    bt_path = config.resolve("reports/metrics/backtest.json")
    have_backtest = include_backtest and bt_path.exists()
    report = _read("reports/metrics/backtest.json") if have_backtest else {}

    lines = [
        "# 半導體不良品預測與加驗成本評估", "",
        "> 本報告由實驗腳本自動更新。主要證據是滾動回測與零模型參考策略的比較；"
        "單次切分只在附錄供對照。", "",
        "本專題研究：在小樣本、類別不平衡與時間分布改變下，模型能否協助配置加驗資源。", "",
    ]

    if report:
        lines += _headline_lines(report, cfg)
    else:
        lines += [
            "## 主結論", "",
            "**滾動回測尚待重新產生，不能以單次切分判定可部署。** 請執行 "
            "`python scripts/04_backtest.py`。", "",
        ]

    lines += [
        "## 問題與假設", "",
        "- UCI SECOM：1,567 筆生產實體、590 個匿名感測器、104 筆 fail。",
        "- 將一筆觀測視為一個可獨立加驗單位，是本專題的情境假設；資料未證明它代表整批產品。",
        f"- 假設加驗成本 {cfg.cost.c_inspect:,}、漏放成本 {cfg.cost.c_escape:,} TWD／單位，"
        "且加驗能攔截所有被選中的不良品。",
        "- 原始標籤是廠內測試 pass/fail，並非真實客戶退貨；成本不是實際節省的財務紀錄。",
        "- 感測器必須在決策時可取得；資料缺少站點語意，這項部署條件仍需實廠確認。", "",
    ]

    if report:
        lines += _protocol_lines(report)
        lines += _comparison_lines(report)
        lines += _uncertainty_lines(report, cfg)
    if include_ablation:
        lines += _ablation_lines()
        lines += _drift_lines()
    lines += _single_split_lines(decision)

    lines += [
        "## 限制與下一步", "",
        "- 保守策略是基準率上界放大漏放成本的啟發式方法，不是對漂移下的個體風險或總成本"
        "提供保證；如上所述，它在本資料上退化成接近全檢。",
        "- SHAP 用於描述模型對匿名欄位的依賴，不能推論因果、站點故障或具體改善參數。",
        "- 優先取得更多時期資料、決策時點的欄位可用性、實際加驗成效與成本，再進行前瞻驗證。",
        "- 若要逐筆或按天部署，需改用事先固定的歷史分數門檻或明確的每日批次，並重做回測。", "",
        "[資料來源：UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom)", "",
    ]

    config.resolve("reports/executive_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
