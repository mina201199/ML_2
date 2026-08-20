"""互動儀表板：拉成本 slider，看最佳門檻與省下的錢怎麼變。

    streamlit run app/streamlit_app.py

這頁存在的理由：面試官不會讀你的 notebook，但他會拉一次 slider。
讓他自己把公司真實的成本比拉進去，看模型划不划算 —— 那一刻他就在做決策了。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from secom import config as cfg_mod
from secom import decision, evaluate

st.set_page_config(page_title="加驗決策引擎 · SECOM", page_icon="🔬", layout="wide")


@st.cache_resource
def load_bundle():
    path = cfg_mod.resolve("models/fitted.pkl")
    if not path.exists():
        return None
    b = joblib.load(path)
    model, split = b["models"]["lgbm"], b["split"]
    return {
        "split": split,
        "shap": b["shap_ranking"],
        "p_val": model.predict_proba(split.X_val),
        "p_test": model.predict_proba(split.X_test),
    }


cfg = cfg_mod.load()
bundle = load_bundle()

st.title("半導體測試線 · 加驗決策引擎")

if bundle is None:
    st.error("找不到 models/fitted.pkl。請先執行 `python scripts/02_train.py`。")
    st.stop()

split, p_val, p_test = bundle["split"], bundle["p_val"], bundle["p_test"]

st.caption(
    "模型輸出機率之後還有一步：**要不要加驗？** 0.5 是套件的預設值，不是產線的答案。"
    "下面的成本假設由你決定，門檻由成本決定。"
)

# ── 側邊欄：成本假設 ──────────────────────────────────────────────
with st.sidebar:
    st.header("成本假設")
    c_ins = st.number_input(
        "加驗成本（TWD/批）", 100, 50_000, int(cfg.cost.c_inspect), step=100,
        help="攔停、複測、重工的成本",
    )
    c_esc = st.number_input(
        "流出成本（TWD/批）", 1_000, 2_000_000, int(cfg.cost.c_escape), step=1_000,
        help="一批壞品到客戶端：RMA、報廢、賠償、商譽",
    )
    ratio = c_esc / max(c_ins, 1)
    st.metric("成本比", f"{ratio:.1f} : 1")

    use_cap = st.checkbox("加上產能上限", value=False,
                          help="產線一天只驗得動這麼多比例的批次")
    cap = st.slider("最多可加驗比例", 0.02, 1.0,
                    float(cfg.cost.inspect_capacity_frac), 0.01,
                    format="%.0f%%", disabled=not use_cap)
    cap_used = cap if use_cap else None

    policy_kind = st.radio(
        "決策參數化方式", ["絕對機率門檻", "分位數（標記前 k%）"], index=1,
        help="分位數是相對規則，基準率漂移時比絕對門檻耐用",
    )

    st.divider()
    st.caption(
        "門檻一律在 **val** 上選，只在 **test** 上評估。"
        "在 test 上調門檻等於交卷後改題目。"
    )

# ── 在 val 上選門檻，套到 test ───────────────────────────────────
if policy_kind == "絕對機率門檻":
    chosen = decision.optimal_threshold(split.y_val, p_val, c_ins, c_esc, cap_used)
    t_test = chosen["threshold"]
    label = f"絕對門檻 t*={t_test:.4f}"
else:
    chosen = decision.optimal_flag_rate(split.y_val, p_val, c_ins, c_esc, cap_used)
    t_test = decision.threshold_for_flag_rate(p_test, chosen["target_flag_rate"])
    label = f"標記前 {chosen['target_flag_rate']:.0%}"

policies = decision.policy_table(split.y_test, p_test, t_test, c_ins, c_esc)
model_row = policies[policies.index.str.startswith("模型導向")].iloc[0]
best_base = policies.iloc[:2]["cost_per_lot"].min()
be = decision.breakeven_ratio(split.y_test, p_test, capacity_frac=cap_used)

# ── 頂部指標 ─────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("每批期望成本", f"{model_row['cost_per_lot']:,.0f}",
          f"{model_row['節省比例']:+.1%} vs 最佳基準")
k2.metric("加驗比例", f"{model_row['flag_rate']:.1%}",
          help="被標記加驗的批次佔比")
k3.metric("壞品攔截率", f"{model_row['catch_rate']:.1%}",
          f"漏放 {int(model_row['missed'])} 批", delta_color="off")
k4.metric("損益兩平成本比",
          f"{be:.1f} : 1" if be else "無",
          "超過此比值模型才划算" if be else "此設定下模型不划算",
          delta_color="off")

if model_row["節省比例"] <= 0:
    if use_cap:
        st.warning(
            f"**產能上限 {cap:.0%} 在成本比 {ratio:.1f}:1 下不可行。**"
            "漏放一批的代價太高，只驗這麼少的比例怎麼配門檻都省不了錢 —— "
            "該去談的是產能，不是模型。把上限拉高或關掉，看它從哪裡開始划算。"
        )
    else:
        st.warning(
            f"**在這組假設下模型不該上線。** 成本比 {ratio:.1f}:1 低於損益兩平點"
            f"{f'（{be:.1f}:1）' if be else ''}，直接全檢或直接不檢更省。"
            "能講出模型什麼時候沒用，比多 0.01 AUC 有價值。"
        )

st.divider()

# ── 成本曲線 ─────────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.subheader("成本 vs 決策門檻")
    curve = decision.cost_curve(split.y_test, p_test, c_ins, c_esc)
    base = alt.Chart(curve).encode(
        x=alt.X("threshold:Q", title="決策門檻（校準後 P(fail)）"),
        y=alt.Y("cost_per_lot:Q", title="每批期望成本 (TWD)"),
    )
    layers = [
        base.mark_line(color="#A9542C", strokeWidth=2.2),
        alt.Chart(pd.DataFrame({"y": [policies.iloc[0]["cost_per_lot"]]}))
        .mark_rule(color="#8A9A9D", strokeDash=[5, 4]).encode(y="y:Q"),
        alt.Chart(pd.DataFrame({"y": [policies.iloc[1]["cost_per_lot"]]}))
        .mark_rule(color="#25707F", strokeDash=[2, 3]).encode(y="y:Q"),
        alt.Chart(pd.DataFrame({"x": [t_test]}))
        .mark_rule(color="#A9542C", strokeWidth=1.6).encode(x="x:Q"),
    ]
    st.altair_chart(alt.layer(*layers).properties(height=330), use_container_width=True)
    st.caption(
        "灰虛線 = 不檢；藍點線 = 全檢；直立線 = 目前選中的門檻。"
        f"目前策略：{label}"
    )

with right:
    st.subheader("策略比較")
    show = policies[["flag_rate", "catch_rate", "missed", "cost_per_lot", "節省比例"]].copy()
    show.columns = ["加驗率", "攔截率", "漏放", "每批成本", "vs 基準"]
    st.dataframe(
        show.style.format(
            {"加驗率": "{:.1%}", "攔截率": "{:.1%}", "漏放": "{:.0f}",
             "每批成本": "{:,.0f}", "vs 基準": "{:+.1%}"}
        ),
        use_container_width=True,
    )
    at_t = evaluate.at_threshold(split.y_test, p_test, t_test)
    st.caption(
        f"門檻下混淆矩陣：TP={at_t['tp']}　FP={at_t['fp']}　"
        f"FN={at_t['fn']}　TN={at_t['tn']}　"
        f"precision={at_t['precision']:.3f}　recall={at_t['recall']:.3f}"
    )

st.divider()

# ── 損益兩平與模型品質 ───────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("損益兩平敏感度")
    sens = decision.sensitivity(split.y_test, p_test, capacity_frac=cap_used).reset_index()
    long = sens.melt(
        id_vars="cost_ratio",
        value_vars=["cost_不檢", "cost_全檢", "cost_模型"],
        var_name="策略", value_name="成本",
    )
    st.altair_chart(
        alt.Chart(long).mark_line(strokeWidth=2).encode(
            x=alt.X("cost_ratio:Q", scale=alt.Scale(type="log"), title="成本比（流出 ÷ 加驗）"),
            y=alt.Y("成本:Q", scale=alt.Scale(type="log"), title="總成本（以加驗成本為單位）"),
            color=alt.Color("策略:N", scale=alt.Scale(
                domain=["cost_不檢", "cost_全檢", "cost_模型"],
                range=["#8A9A9D", "#25707F", "#A9542C"])),
        ).properties(height=290),
        use_container_width=True,
    )
    st.caption("三條線交會的地方就是決策改變的地方。")

with c2:
    st.subheader("模型排序能力（test）")
    sc = evaluate.scores(split.y_test, p_test)
    st.dataframe(
        pd.DataFrame({
            "指標": ["PR-AUC", "÷ 盛行率地板", "ROC-AUC", "Brier",
                     "recall@10%", "recall@20%", "lift@20%"],
            "值": [f"{sc['pr_auc']:.3f}", f"{sc['pr_auc_over_floor']:.2f}",
                   f"{sc['roc_auc']:.3f}", f"{sc['brier']:.4f}",
                   f"{sc['recall@10%']:.1%}", f"{sc['recall@20%']:.1%}",
                   f"{sc['lift@20%']:.2f}"],
        }),
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "訊號真實但很弱，這是 SECOM 的真實難度。決策層的價值在於："
        "**即使排序能力有限，只要成本不對稱夠大，模型依然能省錢。**"
    )

with st.expander("關鍵感測器（SHAP 前 15 名）"):
    top = bundle["shap"].head(15)
    st.altair_chart(
        alt.Chart(top).mark_bar(color="#25707F").encode(
            x=alt.X("mean_abs_shap:Q", title="平均 |SHAP|"),
            y=alt.Y("feature:N", sort="-x", title=None),
        ).properties(height=340),
        use_container_width=True,
    )
    st.caption(
        "SECOM 特徵是匿名的，只能給索引。真實廠內這一步的產出會是"
        "「站點 + 參數」清單，直接交給製程工程師。"
    )
