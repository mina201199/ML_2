"""互動儀表板：探索不同成本與產能假設下的加驗策略。

這支 app 的定位需要說清楚，否則它會跟報告的結論打對台。

**它不是結論的來源。** 側邊欄操作的是單次時間切分（314 筆、17 筆 fail），也就是
報告裡被降級為附錄的那個情境。單一時間窗的正號完全落在滾動回測顯示的折間變異
範圍之內，所以本頁最上方固定顯示滾動回測的判定，而不是讓側邊欄的試算結果
充當頭條。

它的價值在於回答「如果成本假設不同會怎樣」—— 那是一個真正需要互動的問題，
靜態報告答不了。

資料來源刻意只用兩個小檔（都在版控裡）：
    reports/metrics/scored_holdout.json   驗證／測試窗的標籤與校準後機率
    reports/metrics/backtest.json         滾動回測的結論與不確定性

不讀 models/fitted.pkl —— 那個檔 7.5 MB、含完整資料，而 data/ 與 models/ 都不進版控。
這支 app 從頭到尾只需要 y 與 p 兩個陣列（下游全是 decision/evaluate 的純函式），
所以任何人 clone 或任何託管平台都能直接跑起來，不必先訓練。
"""

import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from secom import config, decision, evaluate

ACTION_LABELS = {
    'inspect_none': '永遠不檢',
    'inspect_all': '永遠全檢',
    'random_at_capacity': '按配額隨機加驗',
}

st.set_page_config(page_title='加驗成本評估 · SECOM', page_icon='🔬', layout='wide')


@st.cache_data
def load_holdout():
    """讀評分後的驗證／測試窗。由 scripts/02_train.py 產生。"""
    path = config.resolve('reports/metrics/scored_holdout.json')
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding='utf-8'))
    return dict(
        y_val=np.asarray(d['val']['y']), p_val=np.asarray(d['val']['p']),
        y_test=np.asarray(d['test']['y']), p_test=np.asarray(d['test']['p']),
        windows=d['windows'],
    )


@st.cache_data
def load_shap():
    path = config.resolve('reports/metrics/shap_ranking.csv')
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def load_backtest():
    path = config.resolve('reports/metrics/backtest.json')
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


cfg = config.load()
holdout = load_holdout()
shap_ranking = load_shap()
report = load_backtest()

st.title('半導體不良品預測與加驗成本評估')

# ── 主結論固定置頂，不受側邊欄影響 ──
# 這一段刻意寫在最前面：儀表板的第一印象必須跟 reports/executive_summary.md
# 的頭條一致，否則就會出現「報告說模型輸、儀表板說 +12.7%」的自相矛盾。
if report and 'dominance' in report:
    dom = report['dominance']
    n_configs = sum(d['n_configurations'] for d in dom.values())
    n_beat = sum(d['n_beating_model_free'] for d in dom.values())
    if n_beat == 0:
        st.error(
            f'**滾動回測結論：{n_beat} / {n_configs} 個模型導向配置贏過零模型策略。**  \n'
            + '　'.join(
                f'{regime}：'
                f'{ACTION_LABELS.get(d["best_model_free_action"], d["best_model_free_action"])}'
                f' {d["best_model_free_cost"]:,.0f}／筆'
                f' vs 最佳模型 {d["cheapest_model_driven_cost"]:,.0f}／筆'
                for regime, d in dom.items()
            )
            + '  \n本頁以下是**單次切分的情境試算**，用來看成本假設如何改變策略行為，'
              '不是結論。完整證據見 `reports/executive_summary.md`。'
        )
    else:
        st.warning(
            f'滾動回測：{n_beat} / {n_configs} 個配置贏過零模型策略。'
            '以下的單次切分試算仍不能單獨當作結論。'
        )
else:
    st.info('尚未產生滾動回測。請執行 `python scripts/04_backtest.py` 取得主結論。')

if holdout is None:
    st.error('缺少 reports/metrics/scored_holdout.json，請先執行 '
             '`python scripts/02_train.py`。')
    st.stop()

y_val, pv = holdout['y_val'], holdout['p_val']
y_test, pt = holdout['y_test'], holdout['p_test']
window = holdout['windows']['test']

with st.sidebar:
    st.header('情境假設')
    ci = st.number_input('加驗成本（TWD／單位）', 100, 50000,
                         int(cfg.cost.c_inspect), step=100)
    ce = st.number_input('漏放成本（TWD／單位）', 1000, 2000000,
                         int(cfg.cost.c_escape), step=1000)
    st.metric('成本比', f'{ce / ci:.1f} : 1')

    # 損益兩平盛行率：加驗開始划算的那條線。跟 secom/drift.py 的告警錨點同一個量。
    breakeven = ci / ce
    st.caption(f'損益兩平盛行率 p\\* = {breakeven:.2%}。測試窗實際盛行率 '
               f'{y_test.mean():.2%} —— 盛行率在 p\\* 之上時，「全檢／按配額隨機加驗」'
               '本身就已經優於不檢，模型必須再贏過它才有價值。')

    use_cap = st.checkbox('限制加驗產能', value=False)
    cap_percent = st.slider('最多加驗比例（%）', 2, 100,
                            int(cfg.cost.inspect_capacity_frac * 100),
                            disabled=not use_cap)
    cap = cap_percent / 100 if use_cap else None
    kind = st.radio('策略', ['絕對門檻', '分位數', '保守分位數'], index=1)
    st.caption('門檻與基準在驗證窗選定。假設被加驗的不良品全部可攔截，成本為示意值。')

st.subheader('單次切分情境試算（報告附錄的那一個情境）')
st.caption(f"測試窗 {window['from'][:10]} – {window['to'][:10]}："
           f"{window['n']} 筆、{window['n_fail']} 筆 fail。")

if kind == '絕對門檻':
    selected = decision.optimal_threshold(y_val, pv, ci, ce, cap)
    t = selected['threshold']
else:
    fn = decision.robust_flag_rate if kind == '保守分位數' else decision.optimal_flag_rate
    selected = fn(y_val, pv, ci, ce, cap)
    t = decision.threshold_for_flag_rate(pt, selected['target_flag_rate'])
if cap is not None:
    t = max(t, decision.threshold_for_flag_rate(pt, cap))

policies = decision.policy_table(y_test, pt, t, ci, ce,
                                 y_select=y_val, capacity_frac=cap)
row = policies[policies.index.str.startswith('模型導向')].iloc[0]

a, b, c = st.columns(3)
# delta_color='off' 是刻意的：這個百分比的正負不是結論，只是這一個時間窗的觀察。
# 綠色向上箭頭會讓它讀起來像一個已經確立的勝利。
a.metric('測試集成本／單位', f'{row.cost_per_lot:,.0f}',
         f"{row['節省比例']:+.1%} 相對事前基準（僅此窗）", delta_color='off')
b.metric('實際加驗率', f'{row.flag_rate:.1%}')
c.metric('不良品攔截率', f'{row.catch_rate:.1%}',
         f'漏放 {int(row.missed)} 筆', delta_color='off')

st.caption(
    f"事前選定的基準：{policies.attrs['baseline_name']}。"
    '分位數假設整段測試窗可一起排序；同分邊界一起排除，可能未用滿產能。'
    f"此窗只有 {window['n']} 筆、{window['n_fail']} 筆 fail —— "
    '這個百分比的解析度大約是「一筆不良品」的量級。'
)
if row['節省比例'] > 0 and row.flag_rate > 0.5:
    st.warning(
        f'注意：這個「節省」是在加驗 {row.flag_rate:.0%} 的批次之下取得的。'
        '高加驗率下的成本改善主要來自加驗比例本身，不是模型排序 —— '
        '報告的置換檢定（p = 0.52）顯示排序與同配額隨機抽驗無法區分。'
    )
elif row['節省比例'] <= 0:
    st.info('此策略在這個測試窗未優於事先選定的可行基準。')

left, right = st.columns([3, 2])
with left:
    st.markdown('**策略成本比較**')
    show = policies[['flag_rate', 'catch_rate', 'missed', 'cost_per_lot',
                     'feasible']].copy()
    show.columns = ['加驗率', '攔截率', '漏放（隨機策略為期望值）', '成本／單位', '符合產能']
    st.dataframe(
        show.style.format({'加驗率': '{:.1%}', '攔截率': '{:.1%}',
                           '漏放（隨機策略為期望值）': '{:.1f}',
                           '成本／單位': '{:,.0f}'}),
        width='stretch',
    )
    st.caption('全檢若超過產能上限，只作參考，不作可行基準。')
with right:
    st.markdown('**測試集模型品質**')
    sc = evaluate.scores(y_test, pt)
    st.dataframe(
        pd.DataFrame({
            '指標': ['PR-AUC', 'ROC-AUC', 'Brier', 'Recall@20%', 'lift@20%'],
            '值': [sc['pr_auc'], sc['roc_auc'], sc['brier'],
                   sc['recall@20%'], sc['lift@20%']],
        }).style.format({'值': '{:.4f}'}),
        hide_index=True,
    )
    st.caption('lift@20% 是關鍵讀數：低於 1 就代表沒有贏過同配額隨機抽驗。'
               'Brier 同時反映校準與辨識能力，不能單獨證明校準良好。')

st.subheader('成本比敏感度：每個比值都在驗證窗選門檻')
sens = decision.sensitivity(y_test, pt, y_select=y_val, p_select=pv,
                            capacity_frac=cap).reset_index()
long = sens.melt(id_vars='cost_ratio', value_vars=['cost_模型', 'cost_基準'],
                 var_name='策略', value_name='成本')
st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
    x=alt.X('cost_ratio:Q', scale=alt.Scale(type='log'), title='漏放成本 ÷ 加驗成本'),
    y=alt.Y('成本:Q', title='總成本（加驗成本為單位）'),
    color='策略:N').properties(height=280), width='stretch')
st.caption('敏感度圖固定採用驗證窗選出的絕對門檻策略。'
           '有限網格上的優勢不代表超過某一比值就永遠划算。')

st.subheader('滾動回測：主證據')
if report:
    st.caption('以下是 config.yaml 所設定成本的回測，不隨側邊欄試算變動。')

    reference = report.get('reference_policies', {})
    if reference:
        st.markdown('**零模型參考策略** —— 不看分數、不選門檻、固定不變，'
                    '是任何模型導向策略必須跨過的門檻。')
        ref_rows = [
            {'產能情境': regime,
             '零模型策略': ACTION_LABELS.get(action, action),
             '可行': '是' if action in ref['feasible'] else '否',
             '成本／筆': cost,
             '最便宜': '←' if action == ref['best_action'] else ''}
            for regime, ref in reference.items()
            for action, cost in ref['costs'].items()
        ]
        st.dataframe(
            pd.DataFrame(ref_rows).style.format({'成本／筆': '{:,.0f}'}),
            hide_index=True, width='stretch',
        )

    st.markdown('**全部模型導向配置**')
    table = pd.DataFrame([
        {'實驗': key,
         '成本／筆': r['summary']['cost_per_lot'],
         '事前基準／筆': r['summary']['baseline_cost_per_lot'],
         'lift@20% 中位數': r['summary'].get('lift_at_20_median'),
         'ROC-AUC 中位數': r['summary']['roc_auc_median'],
         '總漏放': r['summary']['total_missed']}
        for key, r in report['results'].items()
    ])
    st.dataframe(
        table.style.format({'成本／筆': '{:,.0f}', '事前基準／筆': '{:,.0f}',
                            'lift@20% 中位數': '{:.2f}', 'ROC-AUC 中位數': '{:.3f}'}),
        hide_index=True, width='stretch',
    )
    st.caption('lift@20% 中位數低於 1、或成本高於上表最便宜的零模型策略，'
               '就代表該配置沒有跨過門檻。')

    uncertainty = report.get('uncertainty', {})
    if uncertainty:
        with st.expander('統計不確定性：這些數字看得見多大的訊號'):
            for name, u in uncertainty.items():
                lo, hi = u['bootstrap']['relative_saving_ci']
                st.markdown(
                    f'`{name}`  \n'
                    f'- 相對節省 95% bootstrap 區間 **{lo:+.1%} ~ {hi:+.1%}**'
                    f'（模型較便宜的重抽比例 '
                    f"{u['bootstrap']['share_of_resamples_model_cheaper']:.1%}）  \n"
                    f'- 勝過同配額隨機的置換檢定 '
                    f"**p = {u['permutation_vs_random']['p_value_one_sided']:.3f}**  \n"
                    f"- 此樣本量可偵測的最小 ROC-AUC "
                    f"**{u['power']['min_detectable_auc']:.3f}**"
                    f"（{u['power']['n_pos']} 個正樣本）"
                )
else:
    st.info('請執行 python scripts/04_backtest.py 產生跨時間窗比較。')

with st.expander('SHAP 描述性特徵重要度'):
    if shap_ranking is None:
        st.info('缺少 reports/metrics/shap_ranking.csv。')
        st.stop()
    top = shap_ranking.head(15)
    st.altair_chart(
        alt.Chart(top).mark_bar().encode(
            x='mean_abs_shap:Q', y=alt.Y('feature:N', sort='-x')),
        width='stretch')
    st.caption('SHAP 描述模型對匿名欄位的依賴，不代表因果或可操作的製程參數；'
               '此排名不用於消融特徵選擇。')
