# Yield Escape Prevention — and why my first result was wrong

[繁體中文版本](README.zh-TW.md)

On a semiconductor test line, every lot can be released or sent for extra inspection. Inspection costs money; letting a bad lot reach the customer costs an order of magnitude more. **Which lots should be inspected?**

A conventional train/validation/test split said my model-guided policy cut expected cost per lot by **12.4%**. I then ran a walk-forward backtest across four deployment points. The result:

| Evaluation | Median cost saving | Median ROC-AUC | Verdict |
| --- | --- | --- | --- |
| Single chronological split | **+12.4%** | 0.693 | Looks deployable |
| Walk-forward, 4 origins | **−29.1%** | **0.524** | One lucky window |

**The +12.4% was noise.** Across four honest deployment points the model performs at chance (ROC-AUC 0.524) and the policy loses money. One fold cost 190% *more* than simply inspecting everything.

This repo is the audit that found that, the diagnosis of why it happened, and the fix that at least made the failure survivable.

![Walk-forward backtest](reports/figures/backtest.png)

---

## Why this is the interesting result

Any portfolio can report a number from one split. The harder and more useful skill is knowing when your own number is wrong — before it reaches production and someone acts on it.

Three things this project actually delivers:

1. **An evaluation framework that caught a false positive.** Walk-forward + explicit cost accounting, not a single split and an AUC.
2. **A root-cause diagnosis** of exactly why the naive result inverted (below).
3. **A robust policy** that cut the worst-case loss from −190% to −31% and raised median failure capture from 25% to 100%.

And an honest recommendation: **do not deploy this model.** At a 30:1 cost ratio, near-full inspection is the correct policy on this data. That conclusion is worth more than a fabricated win.

---

## Diagnosis: a decision boundary balanced on 7 noisy samples

Two data facts drive everything.

**1. Yield drifts ~22× across the dataset.** Weekly failure rate runs from 23.1% down to 1.1% over 13 weeks — a classic process ramp.

![Yield drift](reports/figures/drift.png)

This immediately invalidates random splitting: early high-failure lots would land in both train and test, and a model could score well just by recognising "this is an early lot". Most published SECOM results reporting 0.95+ AUC come from exactly this leak, or from resampling/imputing before the split.

**2. At a 30:1 cost ratio, release-vs-inspect hinges on whether the failure rate exceeds 3.33%.** Inspecting costs 1 unit; releasing costs `30 × p`. The two are equal at `p = 1/30 = 3.33%`.

Now look at what each calibration window actually observed:

| Fold | Failures in calibration window | Observed rate | vs 3.33% boundary | Policy chosen | Next window's actual rate | Outcome |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 6 / 235 | 2.55% | just below | inspect 5% | 3.4% | +43.8% |
| 2 | 8 / 235 | 3.40% | just above | inspect 28% | 3.4% | −30.7% |
| 3 | 7 / 235 | 2.98% | just below | **inspect 0%** | **9.7%** (3.2× spike) | **−189.8%** |
| 4 | 20 / 235 | 8.51% | well above | inspect 87% | 2.3% | −27.5% |

Three of four windows sit within half a percentage point of the boundary. **The release/inspect decision was being made by 6–8 noisy positive samples.** Fold 3 is the catastrophe: the window looked calm, the policy chose to inspect nothing, and failures immediately tripled.

This is not a modelling bug. It is a decision made under an unacknowledged confidence interval.

---

## The fix: be pessimistic in proportion to your ignorance

If the true failure rate might be higher than what a small window showed, then the expected number of escapes is also higher. So instead of trusting the point estimate, use a conservative one-sided **Clopper–Pearson upper bound** on the base rate, and inflate the escape cost by `p_upper / p_observed`.

The fewer positive samples you have, the larger that inflation — the policy leans toward inspecting more precisely when it knows least. With enough data the factor approaches 1 and it reduces to the ordinary solution.

Applied to the same four folds:

| Policy | Median saving | Worst fold | Failures caught (median) | Profitable folds |
| --- | --- | --- | --- | --- |
| Absolute probability threshold | −37.7% | −189.8% | 25% | 1 / 4 |
| Quantile ("inspect top k%") | −29.1% | −189.8% | 25% | 1 / 4 |
| **Quantile + conservative base rate** | **−12.6%** | **−30.7%** | **100%** | **2 / 4** |

The robust policy **eliminates the catastrophic fold** and captures every failure in three of the four folds (fold 2 still caught none — the ranker put its failures below any sane cut). It is still net-negative overall — because the underlying model has no signal — but it fails safely instead of ruinously.

That is the correct behaviour: when the ranker is uninformative, a cost-aware robust policy should degenerate toward full inspection. The framework is telling you not to trust the model, which is exactly its job.

---

## I tried to rescue the signal. It didn't work.

Process monitoring rarely cares about a sensor's absolute value — it cares how far the sensor has drifted from its recent normal. So I built deviation features in SQL (DuckDB window functions): for the top 40 sensors by SHAP, the deviation from a trailing 20-lot mean and its standardised version, with the window deliberately excluding the current row to avoid leakage.

| Feature set | Median ROC-AUC | Median saving | Worst fold | Profitable folds |
| --- | --- | --- | --- | --- |
| Original 590 sensors | 0.524 | −12.6% | −30.7% | 2 / 4 |
| + 80 deviation features | 0.543 | −5.1% | **−45.0%** | **1 / 4** |

Median ROC-AUC moved 0.524 → 0.543 — still indistinguishable from chance — and the worst fold got *worse*. Per-fold AUCs went from `[0.842, 0.252, 0.528, 0.519]` to `[0.841, 0.579, 0.506, 0.439]`: the features reshuffled which fold wins rather than adding information.

**Kept in the repo as a negative result.** "I tried this and it didn't work" is worth more than a list of things I might try.

---

## Data

[UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom) — production measurements from a semiconductor fab, 2008-07-19 to 2008-10-17.

| | |
| --- | --- |
| Lots | 1,567 |
| Sensor features | 590 (anonymised, `f000`–`f589`) |
| Failures | 104 (6.64%, imbalance 1:14.1) |
| Missing values | 41,951 cells (4.54%) |
| Zero-variance columns | 116 |
| Span | 89 days |

Two limits worth stating plainly. The dataset is **small** — 1,567 lots and 104 failures, so every fold's estimate carries wide error bars, which is part of the story rather than an excuse. And the features are **anonymised**, so SHAP can only return indices, never "which parameter on which station". The method transfers; the specific findings do not.

---

## Reproduce

```bash
pip install -r requirements.txt
python scripts/01_build_data.py    # download from UCI (no Kaggle token needed)
python scripts/02_train.py         # chronological split, 3 models, SHAP
python scripts/03_decide.py        # cost-optimal threshold on one split
python scripts/04_backtest.py      # walk-forward audit  <- the important one
python scripts/05_sql_report.py    # SQL profiling, drift, feature ablation
```

Every figure and metric in this README is regenerated by these scripts. Reruns are byte-identical, so `git status` stays clean — the results are fully deterministic.

Interactive dashboard (drag the cost sliders and watch the optimal policy move):

```bash
streamlit run app/streamlit_app.py
```

Tests:

```bash
python -m pytest tests/ -v
```

`tests/test_no_leakage.py` covers the three things that would silently invalidate everything: the label column never enters the feature set, splits never overlap in time, and preprocessing parameters are learned from training data only. The first two were written after I hit those bugs.

---

## Three bugs I hit, and what they cost

**`startswith("f")` matched the label column.** The label is named `fail`. Filtering feature columns by prefix silently fed the answer to the model. Caught only because the feature count printed 591 instead of 590. Now matched strictly against `^f\d{3}$`, with the comment left in place.

**Early stopping on the validation set stopped at tree 1.** The validation window held 11 failures; PR-AUC on 11 positives is noise. Worse, that same window was doing triple duty — early stopping, calibration, and threshold selection. Tree count now comes from expanding-window CV *inside* the training data, using the **mean validation curve across folds** rather than the median of per-fold argmaxes (those measured `[2, 1, 88]` — averaging the curves first estimates one quantity with three times the data).

**Platt calibration crushed the dynamic range.** Raw probabilities spanned 0.012–0.57 (47×); after calibrating on 11 positives they spanned 0.028–0.054 (2×). The slope was shrunk almost to zero, destroying the ranking. Calibration now uses out-of-fold training predictions plus the validation window — 55 positives instead of 11.

---

## What I would do with real fab data

- Map sensor indices back to stations and parameters, turning SHAP output into an actionable engineering instruction
- Add maintenance cycles, lot genealogy, and recipe versions — the drift here is almost certainly explained by variables absent from this dataset
- Retrain on a rolling window with drift monitoring, rather than training once
- Replace the illustrative cost parameters with real figures from finance and re-derive the break-even
- Extend the walk-forward audit to more origins once there are enough failures per window to make each fold's estimate meaningful

---

## Repo layout

```
config.yaml              every parameter; no magic numbers in code
secom/
  data.py                download, parse, chronological split
  pipeline.py            preprocessing (structurally train-only)
  models.py              baselines, LightGBM, CV tree count, OOF calibration
  evaluate.py            PR-AUC, recall@k, lift@k, Brier
  decision.py            cost model, optimal threshold, robust policy
  backtest.py            walk-forward evaluation
  sql.py                 DuckDB profiling, drift, window-function features
  plots.py               figures (English labels, report-ready)
scripts/                 01 build · 02 train · 03 decide · 04 backtest · 05 SQL
app/streamlit_app.py     interactive cost dashboard
tests/test_no_leakage.py leakage guards
reports/                 auto-generated summary, figures, metrics
```

## Source

Michael McCann, Adrian Johnston. *SECOM Data Set*. UCI Machine Learning Repository, 2008. <https://archive.ics.uci.edu/dataset/179/secom>
