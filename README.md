# Semiconductor failure prediction and inspection cost evaluation

[![CI](https://github.com/mina201199/secom-cost-sensitive-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/mina201199/secom-cost-sensitive-eval/actions/workflows/ci.yml)

**▶ [Interactive cost explorer](https://mina201199.github.io/secom-cost-sensitive-eval/)** — no install; change the cost and capacity assumptions and watch the policy respond

[繁體中文](README.zh-TW.md) · [Generated experiment report](reports/executive_summary.md)

This project asks whether machine learning can allocate inspection resources when failures are rare, sensor measurements are numerous, and the data distribution changes over time.

## The finding, first

**It cannot here — and the value of this project is that it quantifies why.**

Eighteen model × policy × capacity configurations were compared on identical future windows with walk-forward backtesting, against three fixed policies that use no model at all:

| Capacity regime | Cheapest model-free policy | Cost/unit | Cheapest model-driven config | Cost/unit | Configs beating model-free |
| --- | --- | ---: | --- | ---: | :---: |
| Unconstrained | Always inspect everything | **2,000** | majority/robust\* | 2,000 | **0 / 9** |
| At most 20% inspected | Random at the same quota | **2,644** | lgbm/threshold | 2,677 | **0 / 9** |

\* Read that cell carefully: `majority` is a zero-information DummyClassifier whose conservative policy degenerates to inspecting 100%, so that 2,000 *is* full inspection wearing a model's name — it ties the model-free reference rather than beating it (the verdict uses a strict inequality). Every configuration that actually uses ranking is more expensive: the cheapest LightGBM configuration in the unconstrained regime is `lgbm/robust` at 2,042.

Four quantitative results support the conclusion:

- **The bar can be computed in advance.** With R = escape cost ÷ inspection cost and p = prevalence, whenever p·R ≥ 1 (here 1.40) the condition for a model to add value reduces to **lift > 1** — at the same inspection quota, the model's selection must be denser in failures than random sampling. The measured median lift@20% across four windows is **0.86**.
- **Permutation test p = 0.52.** The flagged set is indistinguishable in cost from a random set at the same quota.
- **The 95% bootstrap interval for the cost difference straddles zero** (relative saving −23.5% to +15.9%), and the model is cheaper in only 43.6% of resamples.
- **This design cannot see a small signal in the first place.** With 33 positives the minimum detectable ROC-AUC is **0.644**; the measured value sits near 0.5, so it neither establishes nor refutes usefulness — it establishes that the sample size cannot answer the question. Detecting AUC 0.60 needs roughly 69 failures (2.1×).

![Feasibility and required lift](reports/figures/feasibility.png)

Drift analysis supplies the mechanism. Against the first fold's training window, PSI is computable for 452 of the 590 sensors — the remaining 138 are entirely missing (16) or constant (122) in that window, so no quantile bins exist and their PSI is undefined rather than zero. Among those 452, the share drifting past PSI > 0.25 rises monotonically across the four evaluation windows: **61% → 64% → 70% → 75%**. The model is not failing to learn; it is being asked to extrapolate onto distributions it never saw.

![Rolling evaluation](reports/figures/backtest.png)

The generated report is the single source for current numbers, assumptions and limitations. The previous +12.4% and 8.7:1 claims are superseded.

## Data and assumptions

[UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom) contains 1,567 observations and 104 failures. The raw sensor file has 590 columns; the repository validates the parsed file rather than relying on the differing column count in UCI's page description.

Labels describe in-house pass/fail tests, not observed customer escapes. Treating each observation as an independently inspectable unit, perfect interception of inspected failures, and costs of TWD 2,000 per inspection and TWD 60,000 per escape are scenario assumptions. Sensor availability at the intended decision time requires confirmation.

**The cost assumption itself sets the difficulty.** At 2,000 : 60,000 we get p·R = 1.40 > 1, which places the scenario in the region where a model only has to beat random sampling — the most permissive region available to it. It did not.

## Evaluation protocol

- Chronological outer windows. Preprocessing and LightGBM bin construction are fitted separately inside each inner training fold.
- Final tree count is selected by the mean inner validation AUC curve. Earlier OOF calibration predictions use a tree count fixed in advance, avoiding retrospective hyperparameter selection for those predictions.
- Calibration and decision selection share a calibration window; scores on that window are not independent results. Performance is measured in the next window.
- Each cost ratio selects a threshold on validation and measures its cost on future observations. There is no test-optimized deployment break-even claim.
- Three models, three policies, and two capacity regimes share identical future windows. Feasible baselines are selected using calibration labels, not evaluation labels.
- **That prospectively selected baseline can pick wrong**, so the headline conclusion is measured against fixed model-free policies instead — they never get the chance to pick wrong, which makes them harder to argue with.
- Retain zero-failure windows for cost evaluation, mark undefined metrics as null, and include remaining observations in the final fold.
- The SQL ablation creates historical deviations for all sensors; it does not select sensors using test-set SHAP rankings.

Quantile decisions assume that the entire decision batch is available for ranking. This is not an online, one-observation-at-a-time simulation. Boundary ties are excluded together, so capacity may be underused. Historical SQL features use the preceding 20 rows; tied timestamps follow input sequence, assumed available operationally.

## Disclosed because it will be asked

- **The "conservative" policy is not a robustification; it is a switch to near-full inspection.** The Clopper–Pearson upper bound inflates escape cost by 1.42–1.95× (per fold), pushing p·λ·R past 1, so the optimal inspection fraction saturates (the four folds select 86% / 28% / 98% / 95%). Its cost improvement comes from the inspection rate, not from ranking.
- **Calibration does not affect all three policies.** Platt scaling is monotone and the quantile policies use only the ordering, so calibration leaves their decisions unchanged; only the absolute-threshold policy uses the probability scale. `test_calibration_does_not_change_rank_based_decisions` verifies this directly.
- **The `min_estimators: 10` floor actually binds.** It is not dormant insurance. Hitting it means the mean inner-CV AUC peaked in under ten trees and then declined.
- **AUC is deliberately excluded from the retraining trigger.** At 33 positives its fluctuation exceeds any real change, so using it as a trigger would only manufacture false alarms.

## Experiments

| Component | Comparison |
| --- | --- |
| Models | Dummy prior, L2 logistic regression, LightGBM |
| Policies | Absolute threshold, quantile, conservative base-rate adjustment |
| Capacity | Unconstrained, at most 20% inspected |
| Model-free references | Inspect nothing, inspect everything, random at an integer quota |
| Metrics | Average precision, ROC-AUC, Brier, lift@20%, interception, cost, variation across windows |
| Uncertainty | Minimum detectable effect, bootstrap interval for cost difference, exact permutation test against random selection |
| Monitoring | Prevalence-regime call anchored on the break-even rate, feature PSI, retraining trigger |
| Ablation | 590 raw sensors vs 1,180 additional historical deviations |

SHAP describes model dependence on anonymous columns, not causality or actionable manufacturing settings.

## Reproduce

```bash
pip install -e ".[dev]"
python scripts/01_build_data.py
python scripts/02_train.py
python scripts/03_decide.py
python scripts/04_backtest.py
python scripts/05_sql_report.py
python scripts/06_build_pages.py
python -m pytest tests/
ruff check .
streamlit run app/streamlit_app.py
```

Run scripts in order: 03 writes the single-split report (comparison only, now demoted to an appendix), 04 produces the headline conclusion and the uncertainty quantification, and 05 adds the ablation and drift monitoring. The whole pipeline takes about 40 seconds.

The dashboard pins the walk-forward verdict to the top of the page, read straight from `backtest.json`, so it cannot contradict the report. Its sidebar controls drive the single-split appendix scenario; the rolling tables below show saved experiments at configuration-file costs, including the model-free reference policies.

**The dashboard runs standalone — no training step required.** It reads two
small version-controlled files (`reports/metrics/scored_holdout.json`, about
20 KB, plus `backtest.json`) rather than the 7.5 MB `models/fitted.pkl`, and it
never touches `data/`. A fresh clone can run `streamlit run app/streamlit_app.py`
immediately, which is also what makes it deployable to a hosted platform. The
leading `.` in `requirements.txt` exists for that: hosted platforms run only
`pip install -r requirements.txt`, never `pip install -e .`.

`requirements.txt` states compatible lower bounds (intent); `requirements-lock.txt` and `reports/metrics/environment.json` record the actual validated versions (fact). Both are generated by `01_build_data.py` from installed package metadata rather than maintained by hand. Fixed seeds aid reproducibility, but byte-identical outputs across versions and platforms are **not** guaranteed.

### Measured from a clean clone

Cloned from GitHub into a fresh virtualenv and run end to end with exactly the commands above:

| Step | Result |
| --- | ---: |
| Clone size | 2.1 MB |
| `pip install -e ".[dev]"` | 81 s |
| Scripts 01 → 06, including the UCI download | **41 s** |
| `pytest tests/` | 73 passed, 13 s |
| `ruff check .` | clean |
| Dashboard `healthz` | 200 after ~2 s |

That clean environment resolved 9 of 13 direct dependencies to different versions than the recorded one — pandas crossed a major version (2.3.3 → 3.0.6) and scikit-learn went 1.7.2 → 1.9.1 — and `reports/executive_summary.md` still came out **byte-identical** to the committed copy. That is an observation, not a guarantee: a single reproduction does not support a cross-version stability claim, so the sentence above still stands.

## Limitations

Four windows and few positive examples provide limited evidence. The methodology changes were informed by earlier results on these same observations; they need confirmation on a genuinely unseen period. OOF and final estimators also differ in training size and tree count, so calibration transfer remains a limitation.

The power analysis gives concrete targets for what "more data" means: detecting AUC 0.60 needs roughly 69 failures across 1,468 observations; AUC 0.55 needs roughly 275 failures across 5,872. Obtain decision-time feature definitions, real costs, and inspection effectiveness before prospective validation.

**A negative result here is not proof that no model can find signal.** It establishes that at this sample size, this granularity, and this set of cost assumptions, the question cannot be answered.

See [the report](reports/executive_summary.md) for current findings and [the Chinese README](README.zh-TW.md) for the file map.

## License

MIT — see [LICENSE](LICENSE). The UCI SECOM data is not redistributed here; `scripts/01_build_data.py` downloads it from the source.
