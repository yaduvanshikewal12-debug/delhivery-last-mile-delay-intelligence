# Delhivery Last-Mile Delay Intelligence Pipeline

A reproducible, offline capstone project using the synthetic shipment data supplied with the assignment. The project builds one connected pipeline: data cleaning and features (Part A), delivery-hours regression (Part B), delay classification and alert threshold selection (Part C), and hub segmentation plus batch scoring (Part D).

> **Data note:** All shipment records are synthetic. This project does not use Delhivery production records.

## How to run

Run every command from the repository root. The versions below are pinned in `requirements.txt`.

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_data.py
python part_a_clean_eda.py
python part_b_regression.py
python part_c_classification.py
python part_d_segments.py
python package_model.py
python predict_delay.py
```

`generate_data.py` regenerates the three committed data files. `part_a_clean_eda.py` creates `data/modelling_table.csv`. Parts B and C read that table and write their result CSVs. Part D writes the hub segments. **`package_model.py` regenerates `models/delay_model.pkl`** and its metadata files. `predict_delay.py` writes `outputs/new_shipment_actions.csv`.

The committed pickle can only be loaded reliably with the scikit-learn version pinned in `requirements.txt`, because pickle model compatibility is version-sensitive. Run `package_model.py` to rebuild it in your environment. If a grader has a different installed scikit-learn version, it should run `package_model.py` rather than rely on the committed pickle.

## Data findings

The data foundation retains **1,200 unique historical shipments** after dropping 24 exact duplicates. The overall delayed count is **278**, for a delay rate of **0.2317**. No null values remain in the 17 model features and the target after cleaning and imputation (`outputs/part_a_checks.csv`).

- **Service type:** Express has the highest delay rate at **0.4897** (119 delayed out of 243). Standard is 0.2056 and Economy is 0.0857 (`outputs/part_a_delay_by_group.csv`).
- **Hub tier:** delay rate rises from **0.0400** for Tier 1, to **0.2150** for Tier 2, to **0.4400** for Tier 3 (`outputs/part_a_delay_by_group.csv`).
- **Distance:** delay rate rises from **0.0219** in the 0–200 km band, to **0.1368** at 201–400 km, **0.4364** at 401–600 km and **0.8523** above 600 km (`outputs/part_a_delay_by_group.csv`).
- **Three strongest correlations with delivery hours by absolute value:** `promised_hours` **+0.8372**, `distance_km` **+0.4013**, and `n_stops` **+0.3247** (`outputs/part_a_correlations.csv`).
- **Declared value transformation:** the median-filled declared-value column has mean INR **1,533.0542** and median INR **1,121.5000**, while its natural-log column has mean **7.0277** and median **7.0224**. The log compresses the right-skewed monetary scale so unusually high declared values have less leverage (`outputs/part_a_checks.csv`).

## Model results (regression)

The mean baseline has validation R² **−0.0212**. A negative R² means that, on the validation set, predicting the training-set mean is worse than predicting the validation-set mean for every parcel.

The validation-RMSE winner is **Linear Regression**, with RMSE **4.8912 hours** and R² **0.9313**. Gradient Boosting is very close (RMSE 4.9165; R² 0.9306), followed by Random Forest (RMSE 5.5095) and the selected Decision Tree (RMSE 6.2796). Linear Regression's single final test evaluation gives RMSE **4.5235 hours** and R² **0.9493** (`outputs/part_b_metrics.csv`).

The depth diagnostic shows the bias/variance pattern: depth 1 has high training and validation errors (11.6334 and 12.1324), indicating high bias. The unlimited tree reaches zero training RMSE but has validation RMSE 7.8368, a gap of 7.8368, indicating high variance. Depth 6 has the lowest validation RMSE, **6.2796** (`outputs/part_b_tree_depth.csv`).

The learning-rate experiment performs best at **0.1** (validation R² 0.9306). At 0.01, learning is too slow for 100 boosting stages (R² 0.7127); at 1.0, the larger steps overshoot useful improvements (R² 0.8440) (`outputs/part_b_gb_learning_rate.csv`).

Five-fold cross-validation slightly favours Gradient Boosting (mean R² 0.9353) over Linear Regression (0.9261), while the fixed validation split favours Linear Regression. I retain Linear Regression because the brief defines the final candidate by the lowest validation RMSE; cross-validation is a stability check, not a replacement for that selection rule. Both cross-validation standard deviations are below 0.01 (`outputs/part_b_cv.csv`).

## Model results (classification)

The blind baseline predicts every parcel as on time. Its validation accuracy is **0.7667**, but recall is **0.0000**: it catches none of the delayed parcels, so the accuracy is not operationally useful. The class distribution is imbalanced, with 23.17% delayed shipments overall.

The five bookkeeping/noise columns together contribute **0.2245** of the default forest's feature importance. All five rank outside the top seven by univariate F-score. The seven highest-F-score features are, in order: `distance_km`, `hub_tier`, `hub_daily_capacity`, `n_stops`, `promised_hours`, `rain_flag`, and `sale_event_flag` (`outputs/part_c_feature_scores.csv`).

The SelectKBest comparison shows the best validation F1 at **k = 7** (0.8333), compared with 0.6988 at k=5, 0.7857 at k=10, 0.8049 at k=12 and 0.7848 at k=17 (`outputs/part_c_selectk.csv`).

Tuning did not beat the untuned forest on the validation F1 metric: grid search reached 0.7750, randomized search 0.7848, and the default forest on the same seven selected features reached **0.8333**. The searches optimise cross-validation accuracy by default, whereas the final candidate is selected using validation F1. With only 180 validation rows, a small number of changed predictions can also move F1 noticeably; the result is not proof that tuning is never useful (`outputs/part_c_tuning.csv`).

The selected alert threshold is **0.5**, the highest tested threshold whose validation recall remains at least 0.80. At 0.5, validation precision is 0.7955 and recall is 0.8333. On the one-time test evaluation, precision is **0.6889**, recall is **0.7561**, and F1 is **0.7209**. Test recall is lower than validation recall by about 0.0772; the test set contains only 41 delayed parcels, so each missed parcel has a visible effect (`outputs/part_c_thresholds.csv`, `outputs/part_c_final_test.csv`).

## Operations recommendation

The **Escalate** playbook segment contains 9 hubs and has a mean hub-level delay rate of **0.4600**. The **Watch** segment contains 8 hubs at 0.1550, and **Standard** contains 7 hubs at 0.0257 (`outputs/segment_profile.csv`).

The tier labels do not match the observed segment for four hubs:

- HUB07, Kolkata: Tier 1 but Watch.
- HUB13, Nagpur: Tier 2 but Escalate.
- HUB15, Kochi: Tier 2 but Escalate.
- HUB22, Dehradun: Tier 3 but Watch.

Use the tier as a starting classification, not as a substitute for observed operating performance (`outputs/tier_vs_segment_mismatch.csv`).

The alert threshold is **0.5**. On the test set, precision 0.6889 means about **6.9 of every 10 flagged parcels** are actually delayed; recall 0.7561 means about **7.6 of every 10 delayed parcels** are caught. In the 48 new parcels, **8 are flagged**, and **6 of those 8** are routed to a hub-manager escalation because their hub is in the Escalate segment (`outputs/new_shipment_actions.csv`).

I keep **K = 3** because the business requires three playbooks and the solution gives meaningful groups of 7, 8 and 9 hubs. Although the across-to-within separation ratio rises from 5.3928 at K=3 to 6.3047 and 6.8866 at K=4 and K=5, those larger-K solutions create clusters of only 4 hubs and 2 hubs. Those groups are too small to support independent operational playbooks (`outputs/part_d_separation.csv`).

## Production plan

| Readiness characteristic | Concrete implementation in this repository |
|---|---|
| Accuracy | Validation and final-test metrics are written to the regression and classification CSVs; the test rows are held back from model selection. |
| Reliability | The packaged model is reloaded and checked; maximum probability difference is 0.0000. |
| Reproducibility | Random seeds are pinned, package versions are pinned, generated data is deterministic, and scoring is repeatable. |
| Maintainability | Shared cleaning and feature-building logic lives in `src/shipment_prep.py`; model and output steps are separated into scripts. |
| Consistency | Training and batch inference call the same `clean_shipments` and `build_features` functions and use the saved feature order and imputation medians. |
| Monitorability | The action list records probability, flag, segment and action; weekly delay and alert metrics can be monitored against explicit triggers. |

Suggested monitoring triggers:

1. Review hub capacity and lane conditions if the weekly delayed share of delivered parcels exceeds **30%** for two consecutive weeks (the synthetic historical baseline is 23.17%).
2. Review threshold calibration and upstream data if more than **30%** of dispatched parcels are flagged in a week.
3. Reassess the model if recall on matured parcels falls below **0.80** over a rolling four-week window.

Retrain monthly using the latest 6–12 months of matured shipment outcomes, after checking label completeness and drift. For the **15–28 September sale window**, compare sale-period shipments with the same kind of sale event in another period, not with ordinary weeks; the `sale_event_flag` captures this demand context. Potential drift includes shifts in distance, stop counts, service mix, hub operating tier, weather exposure, sale-period volume, or a sustained change in observed delay rates.

## Limitations

- The dataset is synthetic and does not represent the full range of real logistics disruptions, customer behaviour or hub processes.
- The test set contains only 41 delayed parcels; precision, recall and F1 therefore have noticeable sampling uncertainty.
- The alert threshold was selected from one validation split and a coarse grid of nine threshold values; it should be revalidated on later matured data before operational use.
- The model uses the fields in the generated dispatch export; it cannot react to unobserved disruptions or changes that are not represented in those features.
