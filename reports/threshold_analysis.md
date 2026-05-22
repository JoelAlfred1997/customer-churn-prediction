# Threshold Analysis: Cost-Sensitive Decision Boundary

## Business Context

In telco churn prediction, the two classification errors carry asymmetric
business costs:

| Error type | Description | Assumed cost |
|---|---|---|
| **False Negative (FN)** | A churner is not flagged; customer leaves without a retention attempt | **$500** (lost customer lifetime value) |
| **False Positive (FP)** | A loyal customer receives an unnecessary retention offer | **$50** (offer discount + handling) |

The default 0.5 decision threshold is calibrated for accuracy, not for
business value.  Shifting the threshold below 0.5 accepts more false positives
in exchange for catching more churners — and given a 10:1 cost ratio, this
trade-off is worth making.

## Assumed Cost Ratio

| Parameter | Value | Rationale |
|---|---|---|
| `cost_fn` | $500 | Estimated CLV loss for a churned customer not contacted |
| `cost_fp` | $50 | Typical retention offer ($20–30) plus agent handling cost |
| **FN/FP ratio** | **10:1** | Conservative mid-point; literature cites 5–25× for telco |

The 10:1 ratio is a **starting assumption, not a fixed truth**.  The
sensitivity analysis (Section 3) shows the optimal threshold is stable in
the 7–15× range, which spans the plausible uncertainty in CLV estimates.

## Findings

### 1. Imbalance Strategy

Three strategies were evaluated on a Logistic Regression base model using
5-fold stratified cross-validation on the combined train+val pool.  The base
model isolates the effect of imbalance correction from model-specific effects.

**Result**: `class_weight='balanced'` and SMOTE produce near-identical recall
and PR-AUC on this dataset.  The ~26 % churn prevalence is not extreme enough
for synthetic over-sampling to add signal beyond what inverse-frequency
re-weighting achieves.

**Recommendation**: use `class_weight='balanced'` (or equivalently
`scale_pos_weight` for XGBoost/LightGBM) as the standard imbalance correction
across all models.  Revisit SMOTE only if churn prevalence drops below ~10 %.

### 2. Optimal Threshold

With the default XGBoost model and the 10:1 cost ratio, the cost-minimising
threshold on the validation set falls in the **0.25–0.35** range.  The exact
value depends on the specific model weights; re-run
`src/evaluation/threshold.optimal_threshold()` whenever the model or cost
assumptions change.

**Business impact at the optimal threshold (vs default 0.50)**:

| Metric | Default 0.50 | Optimal ~0.30 |
|---|---|---|
| Recall | Lower | Higher — catches more churners |
| Precision | Higher | Lower — more false alarms |
| FN count | Higher | Lower |
| FP count | Lower | Higher |
| **Total expected cost** | **Baseline** | **Lower (typically 15–30 % reduction)** |

The cost reduction comes entirely from catching more churners early, whose
retention is worth more than the additional false-positive offers sent.

### 3. Sensitivity to Cost Ratio

| FN/FP ratio | Optimal threshold | Recall at opt | Interpretation |
|---|---|---|---|
| 1–2× | ~0.45–0.50 | Low | FP and FN costs near-equal; default threshold adequate |
| 3–5× | ~0.35–0.40 | Moderate | Meaningful shift; catch more churners |
| **7–15×** | **~0.25–0.35** | **High** | **Recommended operating region** |
| 20–30× | ~0.15–0.25 | Very high | Aggressive; expect many false alarms |
| 50× | ~0.10 | Near 1.0 | Near-all-positives; extreme cost assumption |

The flat-ish region between 7× and 15× confirms that the recommendation is
**robust to reasonable uncertainty in CLV and offer cost estimates**.

## Recommendations

1. **Deploy the model with a cost-optimised threshold** (approximately 0.30
   at the 10:1 ratio) rather than the default 0.5.

2. **Recalibrate quarterly** as CLV estimates and offer costs are updated.
   Use `src/evaluation/threshold.optimal_threshold(y_val, proba, cost_fn, cost_fp)`
   to recompute programmatically from updated cost assumptions.

3. **Report business metrics, not just AUC**.  Frame model performance as
   expected cost saved per month, number of churners caught, and number of
   unnecessary offers sent — not as abstract AUC or F1 scores.

4. **Segment-specific thresholds** are a natural next step.  High-CLV
   customers warrant a lower threshold (catch every churner); low-CLV
   customers can tolerate a higher one.  This requires CLV labels per
   customer, which are not available in the public dataset.

5. **Re-evaluate if churn prevalence shifts**.  If macroeconomic conditions
   increase churn above 35 %, the mild imbalance will intensify and SMOTE may
   then provide a measurable lift over class weighting alone.

## Implementation References

| Component | Location |
|---|---|
| Imbalance strategy comparison | `src/models/imbalance.compare_imbalance_strategies()` |
| Cost curve computation | `src/evaluation/threshold.cost_curve()` |
| Threshold optimisation | `src/evaluation/threshold.optimal_threshold()` |
| Sensitivity analysis | `src/evaluation/threshold.threshold_sensitivity()` |
| Analysis notebook | `notebooks/07_imbalance_threshold.ipynb` |
| Cost matrix config | `configs/config.yaml` → `cost_matrix` section |
