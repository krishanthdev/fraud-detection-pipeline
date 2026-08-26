# What the model is actually using

SHAP explanations for `neural_net__smote__selected`, the promoted champion, on 300 test transactions. Method: **kernel**.

The model agnostic kernel explainer, which is approximate. Values are in probability units, so a contribution of 0.02 means that feature moved the fraud probability by two percentage points.

## What drives the score overall

| Feature | Mean absolute SHAP | Share |
| --- | --- | --- |
| V14 | 0.07267 | 28.4% |
| V12 | 0.03599 | 14.1% |
| V4 | 0.02126 | 8.3% |
| V10 | 0.01581 | 6.2% |
| V7 | 0.00713 | 2.8% |
| V17 | 0.00711 | 2.8% |
| V2 | 0.00670 | 2.6% |
| V3 | 0.00564 | 2.2% |
| V6 | 0.00547 | 2.1% |
| V8 | 0.00519 | 2.0% |

This is a sanity check as much as a finding. The strongest drivers are the same components the exploration flagged as most separable, which is what should happen. An explanation that disagreed with the univariate analysis would mean one of the two is wrong.

## Worked examples

At the operating threshold of 0.9197. These deliberately include the model being wrong, not only the model being right, because the failures are what a reviewer learns from.

### Caught fraud: 0.01, actually fraud

Model score 1.0000 against a baseline of 0.00000.

| Feature | Value | Contribution | Pushing |
| --- | --- | --- | --- |
| V14 | -8.761 | +0.61519 | towards fraud |
| V12 | -6.150 | +0.32551 | towards fraud |
| V28 | 1.521 | -0.16833 | away from fraud |
| V4 | 3.602 | +0.11946 | towards fraud |
| V10 | -3.992 | +0.09232 | towards fraud |
| V7 | -4.397 | +0.03712 | towards fraud |

### Caught fraud: 57.73, actually fraud

Model score 1.0000 against a baseline of 0.00000.

| Feature | Value | Contribution | Pushing |
| --- | --- | --- | --- |
| V14 | -8.724 | +0.59661 | towards fraud |
| V12 | -5.969 | +0.30013 | towards fraud |
| V28 | 1.405 | -0.16143 | away from fraud |
| V10 | -4.869 | +0.10419 | towards fraud |
| V7 | -3.817 | +0.06931 | towards fraud |
| V4 | 1.844 | +0.06883 | towards fraud |

### False alarm: 11.32, actually legitimate

Model score 0.9971 against a baseline of 0.00000.

| Feature | Value | Contribution | Pushing |
| --- | --- | --- | --- |
| V14 | -4.212 | +0.37884 | towards fraud |
| V4 | 4.492 | +0.31739 | towards fraud |
| V12 | -1.746 | +0.21915 | towards fraud |
| V16 | 1.876 | -0.08174 | away from fraud |
| V11 | -0.934 | -0.07826 | away from fraud |
| txn_rate_10 | 30.000 | +0.06318 | towards fraud |

### False alarm: 451.81, actually legitimate

Model score 0.9519 against a baseline of 0.00000.

| Feature | Value | Contribution | Pushing |
| --- | --- | --- | --- |
| V4 | 4.513 | +0.26775 | towards fraud |
| V14 | -2.245 | +0.24047 | towards fraud |
| Amount | 451.810 | +0.16370 | towards fraud |
| amount_ratio_50 | 8.358 | +0.12379 | towards fraud |
| V16 | 1.735 | -0.11166 | away from fraud |
| V18 | 0.643 | +0.10028 | towards fraud |
