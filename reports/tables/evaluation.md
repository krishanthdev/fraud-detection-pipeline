# Evaluation

Champion: **Neural Net**, smote, selected feature set. Chosen from 28 healthy runs by expected cost at a tuned threshold, not by the headline metric.

## The order this was done in

1. Every candidate was tuned on **validation** and the cheapest promoted.
2. The threshold was fixed at **0.919734**.
3. The **test** split was opened once and scored at that threshold.

Nothing after step 2 changed anything from before it. Once a held out set has informed a decision it stops being an estimate of future performance, and there is no way to undo that or to measure how much optimism it added.

## Test results, at the frozen threshold

| Measure | Validation | Test |
| --- | --- | --- |
| Recall (cases) | 0.800 | **0.635** |
| Precision | 0.957 | **0.917** |
| F1 | 0.871 | 0.750 |
| Value recall | 0.980 | **0.515** |
| Frauds caught | 44 of 55 | 33 of 52 |
| False alarms | 2 | 3 |
| Total alerts | 46 | 36 |
| Value caught | 8,230 | 3,176 |
| Value missed | 166 | 2,993 |
| Expected cost | 176 | 3,008 |

Recall moved by -0.165 from validation to test. Some drop is expected, because the threshold was chosen on validation. A large rise would be luck rather than improvement.

**Value recall is the number to read, and it is the worst news here.** Share of fraud *cases* caught and share of fraudulent *money* caught are different numbers, and only the second one is what a fraud team is measured on.

| Split | Frauds | Caught | Missed | Largest missed | Mean missed | Value recall |
| --- | --- | --- | --- | --- | --- | --- |
| validation | 55 | 44 | 11 | 109 | 15 | 0.980 |
| test | 52 | 33 | 19 | 1,097 | 158 | 0.515 |

On validation the model missed only cheap frauds, the largest worth 109. On test it missed the expensive ones, the largest worth 1,097. Case recall fell by a sixth and value recall almost halved, because the two are not tied together.

This is the clearest limitation in the project. With roughly fifty fraud cases per split, whether the model happens to catch the handful of large ones swings value recall enormously, and validation gave an optimistic answer that test did not repeat. Any claim about money saved should carry that caveat.

## Why the cost model matters

Tuned on validation under each cost model. The false alarm price is fixed at 5.00 for both.

| Cost model | Threshold | Recall | Precision | TP | FP | Value caught | Value recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flat | 0.9197 | 0.800 | 0.957 | 44 | 2 | 8,230 | 0.980 |
| amount | 0.9197 | 0.800 | 0.957 | 44 | 2 | 8,230 | 0.980 |

`flat` charges every missed fraud 120, which is the mean fraudulent amount and therefore the right flat number. It is still the wrong summary: the amounts are severely skewed, so a flat charge treats the largest fraud in the file and a 1.00 fraud as equally worth catching.

**On this champion the two agree**, landing on the same threshold. That is not true of every model in the sweep and it is worth saying rather than glossing over. This model's scores are strongly bimodal, so there is a wide gap between the last fraud it is confident about and the next candidate, and moving the price of a mistake within any sensible range does not cross that gap.

Where the two do disagree, on LightGBM and random forest with SMOTE, the amount model gives up about three fraud cases to remove between seven and sixteen false alarms, and the fraudulent value it recovers is unchanged to three decimal places. Counting frauds and recovering money are different objectives, and the second is cheaper to serve.

## What the false alarm price is worth

This is the one number the data cannot supply. It is the support and churn expense of wrongly blocking a customer, which is a business input, so here is what the threshold would be across a range rather than a single answer presented as derived.

| False alarm cost | Threshold | Recall | Precision | Alerts | Value recall |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.9197 | 0.800 | 0.957 | 46 | 0.980 |
| 2 | 0.9197 | 0.800 | 0.957 | 46 | 0.980 |
| 5 | 0.9197 | 0.800 | 0.957 | 46 | 0.980 |
| 10 | 0.9197 | 0.800 | 0.957 | 46 | 0.980 |
| 25 | 0.9197 | 0.800 | 0.957 | 46 | 0.980 |
| 50 | 0.9197 | 0.800 | 0.957 | 46 | 0.980 |

The configured price is 5.00. Find your own number in that table and read the row.

## How the champion was chosen

28 healthy runs were each given their own tuned threshold, because comparing models at a shared threshold measures how their probability scales happen to line up rather than how good they are.

- Promoted: `neural_net__smote__selected`, validation cost 176
- Runner up: `xgboost__smote__all`, validation cost 181, a margin of 5
