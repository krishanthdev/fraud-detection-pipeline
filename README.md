# Fraud Detection Pipeline

An end to end credit card fraud detection system. It goes from a raw transaction file to a
trained and registered model, a scoring API, and a dashboard that explains why a
transaction was flagged. Everything runs from the command line and inside Docker.

> **Status: in progress.** Stages 1 to 4 run on the real data, with baseline models trained
> and compared. Stages 5 to 8 are being added one branch at a time. See the branch plan in
> [CONTRIBUTING.md](CONTRIBUTING.md).

## The problem

Banks, card networks and payment companies score every transaction in real time. Two things
make this hard:

1. **The classes are extremely imbalanced.** In the ULB dataset only 492 of 284,807
   transactions are fraud, which is 0.17 percent. A model that predicts "not fraud" every
   single time is 99.83 percent accurate and completely useless. This is why accuracy is not
   reported anywhere in this project.
2. **The two mistakes cost different amounts.** A false negative is fraud that got through.
   A false positive is a real customer whose card was blocked at the till. The model has to
   sit at a chosen point on that trade off, not at an arbitrary 0.5 threshold.

The goal is to catch as much fraud as possible while keeping false alarms low enough that
the business would actually run it.

## The data

| Dataset | Rows | Fraud rows | Fraud rate | Where |
| --- | --- | --- | --- | --- |
| Credit Card Fraud (ULB) | 284,807 | 492 | 0.172 percent | [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| IEEE CIS Fraud Detection | about 590,000 | about 20,600 | about 3.5 percent | [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) |

The ULB set comes first because it is small and clean, so the whole pipeline can be proven
end to end quickly. The IEEE set is added later for richer feature engineering.

The data is never committed. See [data/raw/README.md](data/raw/README.md) for how to get it.

## The pipeline

Eight stages, each one a command. The real pipeline lives in `src/`. Notebooks are for
exploring only.

| Stage | Command | What it does | Status |
| --- | --- | --- | --- |
| 1. Ingest | `make ingest` | Read the raw file, type it, write an interim table | done |
| 2. Validate | `make validate` | Quality rules and the split. Stops the run if the data is wrong | done |
| . EDA | `make eda` | Measure every feature, then select automatically from what it finds | done |
| 3. Features | `make features` | Build features, then reselect over the engineered set | done |
| 4. Train | `make train` | Sweep models, imbalance strategies and feature sets | baselines done |
| 5. Evaluate | `make evaluate` | Metrics, threshold tuning, SHAP, results table | |
| 6. Register | `make register` | Log to MLflow and promote a champion model | |
| 7. Serve | `make serve` and `make app` | FastAPI endpoint and Streamlit dashboard | |
| 8. Package | `make docker-up` | The whole thing in containers | |

`make all` runs stages 1 to 6 in order. On Windows use `.\tasks.ps1 all` instead, since
Windows does not ship `make`.

### Data quality and the split

Stage 2 runs ten named checks before anything is trained, and writes
[reports/tables/validation_report.md](reports/tables/validation_report.md). Two of them
found something worth knowing about this dataset.

**There are 1,081 exact duplicate rows**, 19 of them fraud. With 28 continuous components
matching to full precision, these are double entries rather than coincidence. They are
dropped, which leaves 283,726 rows. They do not leak across the split, because copies share
a timestamp and land in the same block, but keeping them would let a model count the same
transaction twice.

**The split is by time, not at random.** A real system always scores transactions made
after the ones it learned from, and fraud patterns drift. A random split quietly lets the
model learn from the future and reports a score the deployed system would never reach.

| Split | Rows | Fraud rows | Fraud rate |
| --- | --- | --- | --- |
| Train | 198,608 | 366 | 0.184 percent |
| Validation | 42,558 | 55 | 0.129 percent |
| Test | 42,560 | 52 | 0.122 percent |

That fraud rate is not flat across the three parts, and the reason turned out to be more
interesting than plain drift. See the next section.

It also sets a limit on how much the results can be trusted: with 52 fraud cases in the
test set, one extra catch moves recall by about two points. The results table reports that
uncertainty rather than quoting four decimal places as if they were solid.

## What the exploration found

`make eda` measures every feature on the training split, then selects features from what it
measures. It never reads the test split, and the config refuses to let it. Full output in
[reports/tables/eda_report.md](reports/tables/eda_report.md).

### Fraud follows the clock, not just the calendar

![fraud by hour](reports/figures/07_fraud_by_hour.png)

Fraud is **5.1 times more likely between 01:00 and 05:00** (0.742 percent) than during the
rest of the day (0.145 percent), and those hours are exactly when transaction volume is at
its lowest.

This changes how the split table above should be read. Validation covers only 12:52 to
18:03 on the clock, so it contains none of the high fraud window. Restricting training to
that same clock window drops its fraud rate from 0.184 to 0.156 percent, close to
validation's 0.129. **Most of the gap between the splits is which hours they contain, not
fraud behaviour changing over the two days.**

It also means the chronological split has a real limitation on this dataset. The file is
only 48 hours long, so no single split can cover a full daily cycle, and neither validation
nor test contains the small hours. The IEEE CIS dataset spans months and will not have this
problem.

### How much signal is real

![univariate ranking](reports/figures/04_univariate_ranking.png)

The threshold here is measured, not chosen. Shuffling the target 40 times and recording the
best AUC any of the 30 features still reached by chance gives an average of 0.535 and a peak
of 0.557. So on this dataset, with only 366 fraud rows in training, **a feature carrying no
information at all can still score about 0.55.** Anything below that is indistinguishable
from a column of random numbers.

Taking the maximum across all features on each shuffle is the point. Testing 30 features and
keeping the best is 30 chances to be fooled, and the maximum statistic accounts for that.

### No leakage, no duplicates

Nothing reaches the 0.99 AUC leakage threshold. No two features are correlated above 0.95,
and the strongest pair anywhere is V2 with Amount at 0.55.

Both are the expected answers rather than lucky ones. V1 to V28 are principal components, so
they are orthogonal to each other by construction, and no original column survives that
could encode the answer. The checks still run on every pipeline run, because the IEEE CIS
dataset has real named columns where a leak is a genuine possibility.

## Automatic feature selection

The selection system applies rules with measured thresholds and records the evidence behind
every drop. Output in
[reports/tables/feature_selection.md](reports/tables/feature_selection.md).

Rules come in two tiers, and the split between them matters.

**Tier 1 is correctness.** Leakage, duplicates, constant columns, and features whose next
period values fall outside the training range. Keeping any of these is a mistake rather than
a preference.

**Tier 2 is judgement.** Dropping features with no measurable signal. This is arguable,
because univariate screening cannot see interactions, so it is separated out and can be
switched off.

That produces three feature sets, and the pipeline can train on any of them:

| Set | Features | What it is |
| --- | --- | --- |
| `all` | 30 | no selection, the honest baseline |
| `safe` | 29 | tier 1 only |
| `selected` | 26 | tier 1 and tier 2 |

Whether selection actually helps is a question for the results table, not an assumption.

### What got dropped, and why

| Feature | Tier | Reason |
| --- | --- | --- |
| `Time` | 1 | 0.00 percent of validation values fall inside the training range |
| `V22` | 2 | AUC 0.528 and KS 0.080, both below the noise ceilings |
| `V26` | 2 | AUC 0.527 and KS 0.086, both below the noise ceilings |
| `V13` | 2 | AUC 0.505 and KS 0.083, both below the noise ceilings |

`Time` is the interesting one. It only ever increases, so every future value is larger than
anything the model trained on and the model is being asked to extrapolate past the edge of
its experience. A shifted distribution is survivable. No overlap at all is not. It stays in
the data as a pipeline input, because hour of day gets derived from it, and hour of day both
carries more signal (0.61 AUC against 0.58) and covers the range completely.

### A feature has to fail two tests, not one

Four features sit below the AUC ceiling and are still kept: `Amount`, `V23`, `V15` and
`V25`. They survive because AUC only sees whether fraud sits consistently high or low, and
they separate the classes without doing that.

`Amount` is the clearest case. It scores 0.548 AUC, below the ceiling, while its KS distance
is 0.260, nearly three times the KS ceiling. Fraudulent amounts really are distributed
differently, they are just not consistently larger or smaller. An AUC only version of this
rule dropped `Amount`, and that was wrong.

## Feature engineering

`make features` builds 15 new features from `Time` and `Amount`, carries the 28 components
and `Amount` through unchanged, and then runs the same selection rules again over the
engineered set. Full output in
[reports/tables/feature_engineering.md](reports/tables/feature_engineering.md).

The brief also asks for rolling aggregates per card and category encodings. Neither is
possible on this dataset, and it is better to say so than to skip it quietly: ULB contains
`Time`, `V1` to `V28`, `Amount` and `Class`, and nothing identifies a cardholder. The same
ideas are applied to the global transaction stream instead. Per card versions arrive with
IEEE CIS.

### Two rules that pull in opposite directions

**A feature may look backwards, never forwards.** A rolling average over the previous 50
transactions is exactly what a deployed system has at scoring time, so these are computed
across the whole ordered stream including split boundaries. Building them inside each split
separately would be worse, because validation would start from a cold baseline that
production never has.

**A fitted statistic may only come from training.** Means, standard deviations, quantiles.
These summarise the data they are fitted on, so fitting one on everything folds the future
into every row.

Only two numbers are fitted here, both used to fill the gap at the very start of the stream
where a history feature has no history yet. Everything else is pointwise or built from a
row's own past, which is why there is so little to fit.

### Which new features earned their place

10 of 15 survived selection. The pattern is consistent and makes sense.

| Kept | AUC | | Dropped | AUC |
| --- | --- | --- | --- | --- |
| `txn_rate_10` | 0.626 | | `amount_roll_mean_10` | 0.504 |
| `txn_rate_50` | 0.622 | | `amount_roll_std_10` | 0.510 |
| `hour` | 0.610 | | `amount_dev_10` | 0.508 |
| `hour_sin` | 0.593 | | `amount_roll_mean_50` | 0.533 |
| `seconds_since_prev` | 0.586 | | `amount_roll_std_50` | 0.536 |

**The raw baselines were dropped and the comparisons were kept.** `amount_roll_mean_10`
describes the recent stream, not the transaction being scored, so it carries almost nothing
on its own. `amount_ratio_10`, which compares this amount against that baseline, scores
0.554 with a KS of 0.222. The useful feature was never the baseline, it was the distance
from it.

**The time features are the strongest thing built.** `txn_rate_10` at 0.626 beats every
engineered feature and beats raw `Time` at 0.584, which is the feature it effectively
replaces. That follows directly from the EDA finding: fraud concentrates in the small hours,
and those are exactly the hours when the stream is quiet.

### A caveat on the time features

`hour` has a PSI of 8.99 and `txn_rate_10` a PSI of 1.21, the two largest in the set. That
is not drift in the usual sense, it is the 48 hour window problem again: training covers a
full daily cycle and validation covers only 12:52 to 18:03.

So the strongest new features are also the ones validation is least able to check. They stay
in, because their range coverage is complete and dropping the best signal to avoid an
awkward validation set would be the wrong trade. It does mean the validation score will
understate how much these features are worth, and it is a reason to treat the eventual
train to validation gap with some suspicion rather than as pure overfitting.

## Baseline results

Logistic regression and random forest, each under three imbalance strategies and two feature
sets. XGBoost, LightGBM and the neural net arrive on `feature/advanced-models`.

**These are validation numbers. The test split has not been opened.** Once a held out set has
been used to choose between twelve candidates it is no longer held out, so stage 5 opens it
once, after the threshold is fixed. Full output in
[reports/tables/baseline_results.md](reports/tables/baseline_results.md).

| Model | Imbalance | Features | PR AUC | 95% interval | ROC AUC | P@80% recall | Brier | Fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | SMOTE | 44 | 0.8725 | [0.786, 0.955] | 0.9852 | 0.882 | 0.00073 | 116s |
| Random Forest | class weights | 36 | 0.8671 | [0.775, 0.951] | 0.9822 | **0.957** | 0.00045 | 78s |
| Random Forest | SMOTE | 36 | 0.8669 | [0.778, 0.952] | 0.9833 | 0.882 | 0.00072 | 115s |
| Random Forest | class weights | 44 | 0.8592 | [0.769, 0.943] | 0.9801 | 0.865 | 0.00046 | 80s |
| Random Forest | none | 36 | 0.8586 | [0.769, 0.948] | 0.9812 | 0.849 | 0.00038 | 104s |
| Random Forest | none | 44 | 0.8546 | [0.761, 0.945] | 0.9833 | 0.833 | 0.00039 | 103s |
| Logistic Regression | class weights | 44 | 0.8395 | [0.741, 0.922] | 0.9838 | 0.638 | 0.04777 | 5s |
| Logistic Regression | class weights | 36 | 0.8267 | [0.724, 0.914] | 0.9834 | 0.620 | 0.04498 | 3s |
| Logistic Regression | SMOTE | 44 | 0.8019 | [0.694, 0.891] | 0.9743 | 0.789 | 0.00441 | 6s |
| Logistic Regression | SMOTE | 36 | 0.7928 | [0.684, 0.882] | 0.9768 | 0.776 | 0.00415 | 2s |
| Logistic Regression | none | 44 | 0.7817 | [0.678, 0.879] | 0.9813 | 0.772 | 0.00053 | 4s |
| Logistic Regression | none | 36 | 0.7535 | [0.642, 0.858] | 0.9776 | 0.714 | 0.00058 | 3s |

### The most important number here is the interval

**All 12 runs are statistically indistinguishable.** Every interval overlaps the best run's.
With 55 fraud cases in validation, this data cannot rank these models, and the gap from
0.8725 down to 0.7535 is within the noise.

Without the interval, that table reads as a clean ranking and every gap in it looks like a
finding. This is what the bootstrap was added for, and it is worth more than the winner's
name.

### What can be said

Averaging over the other axes is crude but it is the right first read.

| Model | Mean PR AUC | | Imbalance | Mean PR AUC | | Features | Mean PR AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.863 | | class weights | 0.849 | | 44 | 0.835 |
| Logistic Regression | 0.802 | | SMOTE | 0.835 | | 36 | 0.828 |
| | | | none | 0.814 | | | |

Random forest is ahead of logistic regression everywhere. Imbalance handling helps, and
neither strategy transforms the problem, which is worth knowing before spending a week
tuning SMOTE.

**Feature selection is roughly neutral.** 44 features average 0.835 and 36 average 0.828, a
gap well inside the noise. So the case for selection here is inference cost, fit time and
interpretability, not accuracy, and it would be dishonest to present it as a scoring win.
That was an open question when the selection system was built, and this is the answer.

### Class weighting wrecks calibration

Logistic regression with class weights has the best PR AUC of any linear model and a Brier
score of **0.0478, against 0.00053 unweighted**. That is around 90 times worse calibrated.

It ranks well and its probabilities are fiction. For a system whose output is a fraud score
rather than a yes or no, that matters, and it is invisible if you only look at ranking
metrics. Random forest with class weights does not have this problem (0.00046).

### The primary metric picks a different winner than the operating point does

Best PR AUC is random forest with SMOTE on 44 features. But look at precision at 80 percent
recall, which is much closer to how the system would actually be run:

| | Best PR AUC | Best at the operating point |
| --- | --- | --- |
| | RF, SMOTE, 44 features | RF, class weights, 36 features |
| PR AUC | **0.8725** | 0.8671 |
| Precision at 80 percent recall | 0.882 | **0.957** |
| Brier score | 0.00073 | **0.00045** |
| Features | 44 | **36** |
| Fit time | 116s | **78s** |

PR AUC integrates over every threshold, including ones nobody would ever run at. At the
threshold a cost model would actually choose, the second model catches the same fraud with
roughly a third of the false alarms, is better calibrated, and is cheaper to fit and serve.

This is why stage 5 picks the champion against the cost model rather than against the
headline metric.

## Quick start

You need Python 3.12 and Git. Docker is optional but it is how the finished project runs.

```bash
git clone https://github.com/krishanthdev/fraud-detection-pipeline.git
cd fraud-detection-pipeline
```

On Linux or macOS:

```bash
make setup
```

On Windows:

```bash
.\tasks.ps1 setup
```

Then download the dataset into `data/raw/` as described in
[data/raw/README.md](data/raw/README.md), and run the pipeline:

```bash
make all
```

Check that the config loaded correctly at any time:

```bash
python -m fraud_pipeline show-config
```

## Configuration

Everything the pipeline does is controlled by [configs/config.yaml](configs/config.yaml).
Nothing important is hard coded. The file is validated on load, so a typo fails immediately
with a clear message.

Any value can be overridden without editing the file:

```bash
python -m fraud_pipeline --set project.seed=7 --set models.xgboost.max_depth=8 train
```

## Reproducibility

- One seed in the config drives Python, numpy, scikit learn and torch.
- Splits are done by time, so the model is never trained on transactions that happened
  after the ones it is tested on.
- Dependencies are pinned to exact versions.
- Every run is logged to MLflow with its parameters, metrics and artifacts.

## Repository layout

```
fraud-detection-pipeline/
  app/                  Streamlit dashboard
  configs/config.yaml   every setting the pipeline uses
  data/                 raw, interim, processed (never committed)
  models/               saved artifacts (never committed, tracked in MLflow)
  notebooks/            exploration only
  reports/figures/      plots used by this README
  reports/tables/       generated results tables
  scripts/              helper scripts
  src/fraud_pipeline/   the pipeline itself
  tests/                pytest suite
  .github/workflows/    lint and test on every pull request
```

## Tech stack

Python 3.12, pandas, scikit learn, XGBoost, LightGBM, PyTorch, imbalanced learn, SHAP,
MLflow, FastAPI, Streamlit, Docker, pytest, ruff, GitHub Actions.

## Licence

MIT. See [LICENSE](LICENSE).
