# Baseline model results

12 fits: 2 models x 3 imbalance strategies x 2 distinct feature sets.

Configured feature sets that turned out to be identical, so they were trained once rather than twice: `all` and `safe`. On the engineered features the only tier 1 drop was `Time`, and `Time` is not a model feature, so the safe set has nothing left to remove.

Scored on the **validation** split, 42,558 rows and 55 fraud cases. The test split has not been opened.

## How to read this

The interval is a 95% bootstrap interval on PR AUC, from 500 resamples. With 55 fraud cases in the scoring split, it is wide, and that width
is the point: two models whose intervals overlap cannot be separated on this data,
however different their headline numbers look.

Accuracy is not reported. At this positive rate a model that never predicts fraud
scores over 99.8 percent.

## Results

| Model | Imbalance | Features | PR AUC | Interval | ROC AUC | P@50% recall | P@80% recall | Brier | Fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | smote | all = safe (44) | 0.8725 | [0.786, 0.955] | 0.9852 | 1.000 | 0.882 | 0.00073 | 115.8s |
| Random Forest | class_weight | selected (36) | 0.8671 | [0.775, 0.951] | 0.9822 | 1.000 | 0.957 | 0.00045 | 77.6s |
| Random Forest | smote | selected (36) | 0.8669 | [0.778, 0.952] | 0.9833 | 1.000 | 0.882 | 0.00072 | 114.6s |
| Random Forest | class_weight | all = safe (44) | 0.8592 | [0.769, 0.943] | 0.9801 | 1.000 | 0.865 | 0.00046 | 79.8s |
| Random Forest | none | selected (36) | 0.8586 | [0.769, 0.948] | 0.9812 | 1.000 | 0.849 | 0.00038 | 104.3s |
| Random Forest | none | all = safe (44) | 0.8546 | [0.761, 0.945] | 0.9833 | 1.000 | 0.833 | 0.00039 | 103.4s |
| Logistic Regression | class_weight | all = safe (44) | 0.8395 | [0.741, 0.922] | 0.9838 | 1.000 | 0.638 | 0.04777 | 4.8s |
| Logistic Regression | class_weight | selected (36) | 0.8267 | [0.724, 0.914] | 0.9834 | 1.000 | 0.620 | 0.04498 | 3.1s |
| Logistic Regression | smote | all = safe (44) | 0.8019 | [0.694, 0.891] | 0.9743 | 1.000 | 0.789 | 0.00441 | 6.1s |
| Logistic Regression | smote | selected (36) | 0.7928 | [0.684, 0.882] | 0.9768 | 0.906 | 0.776 | 0.00415 | 2.1s |
| Logistic Regression | none | all = safe (44) | 0.7817 | [0.678, 0.879] | 0.9813 | 0.824 | 0.772 | 0.00053 | 4.4s |
| Logistic Regression | none | selected (36) | 0.7535 | [0.642, 0.858] | 0.9776 | 0.760 | 0.714 | 0.00058 | 2.8s |

## The best run

**Random Forest**, smote, all feature set (44 features).

- PR AUC 0.8725, interval [0.786, 0.955]
- ROC AUC 0.9852
- Precision 0.882 at 80 percent recall
- On the training data it scores 0.9626, a gap of +0.0901

**11 other runs cannot be told apart from it.** Their intervals all
overlap the best run's, so this validation set does not have the resolution to
rank them:

- Logistic Regression, none, all: 0.7817 [0.678, 0.879]
- Logistic Regression, class_weight, all: 0.8395 [0.741, 0.922]
- Logistic Regression, smote, all: 0.8019 [0.694, 0.891]
- Random Forest, none, all: 0.8546 [0.761, 0.945]
- Random Forest, class_weight, all: 0.8592 [0.769, 0.943]
- Logistic Regression, none, selected: 0.7535 [0.642, 0.858]
- Logistic Regression, class_weight, selected: 0.8267 [0.724, 0.914]
- Logistic Regression, smote, selected: 0.7928 [0.684, 0.882]
