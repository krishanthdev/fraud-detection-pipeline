# Baseline model results

30 fits: 5 models x 3 imbalance strategies x 2 distinct feature sets.

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
| Random Forest | smote | all = safe (44) | 0.8725 | [0.786, 0.955] | 0.9852 | 1.000 | 0.882 | 0.00073 | 118.1s |
| Random Forest | class_weight | selected (36) | 0.8671 | [0.775, 0.951] | 0.9822 | 1.000 | 0.957 | 0.00045 | 71.2s |
| Random Forest | smote | selected (36) | 0.8669 | [0.778, 0.952] | 0.9833 | 1.000 | 0.882 | 0.00072 | 104.7s |
| LightGBM | smote | all = safe (44) | 0.8658 | [0.775, 0.950] | 0.9870 | 1.000 | 0.936 | 0.00032 | 5.6s |
| LightGBM | class_weight | selected (36) | 0.8636 | [0.770, 0.948] | 0.9906 | 1.000 | 0.836 | 0.00031 | 4.5s |
| XGBoost | smote | all = safe (44) | 0.8625 | [0.774, 0.946] | 0.9828 | 1.000 | 0.936 | 0.00035 | 7.2s |
| XGBoost | none | all = safe (44) | 0.8616 | [0.771, 0.943] | 0.9871 | 1.000 | 0.846 | 0.00031 | 5.9s |
| XGBoost | class_weight | selected (36) | 0.8606 | [0.766, 0.942] | 0.9906 | 1.000 | 0.900 | 0.00033 | 5.5s |
| XGBoost | class_weight | all = safe (44) | 0.8599 | [0.766, 0.947] | 0.9780 | 1.000 | 0.863 | 0.00033 | 6.5s |
| Random Forest | class_weight | all = safe (44) | 0.8592 | [0.769, 0.943] | 0.9801 | 1.000 | 0.865 | 0.00046 | 82.2s |
| Random Forest | none | selected (36) | 0.8586 | [0.769, 0.948] | 0.9812 | 1.000 | 0.849 | 0.00038 | 95.9s |
| XGBoost | none | selected (36) | 0.8579 | [0.770, 0.941] | 0.9853 | 1.000 | 0.750 | 0.00030 | 5.1s |
| XGBoost | smote | selected (36) | 0.8550 | [0.761, 0.940] | 0.9905 | 1.000 | 0.880 | 0.00039 | 6.3s |
| Random Forest | none | all = safe (44) | 0.8546 | [0.761, 0.945] | 0.9833 | 1.000 | 0.833 | 0.00039 | 108.7s |
| LightGBM | class_weight | all = safe (44) | 0.8542 | [0.761, 0.938] | 0.9867 | 1.000 | 0.898 | 0.00031 | 5.2s |
| Neural Net | none | selected (36) | 0.8532 | [0.760, 0.934] | 0.9877 | 1.000 | 0.830 | 0.00037 | 11.9s |
| LightGBM | smote | selected (36) | 0.8526 | [0.757, 0.941] | 0.9893 | 1.000 | 0.917 | 0.00035 | 4.8s |
| Neural Net | none | all = safe (44) | 0.8518 | [0.758, 0.935] | 0.9770 | 1.000 | 0.917 | 0.00034 | 12.3s |
| Logistic Regression | class_weight | all = safe (44) | 0.8395 | [0.741, 0.922] | 0.9838 | 1.000 | 0.638 | 0.04777 | 4.6s |
| Neural Net | class_weight | selected (36) | 0.8384 | [0.732, 0.925] | 0.9765 | 1.000 | 0.830 | 0.04018 | 2.9s |
| Logistic Regression | class_weight | selected (36) | 0.8267 | [0.724, 0.914] | 0.9834 | 1.000 | 0.620 | 0.04498 | 2.8s |
| Neural Net | class_weight | all = safe (44) | 0.8198 | [0.708, 0.911] | 0.9637 | 1.000 | 0.772 | 0.03873 | 3.1s |
| Neural Net | smote | selected (36) | 0.8133 | [0.707, 0.908] | 0.9777 | 1.000 | 0.957 | 0.00037 | 17.6s |
| Neural Net | smote | all = safe (44) | 0.8041 | [0.692, 0.894] | 0.9735 | 1.000 | 0.710 | 0.00043 | 15.2s |
| Logistic Regression | smote | all = safe (44) | 0.8019 | [0.694, 0.891] | 0.9743 | 1.000 | 0.789 | 0.00441 | 6.2s |
| Logistic Regression | smote | selected (36) | 0.7928 | [0.684, 0.882] | 0.9768 | 0.906 | 0.776 | 0.00415 | 1.9s |
| Logistic Regression | none | all = safe (44) | 0.7817 | [0.678, 0.879] | 0.9813 | 0.824 | 0.772 | 0.00053 | 4.0s |
| Logistic Regression | none | selected (36) | 0.7535 | [0.642, 0.858] | 0.9776 | 0.760 | 0.714 | 0.00058 | 2.6s |
| LightGBM | none | all = safe **[1]** (44) | 0.3266 | [0.192, 0.457] | 0.7775 | 0.057 | 0.001 | 0.00120 | 4.4s |
| LightGBM | none | selected **[1]** (36) | 0.2964 | [0.178, 0.413] | 0.8266 | 0.455 | 0.001 | 0.00150 | 3.8s |

## The best run

**Random Forest**, smote, all feature set (44 features).

- PR AUC 0.8725, interval [0.786, 0.955]
- ROC AUC 0.9852
- Precision 0.882 at 80 percent recall
- On the training data it scores 0.9626, a gap of +0.0901

## [1] Runs that produced a collapsed model

2 run(s) emitted fewer than 200 distinct scores across 42,558 rows. A model that has stopped splitting still returns valid probabilities and still produces a plausible looking ROC AUC, but almost every row is tied, so the ranking behind those numbers is largely arbitrary.

They are shown for completeness and excluded from the comparison. Read them as evidence that the configuration is wrong, not as measurements of it.

| Run | Distinct scores | PR AUC |
| --- | --- | --- |
| LightGBM, none, all | 13 | 0.3266 |
| LightGBM, none, selected | 5 | 0.2964 |

**27 other runs cannot be told apart from it.** Their intervals all
overlap the best run's, so this validation set does not have the resolution to
rank them:

- Logistic Regression, none, all: 0.7817 [0.678, 0.879]
- Logistic Regression, class_weight, all: 0.8395 [0.741, 0.922]
- Logistic Regression, smote, all: 0.8019 [0.694, 0.891]
- Random Forest, none, all: 0.8546 [0.761, 0.945]
- Random Forest, class_weight, all: 0.8592 [0.769, 0.943]
- XGBoost, none, all: 0.8616 [0.771, 0.943]
- XGBoost, class_weight, all: 0.8599 [0.766, 0.947]
- XGBoost, smote, all: 0.8625 [0.774, 0.946]
