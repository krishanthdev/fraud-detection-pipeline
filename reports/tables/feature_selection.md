# Feature selection

Active set: **selected**, 26 of 30 features kept.

| Set | Features | What it is |
| --- | --- | --- |
| all | 30 | no selection, the honest baseline |
| safe | 29 | tier 1 only, the drops that are correctness problems |
| selected | 26 | tier 1 and tier 2 |

## Noise ceiling

with the target shuffled 40 times, the best univariate AUC any of 30 features reached by chance averaged 0.5352 and peaked at 0.5567. The 95% quantile, 0.5486, is the AUC noise ceiling. The same procedure puts the KS ceiling at 0.0952, against a chance average of 0.0781.

A feature has to fall below **both** ceilings before it counts as noise. AUC
only sees whether fraud sits consistently high or low, so it is blind to a
feature where fraud clusters in the middle of the range. KS catches that, and
requiring both is what stops the rule throwing away real signal.

## Warnings

- 5 feature(s) shifted between train and validation with PSI at or above 0.25: Time (12.43), V1 (0.94), V3 (0.78), V28 (0.53), V11 (0.33). These are kept, but they are the first place to look if the model does worse on test than on validation.

## Dropped

| Feature | Tier | Why |
| --- | --- | --- |
| Time | 1 | only 0.00% of validation values fall inside the training range (limit 95%), so it cannot generalise |
| V22 | 2 | no measurable signal: AUC 0.5278 below the 0.5486 ceiling and KS 0.0800 below the 0.0952 ceiling |
| V26 | 2 | no measurable signal: AUC 0.5274 below the 0.5486 ceiling and KS 0.0861 below the 0.0952 ceiling |
| V13 | 2 | no measurable signal: AUC 0.5050 below the 0.5486 ceiling and KS 0.0826 below the 0.0952 ceiling |

## Kept

| Feature | AUC | KS | Mutual info | PSI | Range coverage |
| --- | --- | --- | --- | --- | --- |
| V14 | 0.9492 | 0.8423 | 0.0297 | 0.064 | 1.0000 |
| V12 | 0.9405 | 0.7830 | 0.0274 | 0.191 | 1.0000 |
| V4 | 0.9373 | 0.7596 | 0.0201 | 0.105 | 1.0000 |
| V3 | 0.9188 | 0.7052 | 0.0189 | 0.777 | 1.0000 |
| V10 | 0.9172 | 0.8159 | 0.0279 | 0.036 | 1.0000 |
| V11 | 0.9152 | 0.7666 | 0.0253 | 0.327 | 1.0000 |
| V2 | 0.8735 | 0.6539 | 0.0141 | 0.021 | 1.0000 |
| V16 | 0.8556 | 0.7156 | 0.0235 | 0.004 | 0.9999 |
| V7 | 0.8422 | 0.6947 | 0.0175 | 0.078 | 1.0000 |
| V9 | 0.8360 | 0.5668 | 0.0157 | 0.049 | 1.0000 |
| V1 | 0.8128 | 0.5173 | 0.0093 | 0.938 | 1.0000 |
| V17 | 0.8112 | 0.7549 | 0.0302 | 0.032 | 1.0000 |
| V6 | 0.7760 | 0.4913 | 0.0091 | 0.055 | 1.0000 |
| V18 | 0.7639 | 0.5357 | 0.0170 | 0.025 | 1.0000 |
| V5 | 0.7519 | 0.5055 | 0.0113 | 0.151 | 1.0000 |
| V21 | 0.7443 | 0.5377 | 0.0107 | 0.116 | 1.0000 |
| V27 | 0.6965 | 0.4774 | 0.0098 | 0.045 | 1.0000 |
| V19 | 0.6756 | 0.3566 | 0.0053 | 0.018 | 1.0000 |
| V8 | 0.6669 | 0.3839 | 0.0078 | 0.059 | 1.0000 |
| V20 | 0.6455 | 0.3637 | 0.0043 | 0.057 | 1.0000 |
| V28 | 0.6322 | 0.3806 | 0.0074 | 0.534 | 1.0000 |
| V24 | 0.5503 | 0.1068 | 0.0015 | 0.116 | 1.0000 |
| V23 | 0.5481 | 0.2121 | 0.0029 | 0.115 | 1.0000 |
| Amount | 0.5479 | 0.2604 | 0.0069 | 0.003 | 1.0000 |
| V15 | 0.5401 | 0.1024 | 0.0003 | 0.207 | 1.0000 |
| V25 | 0.5155 | 0.1045 | 0.0013 | 0.220 | 1.0000 |

## Kept as pipeline inputs

Columns kept in the data files whatever the rules decided, because later stages derive features from them. Being an input is not the same as being a model feature.

Time, Amount
