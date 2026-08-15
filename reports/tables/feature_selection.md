# Feature selection

Active set: **selected**, 36 of 44 features kept.

| Set | Features | What it is |
| --- | --- | --- |
| all | 44 | no selection, the honest baseline |
| safe | 44 | tier 1 only, the drops that are correctness problems |
| selected | 36 | tier 1 and tier 2 |

## Noise ceiling

with the target shuffled 40 times, the best univariate AUC any of 44 features reached by chance averaged 0.5372 and peaked at 0.5567. The 95% quantile, 0.5518, is the AUC noise ceiling. The same procedure puts the KS ceiling at 0.0955, against a chance average of 0.0801.

A feature has to fall below **both** ceilings before it counts as noise. AUC
only sees whether fraud sits consistently high or low, so it is blind to a
feature where fraud clusters in the middle of the range. KS catches that, and
requiring both is what stops the rule throwing away real signal.

## Warnings

- 9 feature(s) shifted between train and validation with PSI at or above 0.25: hour (8.99), hour_sin (6.23), hour_cos (4.97), txn_rate_50 (1.56), txn_rate_10 (1.21), V1 (0.94), V3 (0.78), V28 (0.53), V11 (0.33). These are kept, but they are the first place to look if the model does worse on test than on validation.

## Dropped

| Feature | Tier | Why |
| --- | --- | --- |
| amount_roll_std_50 | 2 | no measurable signal: AUC 0.5358 below the 0.5518 ceiling and KS 0.0728 below the 0.0955 ceiling |
| amount_roll_mean_50 | 2 | no measurable signal: AUC 0.5327 below the 0.5518 ceiling and KS 0.0604 below the 0.0955 ceiling |
| V22 | 2 | no measurable signal: AUC 0.5278 below the 0.5518 ceiling and KS 0.0800 below the 0.0955 ceiling |
| V26 | 2 | no measurable signal: AUC 0.5274 below the 0.5518 ceiling and KS 0.0861 below the 0.0955 ceiling |
| amount_roll_std_10 | 2 | no measurable signal: AUC 0.5101 below the 0.5518 ceiling and KS 0.0576 below the 0.0955 ceiling |
| amount_dev_10 | 2 | no measurable signal: AUC 0.5084 below the 0.5518 ceiling and KS 0.0724 below the 0.0955 ceiling |
| V13 | 2 | no measurable signal: AUC 0.5050 below the 0.5518 ceiling and KS 0.0826 below the 0.0955 ceiling |
| amount_roll_mean_10 | 2 | no measurable signal: AUC 0.5041 below the 0.5518 ceiling and KS 0.0561 below the 0.0955 ceiling |

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
| txn_rate_10 | 0.6255 | 0.2378 | 0.0031 | 1.211 | 1.0000 |
| txn_rate_50 | 0.6217 | 0.2402 | 0.0024 | 1.559 | 1.0000 |
| hour | 0.6096 | 0.2289 | 0.0047 | 8.986 | 1.0000 |
| hour_sin | 0.5932 | 0.1808 | 0.0024 | 6.234 | 1.0000 |
| seconds_since_prev | 0.5863 | 0.1583 | 0.0024 | 0.065 | 1.0000 |
| amount_ratio_10 | 0.5538 | 0.2217 | 0.0014 | 0.002 | 1.0000 |
| V24 | 0.5503 | 0.1068 | 0.0015 | 0.116 | 1.0000 |
| V23 | 0.5481 | 0.2121 | 0.0029 | 0.115 | 1.0000 |
| Amount | 0.5479 | 0.2604 | 0.0072 | 0.003 | 1.0000 |
| amount_log | 0.5479 | 0.2604 | 0.0068 | 0.003 | 1.0000 |
| amount_ratio_50 | 0.5464 | 0.2257 | 0.0020 | 0.003 | 1.0000 |
| V15 | 0.5401 | 0.1024 | 0.0003 | 0.207 | 1.0000 |
| hour_cos | 0.5394 | 0.1267 | 0.0027 | 4.965 | 1.0000 |
| V25 | 0.5155 | 0.1045 | 0.0013 | 0.220 | 1.0000 |
| amount_dev_50 | 0.5120 | 0.1190 | 0.0002 | 0.002 | 1.0000 |

## Kept as pipeline inputs

Columns kept in the data files whatever the rules decided, because later stages derive features from them. Being an input is not the same as being a model feature.

Amount
