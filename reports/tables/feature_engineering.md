# Feature engineering

29 original columns carried through, 15 new features built, 44 in total.

After selection the active set keeps **36**, of which 10 are new.

## What was fitted, and on what

Only two numbers are fitted, both on the train split (198,608 rows). They fill the gap at the very start of the
stream where a history feature has no history yet.

- amount median: 23.00
- gap between transactions, median: 0.00 seconds

Everything else is either pointwise or built from a row's own earlier history, so
there is nothing else that could carry information from one split into another.

## The new features

| Feature | AUC | KS | PSI | Range coverage | Kept |
| --- | --- | --- | --- | --- | --- |
| amount_log | 0.5479 | 0.2604 | 0.003 | 1.0000 | yes |
| hour | 0.6096 | 0.2289 | 8.986 | 1.0000 | yes |
| hour_sin | 0.5932 | 0.1808 | 6.234 | 1.0000 | yes |
| hour_cos | 0.5394 | 0.1267 | 4.965 | 1.0000 | yes |
| seconds_since_prev | 0.5863 | 0.1583 | 0.065 | 1.0000 | yes |
| amount_roll_mean_10 | 0.5041 | 0.0561 | 0.033 | 1.0000 | no |
| amount_roll_std_10 | 0.5101 | 0.0576 | 0.028 | 1.0000 | no |
| amount_dev_10 | 0.5084 | 0.0724 | 0.001 | 1.0000 | no |
| amount_ratio_10 | 0.5538 | 0.2217 | 0.002 | 1.0000 | yes |
| txn_rate_10 | 0.6255 | 0.2378 | 1.211 | 1.0000 | yes |
| amount_roll_mean_50 | 0.5327 | 0.0604 | 0.145 | 1.0000 | no |
| amount_roll_std_50 | 0.5358 | 0.0728 | 0.068 | 1.0000 | no |
| amount_dev_50 | 0.5120 | 0.1190 | 0.002 | 1.0000 | yes |
| amount_ratio_50 | 0.5464 | 0.2257 | 0.003 | 1.0000 | yes |
| txn_rate_50 | 0.6217 | 0.2402 | 1.559 | 1.0000 | yes |

## Dropped

| Feature | New | Tier | Why |
| --- | --- | --- | --- |
| amount_roll_std_50 | yes | 2 | no measurable signal: AUC 0.5358 below the 0.5518 ceiling and KS 0.0728 below the 0.0955 ceiling |
| amount_roll_mean_50 | yes | 2 | no measurable signal: AUC 0.5327 below the 0.5518 ceiling and KS 0.0604 below the 0.0955 ceiling |
| V22 | no | 2 | no measurable signal: AUC 0.5278 below the 0.5518 ceiling and KS 0.0800 below the 0.0955 ceiling |
| V26 | no | 2 | no measurable signal: AUC 0.5274 below the 0.5518 ceiling and KS 0.0861 below the 0.0955 ceiling |
| amount_roll_std_10 | yes | 2 | no measurable signal: AUC 0.5101 below the 0.5518 ceiling and KS 0.0576 below the 0.0955 ceiling |
| amount_dev_10 | yes | 2 | no measurable signal: AUC 0.5084 below the 0.5518 ceiling and KS 0.0724 below the 0.0955 ceiling |
| V13 | no | 2 | no measurable signal: AUC 0.5050 below the 0.5518 ceiling and KS 0.0826 below the 0.0955 ceiling |
| amount_roll_mean_10 | yes | 2 | no measurable signal: AUC 0.5041 below the 0.5518 ceiling and KS 0.0561 below the 0.0955 ceiling |
