# Model registry

**Promoted.** No champion is registered yet, so this one takes the title unopposed.

## The decision

| | |
| --- | --- |
| Challenger | `neural_net__smote__selected` |
| Challenger expected cost | 176.3000 |
| Current champion | none |
| Champion expected cost | n/a |
| Margin required | 0.0050 |
| Margin achieved | n/a |
| Registered version | 1 |

## Why a margin

Without one, every rerun swaps the model whenever the number moves at all, and with about fifty fraud cases in the scoring split it moves for no reason. A margin turns **different** into **better**. Its size is a decision to argue about rather than a default to ignore.

## Why cost rather than the headline metric

Stage 5 chose this champion on expected cost at a tuned threshold. Judging promotion on anything else would let a model win the selection and lose the promotion, which is not a difference anyone could explain to the team running it.

The score used is the **validation** cost, 176.30. Test is reported and never used to choose. Promoting on the test number would fold the held out estimate back into the decision it is supposed to be independent of.

## What the promoted model actually does

- Threshold 0.919734
- Test recall 0.635, catching 33 of 52 frauds
- Test value recall **0.515**, which is the share of fraudulent money recovered and the number worth watching
- 3 false alarms across 42,560 transactions
