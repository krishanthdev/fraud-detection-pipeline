# Getting the raw data

Nothing in `data/` is committed to git. You have to put the dataset here yourself, once.

## Credit Card Fraud Detection (ULB), the default

This is the dataset the pipeline is pointed at right now.

1. Open <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
2. Sign in to Kaggle and press **Download** (about 144 MB zipped).
3. Unzip it and put `creditcard.csv` in this folder.

The result should be:

```
data/raw/creditcard.csv
```

Check it landed correctly:

```bash
python -m fraud_pipeline ingest
```

The ingest stage checks the file has 284,807 rows, 31 columns and 492 fraud rows before it
writes anything.

### What is inside

| Column | Meaning |
| --- | --- |
| `Time` | Seconds since the first transaction in the file |
| `V1` to `V28` | Anonymised principal components. The original features are confidential |
| `Amount` | Transaction amount |
| `Class` | 1 for fraud, 0 for a normal transaction |

The data covers two days of European card transactions from September 2013.

## IEEE CIS Fraud Detection, added later

Used by the `feature/ieee-cis-dataset` branch.

1. Open <https://www.kaggle.com/c/ieee-fraud-detection/data>
2. Accept the competition rules, then download `train_transaction.csv`,
   `train_identity.csv`, `test_transaction.csv` and `test_identity.csv`.
3. Put them in this folder.

This one is about 1.2 GB unzipped, so leave it until the ULB pipeline is finished.

## If you prefer the Kaggle API

The manual download is the supported path, but the API works too.

```bash
pip install kaggle
# put kaggle.json in ~/.kaggle/ (Windows: %USERPROFILE%\.kaggle\)
kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

`kaggle.json` is in `.gitignore`. Never commit it.
