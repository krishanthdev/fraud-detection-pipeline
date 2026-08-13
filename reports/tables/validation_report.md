# Data validation report: Credit Card Fraud Detection (ULB)

15 checks ran. 14 passed, 0 failed, 1 warned.

| Check | Result | Detail |
| --- | --- | --- |
| row_count | pass | 284,807 rows, minimum is 100,000 |
| expected_row_count | pass | 284,807 rows, the published dataset has 284,807 |
| dtypes_are_numeric | pass | all 31 columns are numeric |
| no_missing_values | pass | 0 missing cells (0.000000 of the table), limit is 0.0 |
| target_values | pass | Class holds only [0, 1] |
| fraud_rate | pass | 0.1727 percent fraud, expected between 0.0500 and 1.0000 percent |
| amount_not_negative | pass | 0 rows have Amount below 0.0 |
| time_is_ordered | pass | Time increases through the file, so a time split is safe |
| duplicate_rows | warn | 1,081 exact duplicate rows, 19 of them fraud |
| constant_columns | pass | no constant columns |
| duplicates_dropped | pass | 1,081 duplicate rows removed, 283,726 rows remain |
| split_fraud_coverage | pass | every split has at least 30 fraud rows |
| split_train | pass | 198,608 rows, 366 fraud (0.1843 percent) |
| split_validation | pass | 42,558 rows, 55 fraud (0.1292 percent) |
| split_test | pass | 42,560 rows, 52 fraud (0.1222 percent) |
