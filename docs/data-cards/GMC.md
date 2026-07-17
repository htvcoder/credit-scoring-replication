# Data Card: GMC

## Nguồn và file

- Dataset: Give Me Some Credit.
- Active file: `data/raw/gmc/cs-training.csv`.
- Dictionary: `data/raw/gmc/Data Dictionary.xls`.
- Checksum: `1BD46DA486A5708C58C7B01A034FAE2A13B327F6F7B62EA7BA4FE3B5824B24AC`.
- Source: Kaggle Give Me Some Credit competition.
- License/access: Kaggle competition terms; login/credentials may be required.

## Shape, Target and Classes

| Item | Value |
|---|---:|
| Shape | 150,000 x 12 |
| Input variables | 10 |
| Identifier/index | `Unnamed: 0` |
| Target column | `SeriousDlqin2yrs` |
| Good/non-default | `0`, 139,974 rows |
| Bad/default | `1`, 10,026 rows |
| Default rate | 0.0668400000 |
| Missing cells | 33,655 |
| Duplicate rows including index | 0 |

## Metadata

- Numeric columns: `RevolvingUtilizationOfUnsecuredLines`, `age`, `NumberOfTime30-59DaysPastDueNotWorse`, `DebtRatio`, `MonthlyIncome`, `NumberOfOpenCreditLinesAndLoans`, `NumberOfTimes90DaysLate`, `NumberRealEstateLoansOrLines`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfDependents`.
- Categorical columns: none in the active metadata.
- Identifier columns: `Unnamed: 0`.
- Ignored columns: `Unnamed: 0`.

## Preprocessing Caveats

- Source-level preprocessing: Kaggle CSV includes an index column `Unnamed: 0`; it is not a model input.
- Project-level format conversion: none.
- Experimental preprocessing not yet applied: no imputation, WOE, VIF, encoding, scaling, class balancing, split creation or feature selection.

## Usability

Usable for Phase 0 and core partial replication. Kaggle access terms must be respected before redistributing raw data.
