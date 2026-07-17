# Data Card: TC

## Nguồn và file

- Dataset: Default of Credit Card Clients.
- Active file: `data/raw/tc/default of credit card clients.xls`.
- Checksum: `30C6BE3ABD8DCFD3E6096C828BAD8C2F011238620F5369220BD60CFC82700933`.
- Source: UCI Default of Credit Card Clients, Yeh and Lien.
- License/access: Chưa xác định; UCI access terms apply.

## Shape, Target and Classes

| Item | Value |
|---|---:|
| Shape | 30,000 x 25 |
| Input variables | 23 |
| Identifier | `ID` |
| Target column | `default payment next month` |
| Good/non-default | `0`, 23,364 rows |
| Bad/default | `1`, 6,636 rows |
| Default rate | 0.2212000000 |
| Missing cells | 0 |
| Duplicate rows including ID | 0 |

## Metadata

- Numeric columns: `LIMIT_BAL`, `AGE`, `BILL_AMT1`, `BILL_AMT2`, `BILL_AMT3`, `BILL_AMT4`, `BILL_AMT5`, `BILL_AMT6`, `PAY_AMT1`, `PAY_AMT2`, `PAY_AMT3`, `PAY_AMT4`, `PAY_AMT5`, `PAY_AMT6`.
- Categorical/ordinal encoded columns: `SEX`, `EDUCATION`, `MARRIAGE`, `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_5`, `PAY_6`.
- Identifier columns: `ID`.
- Ignored columns: `ID`.

## Preprocessing Caveats

- Source-level preprocessing: several categorical/ordinal variables are numerically encoded in the source workbook.
- Project-level format conversion: none.
- Experimental preprocessing not yet applied: no imputation, WOE, VIF, encoding, scaling, class balancing, split creation or feature selection.

## Usability

Usable for Phase 0 and core partial replication. `ID` must not be used as a model input.
