# Data Card: AC

## Nguồn và file

- Dataset: Australian Credit Approval.
- Active file: `data/raw/ac/australian.dat`.
- Documentation: `data/raw/ac/australian.doc`, `data/raw/ac/Index`.
- Checksum: `3ABE5AF151AFA50B999A9BA21CBC884E80D80A4050B716CF88B34A2E6ECB731C`.
- Source: UCI Australian Credit Approval / Quinlan credit approval data.
- License/access: Chưa xác định; UCI access terms apply.

## Shape, Target and Classes

| Item | Value |
|---|---:|
| Shape | 690 x 15 |
| Input variables | 14 |
| Target column | `target` |
| Good/non-default | `0`, 383 rows |
| Bad/default | `1`, 307 rows |
| Default rate | 0.4449275362 |
| Missing cells | 0 |
| Duplicate rows | 0 |

Target caveat: `australian.doc` lists class `+` / class 2 as 307 rows and class `-` / class 1 as 383 rows. The paper's default rate 0.445 matches 307/690, so the project maps raw value `1` to bad/default and raw value `0` to good/non-default.

## Metadata

- Numeric columns: `A2`, `A3`, `A7`, `A10`, `A13`, `A14`.
- Categorical columns: `A1`, `A4`, `A5`, `A6`, `A8`, `A9`, `A11`, `A12`.
- Identifier columns: none.
- Ignored columns: none.

## Preprocessing Caveats

- Source-level preprocessing: source documentation says missing values were replaced by mean/mode and categorical labels were recoded to numeric labels before distribution.
- Project-level format conversion: none.
- Experimental preprocessing not yet applied: no imputation, WOE, VIF, encoding, scaling, class balancing, split creation or feature selection.

## Usability

Usable for Phase 0 and core partial replication, with the target-semantics caveat above.
