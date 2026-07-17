# Data Card: GC

## Nguồn và file

- Dataset: German Credit Data.
- Active file: `data/raw/gc/german.data`.
- Documentation: `data/raw/gc/german.doc`, `data/raw/gc/Index`.
- Checksum: `B21F3D81DB8071257D5FF1DEAEBA1FD4303B62712E6FCC9715C7A86202CB5871`.
- Source: UCI German Credit Data, Prof. Hans Hofmann.
- License/access: Chưa xác định; UCI access terms apply.

## Shape, Target and Classes

| Item | Value |
|---|---:|
| Shape | 1,000 x 21 |
| Input variables | 20 |
| Target column | `target` |
| Raw good | `1`, 700 rows |
| Raw bad | `2`, 300 rows |
| Pipeline mapping | `1 -> 0`, `2 -> 1` |
| Default rate | 0.3000000000 |
| Missing cells | 0 |
| Duplicate rows | 0 |

## Metadata

- Numeric columns: `A2`, `A5`, `A8`, `A11`, `A13`, `A16`, `A18`.
- Categorical columns: `A1`, `A3`, `A4`, `A6`, `A7`, `A9`, `A10`, `A12`, `A14`, `A15`, `A17`, `A19`, `A20`.
- Identifier columns: none.
- Ignored columns: none.

## Preprocessing Caveats

- Source-level preprocessing: the repo uses original symbolic `german.data`; UCI also provides `german.data-numeric`, a transformed numeric file, but that file is not active for core verification.
- Project-level format conversion: none.
- Experimental preprocessing not yet applied: no imputation, WOE, VIF, encoding, scaling, class balancing, split creation or feature selection.

## Usability

Usable for Phase 0 and core partial replication.
