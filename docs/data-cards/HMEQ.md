# Data Card: HMEQ

## Nguồn và phiên bản

- Dataset: Home Equity Loan, HMEQ.
- Registry entry: `data/datasets.yaml`, dataset id `hmeq`.
- File hiện có trước remediation: `data/raw/hmeq/hmeq.csv`.
- File full dùng cho core replication: `data/raw/hmeq/hmeq_full.csv`.
- Nguồn xác minh schema và quy mô: SAS documentation cho `Sampsio.Hmeq`, mô tả 5.960 mortgage applicants và biến response `Bad`.
- Nguồn tải artifact hiện tại: `https://raw.githubusercontent.com/Carl-Lejerskar/HMEQ/master/hmeq.csv`.
- Lưu ý checksum: file tải hiện tại khớp shape/schema/class distribution của HMEQ chuẩn, nhưng SHA-256 là `DFDBC2B7CDF728A15B53E323CDE6127995715DFA6B178BD3C1E3D9916D0367AA`, không phải checksum SAS kỳ vọng `AECB99E8E6B3CCF5F3C0F8EE189BBCD6B7B457FCCC5F8A61D8C9F1A0B27074CD`.

## Schema

| Cột | Vai trò | Kiểu kiểm tra | Ghi chú |
|---|---|---|---|
| `BAD` | Target | Numeric/binary | `1 = default/seriously delinquent`, `0 = current/non-default`. |
| `LOAN` | Input | Numeric | Requested loan amount. |
| `MORTDUE` | Input | Numeric | Amount due on existing mortgage. |
| `VALUE` | Input | Numeric | Current property value. |
| `REASON` | Input | Categorical | `DebtCon`, `HomeImp`, hoặc missing. |
| `JOB` | Input | Categorical | Occupational category hoặc missing. |
| `YOJ` | Input | Numeric | Years at present job. |
| `DEROG` | Input | Numeric | Number of major derogatory reports. |
| `DELINQ` | Input | Numeric | Number of delinquent credit lines. |
| `CLAGE` | Input | Numeric | Age of oldest credit line in months. |
| `NINQ` | Input | Numeric | Number of recent credit inquiries. |
| `CLNO` | Input | Numeric | Number of credit lines. |
| `DEBTINC` | Input | Numeric | Debt-to-income ratio. |

## Verification Summary

| File | SHA-256 | Shape | BAD=0 | BAD=1 | Default rate | Missing values | Duplicate rows | Core usable |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `data/raw/hmeq/hmeq.csv` | `E284FA61851D066FC3F6FE1E1024F07D1AF94A6830C0BC005F27B54398C6BBC1` | 604 x 13 | 375 | 229 | 0.3791390728 | 950 | 0 | No |
| `data/raw/hmeq/hmeq_full.csv` | `DFDBC2B7CDF728A15B53E323CDE6127995715DFA6B178BD3C1E3D9916D0367AA` | 5,960 x 13 | 4,771 | 1,189 | 0.1994966443 | 5,271 | 0 | Yes |

## Remediation Finding

`data/raw/hmeq/hmeq.csv` is not the paper-compatible HMEQ file. It has the same schema as the full HMEQ file but only 604 rows. After aligning column order and comparing against `hmeq_full.csv`, all 604 rows match the first 604 observations by all columns except the final row's `DEBTINC`: in the full file, row index 603 has `DEBTINC = 34.880462318`; in the 604-row file it is missing. The evidence supports treating the old file as a truncated/incomplete prefix of the full HMEQ dataset, not as a separate validated sample for core replication.

## License and Conditions

License remains `Chưa xác định`. SAS documentation describes the sample data, and the GitHub raw artifact is used only as a reproducible CSV source in this workspace. Any publication should verify redistribution rights before sharing raw data.

## Preprocessing Status

No imputation, WOE, VIF, class balancing, split creation, or duplicate removal has been applied in this remediation step.

Numeric columns are `LOAN`, `MORTDUE`, `VALUE`, `YOJ`, `DEROG`, `DELINQ`, `CLAGE`, `NINQ`, `CLNO`, `DEBTINC`. Categorical columns are `REASON`, `JOB`. There are no identifier or ignored columns in the active full file.
