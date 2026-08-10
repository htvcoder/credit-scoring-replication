# P7C.4B.2d — Review plan target preflight và readiness authorization

P7C.4B.2d là decision package để operator review trước khi cấp quyền chạy target
preflight. Nó không chạy target/canary, không tạo effective authorization, không
nhập giá VM/GPU và không làm execution plan eligible.

## Plan và coverage

Plan P7C.4B.2c có ba MLP, ba proxy (`low`, `typical`, `high`) và hai CPU modes.
Vì vậy có **18 mode-qualified proxy representatives**; đây không phải 18 proxy
độc lập vì mode đã là một trục trong số đó. Sáu dataset tạo 108 strata. Một warmup
mỗi stratum tạo 108 task; hai measured repetitions tạo 216 task; tổng là 324,
bằng immutable upper bound. Warmup không tham gia projection.

Ba proxy đại diện cho đầu nhẹ, trung tâm và nặng của candidate complexity theo
hidden units/dropout/L2/learning rate. Chúng là proxy minh bạch, không phải
candidate được canonical search chọn. Hai repetition chỉ cung cấp feasibility
range empirical min/max; không đủ để khẳng định confidence interval, quantile
ổn định hay bootstrap inference. Sáu dataset cần được giữ trong full preflight để
tránh bỏ sót variation theo data size/features; canary chỉ là gate vận hành.

Mode 1/2 dùng cùng logical task semantics; mode 2 có hai worker process. Aggregate
work, wall-clock, startup/queue, idle và artifact overhead giữ taxonomy riêng;
không được cộng hai lần hoặc suy rộng bounded capacity loss theo global multiplier.

## Staged execution

Stage 0 là static validation: plan/Git/dataset hashes, preprocessing/proxy,
output namespace, disk, process spawn và target environment contract. Fail ở đây
không được chạy workload.

Stage 1 canary chọn deterministic subset immutable: AC/light và GMC/heavy, MLP-3,
mỗi case có warmup và measured task, cho từng mode được target hỗ trợ. Canary mang
`execution_stage: target_canary` và `scientific_projection_eligible: false`; nó
không tự trở thành full evidence.

Stage 2 chạy batch/stratum partial sau canary được review. Dừng khi vượt runtime,
memory/disk, failure-rate, timing invariant, artifact validation hoặc operator
budget. Stage 3 full preflight chỉ được xem xét sau authorization riêng, Stage 0/1
pass và input vận hành đầy đủ.

## Target environment và cost

Operator phải cung cấp provider/region/VM, OS/Python/CPU/vCPU/RAM, GPU/CUDA khi áp
dụng, disk/free disk/network, mode/workers, Git/environment-lock/dataset hashes,
output path, **vm_count** do operator xác nhận, budgets và authorization metadata. Unknown phải là `null`/missing,
không được thay bằng assumption.

Cost output có lower/central/upper wall-clock, hourly price, currency/source/time
và lower/central/upper cost. Không có price thì cost là `unknown`, không phải 0.
`cpu_parallel_2` nghĩa là hai worker process trên **một VM**; nó không suy ra hai VM.
Mode/VM count được tách riêng; two-VM billing chỉ áp dụng khi environment xác nhận
`vm_count: 2`, theo `VM count × wall-clock × price`,
không dùng aggregate work làm billed time.

## Proposal và commands

Authorization proposal là preview object có `artifact_type:
authorization_proposal` và `authorization_effective: false`. Nó bind plan/environment
digest nhưng không thể unlock runner. P7C.4B.2d không ghi proposal file; CLI chỉ
render stdout để operator review.

```powershell
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2d_cli review-plan
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2d_cli inspect-target-requirements --environment <operator-env.json>
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2d_cli render-authorization-proposal --environment <operator-env.json>
```

Readiness hiện tại là `NOT_READY_FOR_AUTHORIZATION`: chưa có target environment,
budget/price input, canary approval hay effective authorization. Controlled fixture
có thể đạt `READY_FOR_CANARY_AUTHORIZATION_REVIEW`; full target preflight vẫn cần
gate chặt hơn. Sau authorization ở một lượt sau, operator vẫn phải dùng runner
P7C.4B.2c để resume/quarantine/validate và chỉ dùng projection từ target artifacts
đã validator PASS.
