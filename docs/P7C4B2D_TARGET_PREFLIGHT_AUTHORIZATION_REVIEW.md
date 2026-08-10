# P7C.4B.2d — Target Preflight & Authorization Review

Checkpoint này chỉ cung cấp Stage 0 tĩnh và review object. Nó không chạy training,
target canary hay full preflight; không tạo effective authorization và không làm
`execution_plan_eligible` thành `true`.

## Ba contract tách biệt

1. **Target environment evidence** là bằng chứng runtime: Git HEAD, plan digest,
   dependency fingerprint, checksum input, disk/output namespace, process-spawn
   probe, mode/worker/VM và runtime/cost envelope. Không chứa operator approval.
2. **Authorization proposal** là preview digest-bound cho đúng bốn task canary của
   một mode. Nó luôn có `authorization_effective: false`.
3. **Effective authorization** nằm ngoài scope checkpoint này. Runner P7C.4B.2c
   vẫn yêu cầu contract/flag authorization riêng; environment hợp lệ hoặc proposal
   hợp lệ không thể unlock runner.

## Stage 0 fail-closed

Stage 0 đối chiếu Git HEAD thực tế bằng `git rev-parse HEAD`, plan digest canonical,
và dependency fingerprint `sha256-path-nul-bytes-v1` trên `pyproject.toml`,
`requirements.txt` và `requirements-dev.txt`. Ba file này là toàn bộ source dependency
canonical hiện có: `pyproject.toml` khai báo build/runtime/optional dependencies,
`requirements.txt` pin runtime và `requirements-dev.txt` pin test; repository không có
constraints, Pipfile, Poetry hay uv lockfile. Với canary mỗi mode, nó đọc registry/checksum source of
truth để tính SHA-256 của đúng AC và GMC; placeholder, hash sai, thiếu hoặc dư
dataset đều bị chặn.

Disk policy `p7c4b2d-v1-5GiB-static-canary-output-and-quarantine-margin` yêu cầu
ít nhất 5 GiB free space: đây là operational floor được version hóa cho output,
quarantine và resume metadata, không phải scientific runtime estimate. Hằng số và
policy string nằm trong validation contract, phải xuất hiện trong evidence (`disk_policy`)
và proposal (`minimum_free_disk_bytes`, `disk_policy`), nên thay đổi policy không thể
âm thầm qua digest/validator. Stage 0 trả available/required/margin. Output phải là namespace cụ thể bên dưới
`artifacts/`, không được là repo root, artifacts root, home hay path traversal; một
target không rỗng bị báo collision, không bị xóa hay ghi đè.

Process-spawn được probe bằng một child Python `sys.exit(0)` có timeout 2 giây.
Probe không import model, đọc dataset hay chạy workload. GPU-less CPU target phải
ghi rõ `gpu_count: 0`, `gpu_vram_bytes: 0`, `gpu_model: "none"`.

`cpu_parallel_2` là hai worker trên một VM; số VM chỉ lấy từ evidence `vm_count`.
Two-VM cost chỉ được tính khi evidence xác nhận `vm_count: 2`.

## Canary và proposal

Mỗi mode chọn bất biến, đúng thứ tự: AC/light MLP-3 warmup, AC/light measured-0,
GMC/heavy MLP-3 warmup, GMC/heavy measured-0. Bốn task/mode này là engineering
validation, `scientific_projection_eligible: false`; canary không phải scientific
evidence và warmup không tham gia projection.

Proposal bind plan digest, environment digest, mode, bốn IDs, output namespace,
runtime/budget, price/currency, timestamp/expiry. Validator chặn task thiếu/dư/trùng,
sai mode, digest, budget, output hoặc expiry. Proposal không được biến thành approval.

## CLI an toàn và exit code

```powershell
# Thu thập evidence local; unknown operator input vẫn null và bị fail-closed.
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2d_cli collect-target-environment --mode cpu_parallel_1 --output-directory artifacts/p7c4b2d-target

# Validate/review static, không chạy workload.
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2d_cli inspect-target-requirements --environment <evidence.json>
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2d_cli review-plan --environment <evidence.json>
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2d_cli render-authorization-proposal --environment <evidence.json>
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2d_cli validate-authorization-proposal --environment <evidence.json> --proposal <proposal.json>
```

## Cung cấp metadata operator ngoài repository

`collect-target-environment` tự đo `vcpu_count` (`os.cpu_count()`), `ram_bytes`
(`psutil.virtual_memory().total`) và CPU model (`platform.processor()`); operator
không được override các giá trị này, Git commit, plan digest, dependency hash,
dataset hashes, output directory, execution mode hay process probe. Provider,
region, instance ID, disk type, network topology, VM count và price/budget envelope
không được suy đoán; chúng phải nằm trong file JSON **ngoài repository**.

Schema `--operator-metadata` yêu cầu đúng toàn bộ 12 field: `provider`, `region`,
`instance_id`, `disk_type`, `network_topology`, `vm_count`, `hourly_price`,
`currency`, `price_source`, `price_observed_at`, `maximum_runtime_hours`,
`maximum_monetary_budget`. Unknown field, thiếu field, `NaN`/`Infinity`, type/range
sai hoặc override canonical field đều bị fail-closed. Collector merge hợp lệ rồi tạo
lại `environment_digest`; không bao giờ sửa file evidence sau khi digest đã có.

Linux:

```bash
mkdir -p /secure/operator-input
cat > /secure/operator-input/p7c4b2d-metadata.json <<'JSON'
{"provider":"example","region":"example-region","instance_id":"operator-id","disk_type":"ssd","network_topology":"single_vm","vm_count":1,"hourly_price":0.25,"currency":"USD","price_source":"operator-invoice","price_observed_at":"2026-08-10T00:00:00Z","maximum_runtime_hours":4.0,"maximum_monetary_budget":1.0}
JSON
python -m creditrep.experiments.p7c4b2d_cli collect-target-environment --mode cpu_parallel_1 --output-directory artifacts/p7c4b2d-target --operator-metadata /secure/operator-input/p7c4b2d-metadata.json > /secure/operator-input/p7c4b2d-evidence.json
```

Windows PowerShell:

```powershell
$meta = 'C:\secure\operator-input\p7c4b2d-metadata.json'
.\.venv\Scripts\python.exe -m creditrep.experiments.p7c4b2d_cli collect-target-environment --mode cpu_parallel_1 --output-directory artifacts/p7c4b2d-target --operator-metadata $meta | Set-Content -Encoding utf8 C:\secure\operator-input\p7c4b2d-evidence.json
```

Exit code `3` from collection is expected: collection does not grant authorization.
Only after a separately stored evidence JSON passes `inspect-target-requirements`
and `review-plan` can it reach operator-review readiness; it remains non-effective
and cannot unlock the runner.

Exit code: `0` valid/ready-for-review; `2` invalid input/proposal; `3` review bị
chặn; `4` internal error. JSON stdout deterministic và chỉ có stable reason codes.

Cấm ở checkpoint này: `p7c4b2c_cli run`, `resume`, `--target-preflight-authorized`,
target canary, model training, hoặc coi proposal là authorization. Không ghi runtime
artifact, credential hay environment evidence thật vào Git.

Reason code tiêu biểu: `git_provenance_unknown`, `git_provenance_mismatch`,
`environment_lock_mismatch`, `dataset_input_missing`, `dataset_input_hash_mismatch`,
`insufficient_free_disk`, `unsafe_output_namespace`, `output_collision`,
`process_spawn_probe_timeout`, `worker_count_mismatch` và
`execution_mode_unsupported`.

Readiness hiện tại là **`READY_FOR_OPERATOR_ENVIRONMENT_COLLECTION`**. Nó không
phải `READY_FOR_CANARY_EXECUTION`, không phải effective authorization và không
cho phép chạy target canary.
