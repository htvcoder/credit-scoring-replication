# P7C.4B.2d — Target Preflight & Authorization Review

Checkpoint này cung cấp Stage 0, proposal và command tạo/validate effective
authorization artifact. Nó không tự chạy training, target canary hay full
preflight và không làm `execution_plan_eligible` thành `true`. Chưa có effective
authorization thật nào được tạo và target canary vẫn chưa chạy.

## Ba contract tách biệt

1. **Target environment evidence** là bằng chứng runtime: Git HEAD, plan digest,
   dependency fingerprint, checksum input, disk/output namespace, process-spawn
   probe, mode/worker/VM và runtime/cost envelope. Không chứa operator approval.
2. **Authorization proposal** là preview digest-bound cho đúng bốn task canary của
   một mode. Nó luôn có `authorization_effective: false`.
3. **Effective authorization** là artifact riêng, chỉ được tạo sau explicit
   operator ceremony. Runner cần đồng thời environment, proposal và effective
   authorization; environment hay proposal riêng lẻ không thể unlock runner.

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

## Effective authorization

Effective authorization là artifact riêng (`artifact_type: effective_authorization`,
schema 1), khác proposal và có `authorization_effective: true`. Chỉ command tạo
artifact mới nhận explicit `operator_identity`, approval phrase
`APPROVE_P7C4B2D_TARGET_CANARY`, và `expires_at` hữu hạn; validity tối đa 24 giờ.
Scope được derive từ proposal/environment đã validate: proposal/environment/plan
digest, Git, stage/mode, bốn task, output, VM, runtime/budget/price/currency và
disk policy. Không có input scope tùy ý.

`authorization_digest` là deterministic integrity digest canonical, không phải
cryptographic signature: người có thể sửa artifact và tự tính lại digest vẫn không
bị chặn bởi digest đơn thuần. Boundary hiện tại dựa trên explicit operator ceremony
và review artifact; signing/KMS không nằm trong checkpoint này và không có secret
trong repository. Validator chặn expiry, tampering không redigest, proposal/env/scope
mismatch; runner phải validate lại trước run và resume.

External revalidation binds proposal/environment/plan/Git/stage/mode/tasks/output
and resource scope. `operator_identity`, the approval text and authorization
timestamps have no cryptographic external attestation: after recomputing the
checksum, changes to those checksum-only fields are detected only by their
semantic rules and, on resume, by the persisted original authorization digest.
The implementation is therefore not tamper-proof and does not authenticate an
operator.

Monetary control is a documented static upper bound, not live cloud billing: at
creation and validation, `maximum_monetary_budget` must cover
`hourly_price * vm_count * maximum_runtime_hours`. The runner establishes a
monotonic deadline from the minimum of time remaining to expiry, maximum runtime,
and persisted remaining runtime. It refuses dispatch after that deadline and
checks UTC expiry, accumulated runtime and live disk before each task. This permits no target start
whose full authorized time window is unaffordable; it does not claim to meter
or reconcile a provider invoice in real time.

`hourly_price` and `maximum_monetary_budget` are canonical positive decimal
strings. Cost comparison uses `Decimal`, so `0.26092 * 1 * 12 = 3.13104 USD`
is exact rather than a binary-float boundary comparison.

## Runner binding và resume provenance

Production dispatch uses a closed mapping: `synthetic_validation` selects only
the synthetic adapter and `target_preflight` selects only the canonical adapter.
Public `run`/`resume` reject arbitrary workload callables. Before creating output,
the target runner resolves the requested path beneath the repository `artifacts/`
root and requires exact equality with the authorized normalized path. Traversal
and symlink escapes are rejected. A parent-directory symlink race cannot be made
fully atomic with portable `pathlib`; the runner therefore resolves before mkdir
and verifies the created directory again before writing its manifests.
The directory-symlink regression remains a required Linux pre-commit check when
Windows cannot create the fixture and reports it skipped.

Target manifests persist authorization/proposal/environment/plan digests, Git,
created/expiry timestamps, stage/mode/tasks, normalized output, VM/worker/task
limits, runtime and monetary scope. `authorization_runtime.json` records original
runtime start and accumulated elapsed time. Resume requires the original
authorization digest and exact provenance; legacy target manifests fail closed.
Runtime accounting uses a conservative wall-clock envelope from the original
start, so crash/downtime is counted and resume cannot reset budget. Running work
is not hard-killed at expiry, but no later task is dispatched.

On every target run and resume, the runner rejects self-consistency as proof of
canonicality. It validates exact closed-world schemas for the full plan, every
task and every nested candidate, then independently reloads the locked P7C.4B.2a
manifest and its P7A source and deterministically rebuilds the P7C.4B.2c plan.
The submitted or persisted plan must be canonical-JSON identical to that rebuild.
Unknown, missing, mistyped, non-finite, out-of-range or reordered values fail
closed before cleanup, runtime mutation, executor construction or submit.

The target runner resolves ordered authorized IDs only from the rebuilt plan's
task objects. Submitted `plan.json` and `manifest.expected_tasks` are persisted
representations used for exact comparison and are never dispatch sources.
Provenance also records the rebuilt canonical task-set digest. Self-consistent
sample, plan, manifest and task-set digests prove only consistency of the current
artifacts, not canonicality or operator identity.

The locked inputs remain unsigned local files. The rebuild prevents an isolated
plan edit followed only by recomputation of local digests, but it is not
cryptographic authenticity against an attacker able to rewrite all locked inputs,
authorization and provenance artifacts together. Checksums are not signatures.

Stage 0 additionally creates `locked_runtime_inputs_digest` from a strict
semantic projection of every mutable repository input consumed by the four
target tasks: all typed fields of `configs/protocols/protocol_a.yaml`; AC/GMC
loading, target, feature-role, missing-value and consumed reader semantics from
`data/datasets.yaml`; and the unique selected rows in
`data/checksums-sha256.csv`, which must match the AC/GMC raw bytes. Unused dataset
entries, source descriptions and documentation are outside the projection
because the workload does not read them. Missing, malformed, duplicate or
inconsistent authoritative content fails closed. Protocol A and the selected
dataset registry are loaded with duplicate-key rejection at every mapping level;
typed fields reject application-level string coercion, booleans in numeric
positions and NaN/Infinity where finite numbers are required.

The canonical plan declares this semantic binding policy; the content digest is
bound through environment digest, proposal, effective authorization and resume
provenance. Run/resume recompute it before any side effect or dispatch. A target
worker recomputes it after process handoff and before attempt-directory creation,
preprocessing or training, then uses the validated typed config/registry objects
instead of reloading unchecked YAML. Each dataset load reads the raw source path
once into an immutable byte snapshot, checks SHA-256 over that snapshot before
parsing, and passes a `BytesIO` over the same bytes to pandas; parsing never reopens
the path or silently falls back to path-based input. This establishes identity of
the checked and parsed bytes, but does not claim that the filesystem read itself is
atomic. It detects localized uncommitted working-tree changes that an unchanged
Git HEAD cannot reveal. Digests and checksums remain integrity checks, not
signatures.

Runtime state schema 2 cross-binds run ID, authorization/proposal/environment/
plan digests, normalized output, runtime limit and immutable start. A mirrored
manifest checkpoint records generation, elapsed runtime, last-accounted timestamp
and state digest. Future timestamps, clock rollback beyond the documented five
second tolerance, lifecycle-inconsistent timestamps and state rollback fail
closed before cleanup or dispatch. This gives cross-file consistency and
localized-corruption detection only; it is not cryptographic authenticity against
an attacker able to rewrite every artifact and recompute checksums.

Environment `free_disk_bytes` remains historical evidence. Immediately before
output mutation and before each target dispatch, the runner calls live
filesystem disk usage for the canonical output filesystem and enforces the
versioned minimum.

Environment, proposal and authorization schemas are closed-world and validate
artifact type, schema version, checkpoint, required fields and primitive types.
Both CLIs share strict JSON loading that rejects duplicate keys at any nesting,
NaN/Infinity, malformed input and non-object top levels with stable reason codes.

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

Sau review/commit/đồng bộ patch, operator thực hiện ceremony riêng ngoài repository:

```bash
python -m creditrep.experiments.p7c4b2d_cli create-effective-authorization \
  --environment <environment.json> --proposal <proposal.json> \
  --operator-identity '<operator supplied identity>' \
  --operator-approval APPROVE_P7C4B2D_TARGET_CANARY \
  --expires-at <timezone-aware-ISO-8601> > <external-authorization.json>
python -m creditrep.experiments.p7c4b2d_cli validate-effective-authorization \
  --environment <environment.json> --proposal <proposal.json> \
  --authorization <external-authorization.json>
```

The spelling of the approval phrase is deliberately exact in code:
`APPROVE_P7C4B2D_TARGET_CANARY`. The command examples are placeholders only and
are not authorization to create a real target artifact in this implementation turn.

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
{"provider":"example","region":"example-region","instance_id":"operator-id","disk_type":"ssd","network_topology":"single_vm","vm_count":1,"hourly_price":"0.25","currency":"USD","price_source":"operator-invoice","price_observed_at":"2026-08-10T00:00:00Z","maximum_runtime_hours":4.0,"maximum_monetary_budget":"1.0"}
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

Cấm trong implementation/remediation này: chạy target `p7c4b2c_cli run`/`resume`,
`--target-preflight-authorized`, target canary, model training, hoặc coi proposal
là authorization. Không ghi runtime artifact, credential hay environment evidence
thật vào Git.

Reason code tiêu biểu: `git_provenance_unknown`, `git_provenance_mismatch`,
`environment_lock_mismatch`, `dataset_input_missing`, `dataset_input_hash_mismatch`,
`locked_runtime_input_mismatch`,
`insufficient_free_disk`, `unsafe_output_namespace`, `output_collision`,
`process_spawn_probe_timeout`, `worker_count_mismatch` và
`execution_mode_unsupported`.

Readiness hiện tại là **`READY_FOR_OPERATOR_ENVIRONMENT_COLLECTION`**. Nó không
phải `READY_FOR_CANARY_EXECUTION`, không phải effective authorization và không
cho phép chạy target canary.
