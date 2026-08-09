# P7C.4B.1d — Independent operational readiness review and B1 closeout

## Phạm vi, phương pháp và giới hạn

B1d review độc lập B1a–B1c bằng source inspection, full regression, validator/read-back artifact và đối chiếu decision record. Không chạy benchmark mới, canonical TC/GMC, GPU, P7C.4B.2, final search hay nested CV. Smoke engineering luôn **non-publishable**.

## Evidence inventory và traceability

| Contract | Evidence | Result |
| --- | --- | --- |
| Sequential/parallel cùng logical-fit set, timing boundary | B1c tests; `smoke-audit-*-v3` manifests/summaries | Pass |
| RSS process-tree concurrent, cap 2, process isolation | B1c telemetry/validator; worker PID assertions | Pass |
| Completion inversion và canonical summary | `test_parallel_completion_order_inversion_keeps_canonical_summary_order` | Pass |
| Failure isolation/false marker | `test_parallel_mixed_failure_preserves_success_and_has_no_run_marker` | Pass |
| Resume/retry/corruption/quarantine | B1b/B1c resume and corruption tests | Pass |
| CLI subprocess và Windows spawn | `test_cli_parallel_spawn_and_exit_paths` | Pass |
| Runtime ignore | `.gitignore` rule and `git check-ignore` | Pass |

Artifact read-back của `smoke-audit-seq-v3` và `smoke-audit-par-v3` đều validator PASS, 4/4 completed, marker hợp lệ và provenance có mặt. Summary giữ thứ tự logical canonical; parallel ghi tối đa 2 worker. Process-tree RSS là maximum của mỗi sample đồng thời: parent cộng descendant sống có PID duy nhất, không cộng các peak bất đồng thời.

## Compute evidence và giới hạn

Sequential/parallel có cùng workload digest `4e455c47be4ff6bd7f6f79f829a112bebb7877aef731634d08dbe87a115397e2`, cùng 4 logical fits, seed/partition/candidate/architecture. Wall clock lần lượt 20.609 s và 13.984 s; speedup 1.474×. Peak process-tree RSS lần lượt 515,325,952 B và 831,856,640 B; memory ratio 1.614×. Cả hai dưới `rss_hard_bytes=12,348,030,976` từ `p7c3_mlp_feasibility_plan.yaml`.

Các số trên chỉ là fixture engineering; không là result khoa học, không chọn backend/candidate và không thay thế benchmark canonical. Giá trị 2.96× cũ bị loại vì timing boundary không tương đương.

## Operator checklist và policy

Trước một execution được phê duyệt: working tree sạch; commit/interpreter/plan/workload digest resolve được; counts, runtime/RAM, disk và RSS threshold được báo; mode thuộc allowlist; `cpu_parallel_2` cap 2; artifact root mới và Git-ignore; resume compatibility pass; không GPU/canonical nếu chưa có approval; operator review và xác nhận go/no-go.

**GO**: `cpu_parallel_2` được chấp nhận như execution mode readiness. **READY FOR PLANNING**: P7C.4B.2 chỉ được lập plan/dry-run. **NOT AUTHORIZED TO EXECUTE**: canonical/GPU execution vẫn bị chặn đến khi DR-P7C-03/04, final manifest, workload/cost policy và human approval hoàn tất.

## Verdict

B1d và toàn P7C.4B.1 pass closeout. Rủi ro còn lại là approval policy, không phải harness readiness. P7C.4B.2 là Next cho planning, chưa được execution.
