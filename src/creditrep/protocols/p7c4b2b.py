"""P7C.4B.2b deterministic preflight planning; never starts a fit."""
from __future__ import annotations
from copy import deepcopy
from statistics import mean, median, pstdev
from typing import Any
from creditrep.config.loader import sha256_canonical

SCIENTIFIC_DIGEST = "4d8636c3606e07e243efd2bc7be12806e7adf4fc1b19dbe0dc113a35adc57f75"
MODES = {"cpu_parallel_1": 1, "cpu_parallel_2": 2}

class PreflightError(ValueError): pass

def plan_digest(plan: dict[str, Any]) -> str:
    value = deepcopy(plan); value.pop("plan_digest", None); return sha256_canonical(value)

def build_plan(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest["lock"]["manifest_sha256"] != SCIENTIFIC_DIGEST: raise PreflightError("scientific_manifest_digest_mismatch")
    models = {item["id"]: item["candidates"] for item in manifest["models"]}
    # Fixed coverage: TC and GMC, each depth, lightweight/median/heaviest tuple.
    units=[]
    for dataset in ("TC", "GMC"):
        for model_id, candidates in models.items():
            for role, index in (("light", 0), ("median", len(candidates)//2), ("heavy", len(candidates)-1)):
                candidate=candidates[index]
                identity={"scientific_manifest_digest":SCIENTIFIC_DIGEST,"dataset_id":dataset,"model_id":model_id,"candidate_id":candidate["candidate_id"],"inner_fold":0,"seed_identity":1701,"coverage_role":role}
                units.append({**identity,"logical_fit_id":sha256_canonical(identity)})
    plan={"schema_version":1,"checkpoint":"P7C.4B.2b","evidence_scope":"engineering_compute_preflight_non_publishable","scientific_manifest_digest":SCIENTIFIC_DIGEST,"machine_role_required":"intended_single_vm_target","modes":MODES,"thread_policy":{"max_workers":2,"threads_per_worker":2,"nested_parallelism":"forbidden"},"limits":{"per_fit_timeout_seconds":1800,"retry_maximum":1,"global_wall_clock_seconds":43200,"minimum_free_disk_bytes":16106127360,"rss_hard_bytes":12348030976,"artifact_size_bytes":2147483648},"warmup_policy":{"separate":True,"per_mode":1},"measured_repetitions":2,"units":units,"runtime_cost_status":"pending_target_measurement_no_b1_fixture_input"}
    plan["plan_digest"]=plan_digest(plan); return plan

def validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("scientific_manifest_digest") != SCIENTIFIC_DIGEST or plan.get("plan_digest") != plan_digest(plan): raise PreflightError("plan_digest_mismatch")
    if plan.get("modes") != MODES or plan.get("thread_policy",{}).get("max_workers") != 2: raise PreflightError("worker_limit_mismatch")
    units=plan.get("units",[])
    if len(units)!=18 or len({x.get("logical_fit_id") for x in units})!=18: raise PreflightError("preflight_unit_coverage_mismatch")
    return {"valid":True,"units":18,"measured_fits_per_mode":36,"warmups_per_mode":18}

def validate_machine(profile: dict[str, Any]) -> dict[str, Any]:
    required={"machine_role","os","cpu_model","physical_cores","logical_cores","ram_total_bytes","disk_free_bytes","python_executable","python_version","dependency_fingerprint","git_commit","scientific_manifest_digest","dataset_fingerprints","worker_limit","utc_started"}
    if not required <= set(profile): raise PreflightError("machine_provenance_missing")
    if profile["machine_role"] != "intended_single_vm_target": raise PreflightError("target_machine_not_confirmed")
    if profile["scientific_manifest_digest"] != SCIENTIFIC_DIGEST: raise PreflightError("scientific_manifest_digest_mismatch")
    if profile["worker_limit"] != 2: raise PreflightError("worker_limit_mismatch")
    return {"valid":True}

def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured=[x for x in records if x.get("classification")=="measured"]
    if not measured: raise PreflightError("warmup_mixed_or_missing_measured")
    times=[x["wall_clock_seconds"] for x in measured]
    if any(not isinstance(x,(int,float)) or x<0 for x in times): raise PreflightError("invalid_timing")
    return {"measured_count":len(measured),"mean_seconds":mean(times),"median_seconds":median(times),"stddev_seconds":pstdev(times) if len(times)>1 else None,"percentiles":"insufficient_sample" if len(times)<20 else "available","warmups_excluded":len(records)-len(measured)}

def project(summary: dict[str, Any], *, price_per_hour: float|None=None) -> dict[str, Any]:
    # A scalar sample cannot represent the locked workload; refuse fake precision.
    result={"single_vm_parallel_1":{"status":"insufficient_stratified_coverage"},"single_vm_parallel_2":{"status":"insufficient_stratified_coverage"},"two_vm_cpu":{"status":"pending_multi_vm_overhead_evidence_not_authorized"},"gpu":{"status":"pending_gpu_preflight_not_authorized"},"cost":{"status":"pending_operator_price_input" if price_per_hour is None else "requires_valid_stratified_projection"}}
    return result
