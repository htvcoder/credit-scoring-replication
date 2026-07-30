"""Đọc, kiểm tra và khóa manifest P7A bằng biểu diễn JSON chính tắc."""

from __future__ import annotations

from copy import deepcopy
from math import comb
from math import ceil
from pathlib import Path
from typing import Any, Mapping

import yaml

from creditrep.config.loader import sha256_canonical
from creditrep.datasets.registry import find_repo_root, load_registry
from creditrep.models.registry import MODEL_REGISTRY


class ProtocolManifestError(ValueError):
    """Manifest P7A vi phạm schema hoặc invariant khoa học."""


DATASETS = ("AC", "GC", "HMEQ", "TH02", "TC", "GMC")
CORE_REQUIRED = {"logistic_regression", "decision_tree", "random_forest", "xgboost", "mlp_1", "mlp_3"}
STATUSES = {"required", "decision_pending", "conditionally_required", "optional_extension"}
ALLOWED_OVERRIDES = {"artifact_root", "device", "threads", "checkpoint_root", "log_level"}


def effective_min_samples_leaf(value: int | float, inner_training_rows: int) -> int:
    """Quy đổi fraction sklearn thành số lá tối thiểu có thể truy vết."""
    if isinstance(value, bool) or not isinstance(inner_training_rows, int) or inner_training_rows <= 0:
        raise ProtocolManifestError("min_samples_leaf: giá trị hoặc inner_training_rows không hợp lệ")
    if isinstance(value, int):
        if value <= 0:
            raise ProtocolManifestError("min_samples_leaf: integer phải dương")
        return value
    if not isinstance(value, float) or not 0 < value <= 0.5:
        raise ProtocolManifestError("min_samples_leaf: fraction phải thuộc (0, 0.5]")
    return ceil(value * inner_training_rows)


def canonical_manifest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Trả payload khóa; metadata lock không tự tham gia vào hash."""
    payload = deepcopy(dict(manifest))
    payload.pop("lock", None)
    return payload


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    return sha256_canonical(canonical_manifest_payload(manifest))


def _fail(field: str, message: str) -> None:
    raise ProtocolManifestError(f"{field}: {message}")


def validate_manifest(manifest: Any, *, repo_root: Path | None = None) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        _fail("manifest", "phải là mapping YAML")
    required = {"schema_version", "protocol", "datasets", "target", "cross_validation", "models", "metrics", "runtime_overrides", "lock"}
    missing = sorted(required - set(manifest))
    if missing:
        _fail("manifest", f"thiếu trường bắt buộc: {missing}")
    protocol = manifest["protocol"]
    if not isinstance(protocol, dict) or protocol.get("status") not in {"locked", "candidate"}:
        _fail("protocol.status", "phải là locked hoặc candidate")
    ids = manifest["datasets"]
    if not isinstance(ids, list) or tuple(ids) != DATASETS or len(set(ids)) != len(ids):
        _fail("datasets", f"phải là danh sách duy nhất theo thứ tự {list(DATASETS)}")
    root = repo_root or find_repo_root()
    registry = load_registry(repo_root=root)
    if set(item.lower() for item in ids) != set(registry):
        _fail("datasets", "không khớp dataset registry")
    target = manifest["target"]
    if target != {"class_0": "good/non-default", "class_1": "bad/default", "y_score": "P(class 1 = bad/default)"}:
        _fail("target", "phải khóa đúng ngữ nghĩa class 0/1 và y_score")
    cv = manifest["cross_validation"]
    repeats = cv.get("outer_repeats") if isinstance(cv, dict) else None
    expected = {"AC": 10, "GC": 10, "TH02": 10, "HMEQ": 5, "TC": 5, "GMC": 5}
    if not isinstance(cv, dict) or repeats != expected or cv.get("outer_splits") != 2 or cv.get("inner_splits") != 5 or cv.get("seed") != 42:
        _fail("cross_validation", "phải dùng Table 3: repeat theo dataset, 2 outer folds, 5 inner folds và seed 42")
    models = manifest["models"]
    if not isinstance(models, list):
        _fail("models", "phải là danh sách")
    model_ids = [item.get("id") for item in models if isinstance(item, dict)]
    if len(model_ids) != len(models) or len(set(model_ids)) != len(model_ids):
        _fail("models", "model id phải duy nhất")
    unknown = [item for item in model_ids if item not in {cap.model_id for cap in MODEL_REGISTRY.capabilities()} | {"catboost", "tabnet", "ft_transformer"}]
    if unknown:
        _fail("models", f"model ID không xác định: {unknown}")
    status_by_id = {item["id"]: item.get("inclusion_status") for item in models}
    if not CORE_REQUIRED <= set(status_by_id) or any(status_by_id[item] != "required" for item in CORE_REQUIRED):
        _fail("models", "thiếu hoặc sai trạng thái required của core models")
    required_statuses = {"mlp_5": "decision_pending", "catboost": "conditionally_required", "tabnet": "conditionally_required", "ft_transformer": "optional_extension"}
    for model_id, status in required_statuses.items():
        if status_by_id.get(model_id) != status:
            _fail(f"models.{model_id}", f"phải có inclusion_status={status}")
    if any(status not in STATUSES for status in status_by_id.values()):
        _fail("models", "có inclusion_status không hợp lệ")
    candidate_space = manifest.get("candidate_search_space")
    if not isinstance(candidate_space, dict) or not isinstance(candidate_space.get("cart_a"), dict):
        _fail("candidate_search_space.cart_a", "thiếu CART-A đã phê duyệt cho P7B")
    cart_a = candidate_space["cart_a"]
    depths = cart_a.get("max_depth")
    leaves = cart_a.get("min_samples_leaf")
    if depths != [3, 5, 7, 9] or leaves != [0.005, 0.01, 0.02]:
        _fail("candidate_search_space.cart_a", "phải là Grid 2 đã phê duyệt")
    if cart_a.get("representation") != "relative_fraction" or cart_a.get("candidate_count") != 12 or cart_a.get("formula") != "4*3":
        _fail("candidate_search_space.cart_a", "thiếu representation/formula/candidate_count Grid 2")
    pilot = manifest.get("pilot_budget")
    if not isinstance(pilot, dict) or pilot.get("total_inner_fits") != 60:
        _fail("pilot_budget", "phải khóa 60 inner fits cho CART pilot")
    if pilot.get("datasets") != ["AC", "HMEQ", "GMC"] or pilot.get("inner_folds") != 5:
        _fail("pilot_budget", "phải dùng AC, HMEQ, GMC và 5 inner folds")
    candidates = pilot.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 4:
        _fail("pilot_budget.candidates", "phải có đúng 4 candidate pilot")
    full_grid = {(depth, leaf) for depth in depths for leaf in leaves}
    pilot_grid = {(item.get("max_depth"), item.get("min_samples_leaf")) for item in candidates if isinstance(item, dict)}
    if len(pilot_grid) != 4 or not pilot_grid <= full_grid:
        _fail("pilot_budget.candidates", "phải duy nhất và thuộc full CART-A grid")
    workload = cart_a.get("full_theoretical_workload")
    if not isinstance(workload, dict) or workload.get("inner_candidate_evaluation_fits") != 5400 or workload.get("outer_selected_model_refits") != 90:
        _fail("candidate_search_space.cart_a.full_theoretical_workload", "phải phân biệt 5400 inner fits và 90 final refits")
    metrics = manifest["metrics"]
    if not isinstance(metrics, dict) or metrics.get("primary") != "roc_auc" or metrics.get("threshold") != 0.5:
        _fail("metrics", "phải khóa roc_auc và threshold 0.5")
    spaces = manifest.get("reference_search_spaces")
    if not isinstance(spaces, dict):
        _fail("reference_search_spaces", "phải khai báo provenance Table 2")
    for model_id, expected_count in (("random_forest", 30), ("xgboost", 108)):
        item = spaces.get(model_id)
        if not isinstance(item, dict) or item.get("declared_configurations") != expected_count:
            _fail(f"reference_search_spaces.{model_id}", f"phải khai báo {expected_count} cấu hình từ Table 2")
        parameters = item.get("parameters", {})
        if not isinstance(parameters, dict):
            _fail(f"reference_search_spaces.{model_id}.parameters", "phải là mapping")
        combinations = 1
        for values in parameters.values():
            if not isinstance(values, list):
                _fail(f"reference_search_spaces.{model_id}.parameters", "mỗi grid phải là list")
            combinations *= len(values)
        if combinations != expected_count:
            _fail(f"reference_search_spaces.{model_id}", f"tổ hợp grid={combinations}, khác {expected_count}")
    mlp = spaces.get("mlp")
    if not isinstance(mlp, dict) or not isinstance(mlp.get("shared"), dict):
        _fail("reference_search_spaces.mlp", "thiếu shared grid")
    shared = mlp["shared"]
    base = len(shared.get("dropout", [])) * len(shared.get("l2", []))
    for model_id, depth, expected_count in (("mlp_1", 1, 144), ("mlp_3", 3, 720), ("mlp_5", 5, 2016)):
        item = mlp.get(model_id)
        if not isinstance(item, dict) or item.get("declared_configurations") != expected_count:
            _fail(f"reference_search_spaces.mlp.{model_id}", f"phải khai báo {expected_count} cấu hình từ Table 2")
        units = len(shared.get("hidden_units", []))
        calculated = comb(units + depth - 1, depth) * base * len(item.get("learning_rate", []))
        if calculated != expected_count:
            _fail(f"reference_search_spaces.mlp.{model_id}", f"công thức non-increasing cho {calculated}, khác {expected_count}")
    overrides = manifest["runtime_overrides"]
    if not isinstance(overrides, dict) or set(overrides.get("allowed", [])) - ALLOWED_OVERRIDES:
        _fail("runtime_overrides", "chỉ cho phép override vận hành đã khai báo")
    return deepcopy(manifest)


def verify_manifest_lock(manifest: Mapping[str, Any]) -> None:
    lock = manifest.get("lock")
    if not isinstance(lock, dict) or lock.get("algorithm") != "sha256-canonical-json":
        _fail("lock", "thiếu thuật toán sha256-canonical-json")
    actual = manifest_hash(manifest)
    if lock.get("manifest_sha256") != actual:
        _fail("lock.manifest_sha256", f"không khớp; expected {actual}")


def load_manifest(path: str | Path, *, repo_root: Path | None = None, verify_lock: bool = True) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = (repo_root or find_repo_root()) / file_path
    try:
        payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProtocolManifestError(f"manifest: không đọc được {file_path}: {exc}") from exc
    manifest = validate_manifest(payload, repo_root=repo_root)
    if verify_lock:
        verify_manifest_lock(manifest)
    return manifest
