"""Lazy PyTorch runtime, device, and reproducibility utilities."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from creditrep.models.neural.exceptions import DevicePolicyError, NeuralDependencyError

DevicePolicy = Literal["auto", "cpu", "cuda"]


def require_torch():
    """Import PyTorch only when neural functionality is actually requested."""
    try:
        import torch
    except ImportError as exc:
        raise NeuralDependencyError(
            "PyTorch is required for neural models. Install with `python -m pip install -e \".[neural]\"`."
        ) from exc
    return torch


@dataclass(frozen=True)
class RuntimeMetadata:
    requested_device: str
    resolved_device: str
    random_seed: int
    deterministic_requested: bool
    deterministic_enabled: bool
    deterministic_limitations: tuple[str, ...]
    cuda_available: bool
    cuda_device_count: int
    framework_name: str
    framework_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_device(policy: DevicePolicy | str):
    torch = require_torch()
    if policy not in {"auto", "cpu", "cuda"}:
        raise DevicePolicyError(f"Unsupported device policy: {policy!r}. Expected 'auto', 'cpu', or 'cuda'.")
    cuda_available = bool(torch.cuda.is_available())
    if policy == "cuda" and not cuda_available:
        raise DevicePolicyError("CUDA was explicitly requested but is not available.")
    resolved = "cuda" if policy == "cuda" or (policy == "auto" and cuda_available) else "cpu"
    return torch.device(resolved)


def configure_runtime(*, seed: int, device_policy: DevicePolicy | str, deterministic: bool) -> RuntimeMetadata:
    """Set seeds at training time, not module import time.

    PyTorch may still be affected by hardware, kernels, and library versions; metadata
    records that limitation instead of claiming universal bitwise reproducibility.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise DevicePolicyError("random seed must be an integer.")
    torch = require_torch()
    device = resolve_device(device_policy)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = bool(torch.cuda.is_available())
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    enabled = False
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            enabled = True
        except RuntimeError:
            enabled = False
    else:
        torch.use_deterministic_algorithms(False)
    limitations = (
        "Determinism is best-effort within the installed PyTorch/runtime stack; "
        "bitwise-identical results are not guaranteed across operating systems or hardware.",
    )
    return RuntimeMetadata(str(device_policy), str(device), seed, deterministic, enabled, limitations,
                           cuda_available, int(torch.cuda.device_count()) if cuda_available else 0,
                           "PyTorch", str(torch.__version__))
