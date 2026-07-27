"""Direct, isolated CLI contract tests for P5C."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from creditrep.datasets.models import LoadedDataset
from creditrep.preprocessing import ProtocolAConfig


def _cli_module():
    path = Path(__file__).parents[1] / "scripts" / "run_model_validation.py"
    spec = importlib.util.spec_from_file_location("p5c_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _dataset():
    return LoadedDataset("TOY", pd.DataFrame({"x": range(24)}), pd.Series([0, 1] * 12), {"numeric_columns": ["x"], "categorical_columns": [], "source_file": "fixture.csv", "row_count": 24, "feature_count": 1}, Path("fixture.csv"))


def _write_config(tmp_path):
    config = tmp_path / "fixture.yaml"
    config.write_text("\n".join(["experiment: {name: cli_fixture, publishable: false, result_scope: model_validation}", "dataset: {id: TOY}", "output: {root_dir: artifacts}", "models: {logistic_regression: [{max_iter: 30, solver: liblinear}]}", "metrics: [{id: roc_auc}]", "optimization_metric: roc_auc", "random_seed: 17", ""]), encoding="utf-8")
    return config


def _patch_runtime(monkeypatch, module, tmp_path):
    monkeypatch.setattr(module, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: _dataset())
    monkeypatch.setattr(module, "get_dataset_checksum", lambda *args, **kwargs: SimpleNamespace(actual_sha256="fixture"))
    monkeypatch.setattr(module, "load_protocol_a_config", lambda *args, **kwargs: ProtocolAConfig())
    monkeypatch.setattr(module, "resolve_repo_path", lambda value, **kwargs: tmp_path / value)


def test_cli_valid_fixture_run_returns_zero_and_writes_validation_artifacts(tmp_path, monkeypatch, capsys):
    module = _cli_module(); config = _write_config(tmp_path); _patch_runtime(monkeypatch, module, tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_model_validation.py", "--config", str(config)])
    assert module.main() == 0
    output = capsys.readouterr().out
    assert "scientific" not in output.lower() and '"completed_folds": 2' in output
    assert list((tmp_path / "artifacts").rglob("predictions.csv"))


def test_cli_invalid_config_returns_nonzero_without_completed_artifacts(tmp_path, monkeypatch, capsys):
    module = _cli_module(); config = tmp_path / "invalid.yaml"; config.write_text("experiment: {}", encoding="utf-8")
    _patch_runtime(monkeypatch, module, tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_model_validation.py", "--config", str(config)])
    assert module.main() == 2
    assert "Model-validation failed" in capsys.readouterr().err
    assert not list(tmp_path.rglob("complete.json"))


def test_cli_validate_only_does_not_train_or_write_fold_results(tmp_path, monkeypatch):
    module = _cli_module(); config = _write_config(tmp_path); _patch_runtime(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "run_folded_model_validation", lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not train")))
    monkeypatch.setattr(sys, "argv", ["run_model_validation.py", "--config", str(config), "--validate-only"])
    assert module.main() == 0
    assert not list(tmp_path.rglob("predictions.csv"))


def test_cli_resume_skips_completed_folds_without_rewrite(tmp_path, monkeypatch, capsys):
    module = _cli_module(); config = _write_config(tmp_path); _patch_runtime(monkeypatch, module, tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_model_validation.py", "--config", str(config)])
    assert module.main() == 0
    prediction = next((tmp_path / "artifacts").rglob("predictions.csv")); before = prediction.read_bytes()
    monkeypatch.setattr(sys, "argv", ["run_model_validation.py", "--config", str(config), "--resume"])
    assert module.main() == 0
    assert '"skipped_folds": 2' in capsys.readouterr().out
    assert prediction.read_bytes() == before


def test_cli_fail_fast_stops_after_first_failed_execution_unit(tmp_path, monkeypatch, capsys):
    import creditrep.experiments.model_validation as runner
    module = _cli_module(); config = _write_config(tmp_path); _patch_runtime(monkeypatch, module, tmp_path)
    monkeypatch.setattr(runner, "_run_fold", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("secret=SECRET_SENTINEL_7a2e")))
    monkeypatch.setattr(sys, "argv", ["run_model_validation.py", "--config", str(config), "--fail-fast"])
    assert module.main() == 2
    captured = capsys.readouterr()
    assert "SECRET_SENTINEL_7a2e" not in captured.out + captured.err
    assert len(list((tmp_path / "artifacts").rglob("failures/*.json"))) == 1
    assert not list((tmp_path / "artifacts").rglob("predictions.csv"))


def test_cli_without_fail_fast_continues_and_returns_nonzero_when_failures_remain(tmp_path, monkeypatch):
    import creditrep.experiments.model_validation as runner
    module = _cli_module(); config = _write_config(tmp_path); _patch_runtime(monkeypatch, module, tmp_path)
    original, calls = runner._run_fold, {"count": 0}
    def fail_once(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1: raise RuntimeError("first failure")
        return original(**kwargs)
    monkeypatch.setattr(runner, "_run_fold", fail_once)
    monkeypatch.setattr(sys, "argv", ["run_model_validation.py", "--config", str(config)])
    assert module.main() == 2
    assert calls["count"] == 2
    assert len(list((tmp_path / "artifacts").rglob("failures/*.json"))) == 1
    assert len(list((tmp_path / "artifacts").rglob("predictions.csv"))) == 1
