"""P6C.2B TC checkpoint profile remains bounded and non-scientific."""

from __future__ import annotations

from pathlib import Path

import yaml

from creditrep.config.model_validation import parse_model_validation_config
from creditrep.models.neural.specifications import FAIR_BUDGET_ID


def test_tc_checkpoint_profile_is_non_publishable_and_uses_one_fair_budget():
    payload = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "p6c_tc_reduced_checkpoint_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    config = parse_model_validation_config(payload)
    assert config.dataset_id == "TC"
    assert sorted(config.model_candidates) == ["mlp_1", "mlp_3", "mlp_5"]
    assert config.publishable is False
    assert config.validation_purpose == "engineering_resource_checkpoint"
    assert config.fair_budget_id == FAIR_BUDGET_ID
    assert config.max_retry_attempts == 1
    assert {
        candidate["max_epochs"]
        for values in config.model_candidates.values()
        for candidate in values
    } == {2}
