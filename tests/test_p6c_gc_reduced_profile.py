"""P6C.2A reduced GC profile must stay explicitly non-scientific."""

from __future__ import annotations

from pathlib import Path

import yaml

from creditrep.config.model_validation import parse_model_validation_config
from creditrep.models.neural.specifications import FAIR_BUDGET_ID


def test_gc_reduced_profile_is_canonical_non_publishable_and_fair():
    root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (root / "configs" / "p6c_gc_reduced_validation_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = parse_model_validation_config(payload)
    assert config.dataset_id == "GC"
    assert sorted(config.model_candidates) == ["mlp_1", "mlp_3", "mlp_5"]
    assert config.publishable is False
    assert config.validation_purpose == "engineering_reduced_validation"
    assert config.fair_budget_id == FAIR_BUDGET_ID
    assert (
        config.max_retry_attempts == 1
        and config.config_hash == parse_model_validation_config(payload).config_hash
    )
    budgets = {
        tuple(sorted(candidate.items()))
        for values in config.model_candidates.values()
        for candidate in values
    }
    assert len(budgets) == 1
