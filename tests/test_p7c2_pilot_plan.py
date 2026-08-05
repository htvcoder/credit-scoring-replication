from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import pytest
import yaml
from creditrep.protocols.p7c2 import P7C2PlanError, load_pilot_plan, validate_pilot_plan

P = Path("configs/protocols/p7c/p7c2_rf_xgboost_pilot_plan.yaml")


def payload():
    return yaml.safe_load(P.read_text(encoding="utf8"))


def test_plan_happy_path():
    assert load_pilot_plan(P, repo_root=Path("."))["expected_fits"]["total"] == 60


@pytest.mark.parametrize(
    "change",
    [
        lambda x: x.__setitem__("execution_status", "completed"),
        lambda x: x["models"][0]["candidates"][1]["parameters"].__setitem__(
            "n_estimators", 999
        ),
        lambda x: x["models"][1]["candidates"].__setitem__(
            1, x["models"][1]["candidates"][0]
        ),
        lambda x: x.__setitem__("artifact_root", "C:/x"),
        lambda x: x["threading"].__setitem__("fits_parallelism", 2),
        lambda x: x["lock"].__setitem__("plan_sha256", "0" * 64),
    ],
)
def test_plan_rejects_mutations(change):
    x = deepcopy(payload())
    change(x)
    with pytest.raises(P7C2PlanError):
        validate_pilot_plan(x, repo_root=Path("."))
