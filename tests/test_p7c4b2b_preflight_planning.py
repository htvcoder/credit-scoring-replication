import pytest
from creditrep.protocols.p7c4b2a import load_manifest
from creditrep.protocols.p7c4b2b import PreflightError, build_plan, validate_plan, summarize, project

def test_plan_is_deterministic_bounded_and_manifest_bound():
    m=load_manifest('configs/protocols/p7c/p7c4b2a_mlp_scientific_manifest.yaml'); a=build_plan(m); b=build_plan(m)
    assert a==b and validate_plan(a)["measured_fits_per_mode"]==36
    assert [x["model_id"] for x in a["units"]].count("mlp_5")==6

def test_summary_excludes_warmups_and_projection_refuses_fake_precision():
    result=summarize([{"classification":"warmup","wall_clock_seconds":1},{"classification":"measured","wall_clock_seconds":2},{"classification":"measured","wall_clock_seconds":4}])
    assert result["measured_count"]==2 and result["warmups_excluded"]==1 and project(result)["gpu"]["status"].startswith("pending")

def test_digest_and_worker_mutation_fail_closed():
    m=load_manifest('configs/protocols/p7c/p7c4b2a_mlp_scientific_manifest.yaml'); p=build_plan(m); p["modes"]["cpu_parallel_2"]=3
    with pytest.raises(PreflightError): validate_plan(p)
