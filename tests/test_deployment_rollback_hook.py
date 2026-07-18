from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_SCRIPT = ROOT / "scripts" / "deploy-production.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "deploy-production.yml"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_deploy_hook_defaults_false_and_validates_boolean():
    script = _read(DEPLOY_SCRIPT)

    assert 'FORCE_POST_DEPLOY_FAILURE="${FORCE_POST_DEPLOY_FAILURE:-false}"' in script
    assert "validate_boolean" in script
    assert 'validate_boolean "${FORCE_POST_DEPLOY_FAILURE}"' in script
    assert "FORCE_POST_DEPLOY_FAILURE must be true or false" in script


def test_forced_failure_runs_after_candidate_validation_before_current_state_write():
    script = _read(DEPLOY_SCRIPT)

    health_log = script.index("Health check passed for candidate image")
    version_check = script.index("if ! verify_version; then", health_log)
    forced_marker = script.index("Forced post-deploy failure requested for rollback verification.")
    rollback_call = script.index('rollback_to_previous || fail "Forced post-deploy failure')
    current_write = script.index('write_state "${current_file}" "${WEBSITE_IMAGE}" "${DEPLOY_SHA}"', forced_marker)

    assert health_log < version_check < forced_marker < rollback_call < current_write


def test_forced_failure_does_not_write_candidate_as_successful_current_state():
    script = _read(DEPLOY_SCRIPT)
    forced_block_start = script.index('if [ "${FORCE_POST_DEPLOY_FAILURE}" = "true" ]; then')
    forced_block_end = script.index('\n  fi\n\n  write_state "${current_file}"', forced_block_start)
    forced_block = script[forced_block_start:forced_block_end]

    assert "rollback_to_previous" in forced_block
    assert "fail " in forced_block
    assert 'write_state "${current_file}"' not in forced_block
    assert "Deployment succeeded" not in forced_block


def test_rollback_logs_restored_image_health_and_final_result():
    script = _read(DEPLOY_SCRIPT)

    assert "Rollback start: restoring previous good image" in script
    assert "Restored image health check passed" in script
    assert "Final rollback result: restored" in script


def test_workflow_dispatch_boolean_input_defaults_false():
    workflow = yaml.safe_load(_read(WORKFLOW))
    dispatch = workflow[True]["workflow_dispatch"]
    forced_input = dispatch["inputs"]["force_post_deploy_failure"]

    assert forced_input["type"] == "boolean"
    assert forced_input["default"] is False
    assert forced_input["required"] is False


def test_main_push_cannot_enable_forced_failure_mode():
    workflow_text = _read(WORKFLOW)

    assert 'force_post_deploy_failure="false"' in workflow_text
    assert 'if [ "${GITHUB_EVENT_NAME}" = "workflow_dispatch" ]; then' in workflow_text
    assert 'force_post_deploy_failure="${FORCE_POST_DEPLOY_FAILURE_INPUT:-false}"' in workflow_text


def test_workflow_passes_validated_boolean_over_ssh_without_secret():
    workflow_text = _read(WORKFLOW)

    assert "FORCE_POST_DEPLOY_FAILURE: ${{ steps.resolve.outputs.force_post_deploy_failure }}" in workflow_text
    assert "FORCE_POST_DEPLOY_FAILURE='${FORCE_POST_DEPLOY_FAILURE}'" in workflow_text
    assert "secrets.force_post_deploy_failure" not in workflow_text.lower()
    assert "secrets.FORCE_POST_DEPLOY_FAILURE" not in workflow_text
