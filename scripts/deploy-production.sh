#!/usr/bin/env bash
set -euo pipefail

APP_NAME="credit-scoring"
DEPLOY_ROOT="${DEPLOY_ROOT:-/opt/credit-scoring-deploy}"
STATE_DIR="${STATE_DIR:-${DEPLOY_ROOT}/state}"
LOG_DIR="${LOG_DIR:-${DEPLOY_ROOT}/logs}"
COMPOSE_FILE="${COMPOSE_FILE:-${DEPLOY_ROOT}/docker-compose.prod.yml}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-credit-scoring-prod}"
PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-http://127.0.0.1}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-120}"
FORCE_POST_DEPLOY_FAILURE="${FORCE_POST_DEPLOY_FAILURE:-false}"
LOCK_DIR="${STATE_DIR}/deploy.lock"

timestamp() {
  date -u +"%Y%m%dT%H%M%SZ"
}

log() {
  printf '[%s] %s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" "$*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

validate_sha() {
  case "${1:-}" in
    (*[!0-9a-f]* | "" ) return 1 ;;
    (????????????????????????????????????????) return 0 ;;
    (*) return 1 ;;
  esac
}

validate_image() {
  case "${1:-}" in
    ghcr.io/*:* ) ;;
    * ) return 1 ;;
  esac

  case "$1" in
    *[!abcdefghijklmnopqrstuvwxyz0123456789./_:@-]* ) return 1 ;;
    *".."* | *"//"* | *" "* ) return 1 ;;
  esac
}

validate_boolean() {
  case "${1:-}" in
    true | false) return 0 ;;
    *) return 1 ;;
  esac
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

http_get() {
  curl -fsS --max-time 10 "$1"
}

compose() {
  WEBSITE_IMAGE="${WEBSITE_IMAGE}" docker compose \
    -p "${COMPOSE_PROJECT_NAME}" \
    -f "${COMPOSE_FILE}" \
    "$@"
}

write_state() {
  local target_file="$1"
  local image="$2"
  local sha="$3"
  local tmp_file="${target_file}.$$"

  {
    printf 'WEBSITE_IMAGE=%q\n' "${image}"
    printf 'DEPLOY_SHA=%q\n' "${sha}"
  } > "${tmp_file}"
  mv "${tmp_file}" "${target_file}"
}

read_state_value() {
  local file="$1"
  local key="$2"

  if [ ! -f "${file}" ]; then
    return 0
  fi

  awk -F= -v key="${key}" '$1 == key { print substr($0, length(key) + 2) }' "${file}" | tail -n 1 | sed "s/^'//; s/'$//"
}

wait_for_health() {
  local deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
  local health_url="${PUBLIC_BASE_URL%/}/health/"

  while [ "${SECONDS}" -lt "${deadline}" ]; do
    if http_get "${health_url}" | grep -q "OK"; then
      return 0
    fi
    sleep 3
  done

  return 1
}

verify_endpoint() {
  local url="$1"
  http_get "$url" >/dev/null
}

verify_version() {
  local version_url="${PUBLIC_BASE_URL%/}/version/"
  http_get "${version_url}" | grep -q "${DEPLOY_SHA}"
}

deploy_image() {
  local image="$1"

  export WEBSITE_IMAGE="${image}"
  docker pull "${WEBSITE_IMAGE}"
  compose up -d --remove-orphans
}

rollback_to_previous() {
  local previous_file="${STATE_DIR}/previous.env"
  local previous_image
  local previous_sha

  if [ ! -f "${previous_file}" ]; then
    log "No previous successful deployment recorded; cannot auto-rollback."
    return 1
  fi

  previous_image="$(read_state_value "${previous_file}" WEBSITE_IMAGE)"
  previous_sha="$(read_state_value "${previous_file}" DEPLOY_SHA)"

  if ! validate_image "${previous_image}" || ! validate_sha "${previous_sha}"; then
    log "Previous deployment state is invalid; cannot auto-rollback."
    return 1
  fi

  log "Rollback start: restoring previous good image ${previous_image}"
  WEBSITE_IMAGE="${previous_image}"
  DEPLOY_SHA="${previous_sha}"
  if ! deploy_image "${WEBSITE_IMAGE}"; then
    log "Rollback deploy command failed for restored image: ${previous_image}"
    return 1
  fi
  if ! wait_for_health; then
    log "Restored image health check failed: ${previous_image}"
    return 1
  fi
  log "Restored image health check passed: ${previous_image}"
  if ! verify_endpoint "${PUBLIC_BASE_URL%/}/"; then
    log "Restored root endpoint check failed: ${previous_image}"
    return 1
  fi
  if ! verify_endpoint "${PUBLIC_BASE_URL%/}/health/"; then
    log "Restored health endpoint check failed: ${previous_image}"
    return 1
  fi
  if ! verify_version; then
    log "Restored version endpoint check failed for ${previous_sha}."
    return 1
  fi
  write_state "${STATE_DIR}/current.env" "${WEBSITE_IMAGE}" "${DEPLOY_SHA}"
  log "Final rollback result: restored ${previous_image}"
}

main() {
  [ "$(id -u)" -ne 0 ] || fail "Do not run production deploy as root."
  [ -n "${WEBSITE_IMAGE:-}" ] || fail "WEBSITE_IMAGE is required."
  [ -n "${DEPLOY_SHA:-}" ] || fail "DEPLOY_SHA is required."
  validate_image "${WEBSITE_IMAGE}" || fail "WEBSITE_IMAGE must be a lowercase ghcr.io image with a tag."
  validate_sha "${DEPLOY_SHA}" || fail "DEPLOY_SHA must be a full 40-character lowercase Git commit SHA."
  validate_boolean "${FORCE_POST_DEPLOY_FAILURE}" || fail "FORCE_POST_DEPLOY_FAILURE must be true or false."

  require_command docker
  require_command curl
  require_command awk
  docker compose version >/dev/null
  [ -f "${COMPOSE_FILE}" ] || fail "Compose file not found: ${COMPOSE_FILE}"

  install -d -m 755 "${STATE_DIR}" "${LOG_DIR}"
  local log_file="${LOG_DIR}/deploy-$(timestamp).log"
  exec > >(tee -a "${log_file}") 2>&1

  if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    fail "Another deployment is already running: ${LOCK_DIR}"
  fi
  trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

  local current_file="${STATE_DIR}/current.env"
  local previous_file="${STATE_DIR}/previous.env"
  local current_image=""
  local current_sha=""

  current_image="$(read_state_value "${current_file}" WEBSITE_IMAGE)"
  current_sha="$(read_state_value "${current_file}" DEPLOY_SHA)"

  log "Starting deployment for candidate image: ${WEBSITE_IMAGE}"
  if [ -n "${current_image}" ] && [ "${current_image}" != "${WEBSITE_IMAGE}" ]; then
    validate_image "${current_image}" || fail "Current WEBSITE_IMAGE is invalid; refusing to update rollback state."
    validate_sha "${current_sha}" || fail "Current DEPLOY_SHA is invalid; refusing to update rollback state."
    write_state "${previous_file}" "${current_image}" "${current_sha}"
    log "Recorded previous good image: ${current_image}"
  elif [ -n "${current_image}" ]; then
    log "Current image already matches candidate: ${current_image}"
  else
    log "No current successful image recorded before this deployment."
  fi

  if ! deploy_image "${WEBSITE_IMAGE}"; then
    log "Deploy command failed."
    rollback_to_previous || fail "Deployment failed and rollback failed."
    fail "Deployment failed; rollback completed."
  fi

  if ! wait_for_health; then
    log "Health check timed out."
    compose logs --tail=100 || true
    rollback_to_previous || fail "Health check failed and rollback failed."
    fail "Health check failed; rollback completed."
  fi
  log "Health check passed for candidate image: ${WEBSITE_IMAGE}"

  if ! verify_endpoint "${PUBLIC_BASE_URL%/}/"; then
    log "Root endpoint check failed."
    rollback_to_previous || fail "Root endpoint failed and rollback failed."
    fail "Root endpoint failed; rollback completed."
  fi

  if ! verify_endpoint "${PUBLIC_BASE_URL%/}/health/"; then
    log "Health endpoint check failed."
    rollback_to_previous || fail "Health endpoint failed and rollback failed."
    fail "Health endpoint failed; rollback completed."
  fi

  if ! verify_version; then
    log "Version endpoint does not match ${DEPLOY_SHA}."
    rollback_to_previous || fail "Version mismatch and rollback failed."
    fail "Version mismatch; rollback completed."
  fi

  if [ "${FORCE_POST_DEPLOY_FAILURE}" = "true" ]; then
    log "Forced post-deploy failure requested for rollback verification."
    rollback_to_previous || fail "Forced post-deploy failure triggered and rollback failed."
    fail "Forced post-deploy failure triggered; rollback completed."
  fi

  write_state "${current_file}" "${WEBSITE_IMAGE}" "${DEPLOY_SHA}"
  log "Deployment succeeded: ${WEBSITE_IMAGE}"
  log "State directory: ${STATE_DIR}"
  log "Log file: ${log_file}"
}

main "$@"
