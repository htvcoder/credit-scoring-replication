#!/usr/bin/env bash
set -euo pipefail

DEPLOY_ROOT="${DEPLOY_ROOT:-/opt/credit-scoring-deploy}"
STATE_DIR="${STATE_DIR:-${DEPLOY_ROOT}/state}"
LOG_DIR="${LOG_DIR:-${DEPLOY_ROOT}/logs}"
COMPOSE_FILE="${COMPOSE_FILE:-${DEPLOY_ROOT}/docker-compose.prod.yml}"
COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-credit-scoring-prod}"
PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-http://127.0.0.1}"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-120}"
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

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

read_state_value() {
  local file="$1"
  local key="$2"

  awk -F= -v key="${key}" '$1 == key { print substr($0, length(key) + 2) }' "${file}" | tail -n 1 | sed "s/^'//; s/'$//"
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

http_get() {
  curl -fsS --max-time 10 "$1"
}

compose() {
  WEBSITE_IMAGE="${WEBSITE_IMAGE}" docker compose \
    -p "${COMPOSE_PROJECT_NAME}" \
    -f "${COMPOSE_FILE}" \
    "$@"
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

main() {
  [ "$(id -u)" -ne 0 ] || fail "Do not run production rollback as root."
  require_command docker
  require_command curl
  docker compose version >/dev/null
  [ -f "${COMPOSE_FILE}" ] || fail "Compose file not found: ${COMPOSE_FILE}"
  [ -f "${STATE_DIR}/previous.env" ] || fail "No previous successful image recorded."

  install -d -m 755 "${STATE_DIR}" "${LOG_DIR}"
  local log_file="${LOG_DIR}/rollback-$(timestamp).log"
  exec > >(tee -a "${log_file}") 2>&1

  if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    fail "Another deployment is already running: ${LOCK_DIR}"
  fi
  trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

  WEBSITE_IMAGE="$(read_state_value "${STATE_DIR}/previous.env" WEBSITE_IMAGE)"
  DEPLOY_SHA="$(read_state_value "${STATE_DIR}/previous.env" DEPLOY_SHA)"
  validate_image "${WEBSITE_IMAGE}" || fail "Previous WEBSITE_IMAGE is invalid."
  validate_sha "${DEPLOY_SHA}" || fail "Previous DEPLOY_SHA is invalid."
  export WEBSITE_IMAGE DEPLOY_SHA

  log "Rolling back to ${WEBSITE_IMAGE}"
  docker pull "${WEBSITE_IMAGE}"
  compose up -d --remove-orphans
  wait_for_health || fail "Rollback health check failed."
  http_get "${PUBLIC_BASE_URL%/}/" >/dev/null || fail "Rollback root endpoint check failed."
  http_get "${PUBLIC_BASE_URL%/}/health/" >/dev/null || fail "Rollback health endpoint check failed."
  http_get "${PUBLIC_BASE_URL%/}/version/" | grep -q "${DEPLOY_SHA}" || fail "Rollback version check failed."
  write_state "${STATE_DIR}/current.env" "${WEBSITE_IMAGE}" "${DEPLOY_SHA}"
  log "Rollback succeeded: ${WEBSITE_IMAGE}"
  log "Log file: ${log_file}"
}

main "$@"
