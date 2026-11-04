#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# agent-canvas all-in-one entrypoint
#
# Starts three services (plus an optional fourth):
#   1. Agent Server   on port $AGENT_SERVER_PORT  (default 18000)
#   2. Automation     on port $AUTOMATION_PORT     (default 18001)
#   3. Static server  on port $PORT               (default 8000)
#      Routes /api/automation/* → automation, /api/* → agent-server,
#      and serves the frontend static build for everything else.
#   4. (Optional) Public-mode static server on $PUBLIC_MODE_PORT
#      Same frontend, but with --auth-required (no baked session key).
#      Used by auth-mode E2E tests. Only started when PUBLIC_MODE_PORT is set.
#
# Environment variables:
#   PORT                 – Unified entry point port (default: 8000)
#   AGENT_SERVER_PORT    – Internal agent-server port (default: 18000)
#   AUTOMATION_PORT      – Internal automation port (default: 18001)
#   AGENT_CANVAS_BASE_PATH – Static frontend mount path (default: /canvas)
#   VSCODE_PORT          – Internal editor port (default: 8001). The image does
#                          not EXPOSE it and the editor is reached through
#                          VSCODE_BASE_PATH on $PORT, but openvscode-server
#                          binds 0.0.0.0, so `docker run --network host` does
#                          leave it directly reachable with only its connection
#                          token in front of it.
#   VSCODE_BASE_PATH     – Path prefix the editor is served under on $PORT
#                          (default: /vscode). Exported to agent-server as
#                          OH_VSCODE_BASE_PATH and routed by the static server.
#                          agent-server's own OH_VSCODE_PORT / OH_VSCODE_BASE_PATH
#                          take precedence over these aliases; whichever is set,
#                          one effective pair drives both the editor process and
#                          the proxy route.
#   PUBLIC_MODE_PORT     – If set, starts a second static server on this port
#                          with --auth-required (no session key injected)
#   MARS_WEB             – Set to exactly 1 to opt in to the experimental
#                          DigitalOcean web bridge on $PORT (off by default).
#                          The public-mode server never hosts it.
#   MARS_WEB_KEY         – Key the bridge's /mars/* routes require (defaults to
#                          the session API key)
#   MARS_TOKEN           – DigitalOcean personal access token to sign the bridge
#                          in with at start (needs MARS_WEB=1)
#   AGENT_CANVAS_ALLOW_LAN_SESSION_KEY – Set to true only when the published
#                          host port is restricted to loopback and you accept
#                          embedding the session key in the served HTML
#   OH_SECRET_KEY        – Secret key for settings encryption (auto-generated
#                          and persisted if not provided)
#   OPENHANDS_AUTOMATION_API_KEY – Override automation backend auth key
#                          (defaults to session API key — both backends
#                          use the same `X-Session-API-Key` header)
#   AUTOMATION_AGENT_SERVER_URL  – URL the automation service uses to reach the
#                          agent-server (default: http://127.0.0.1:AGENT_SERVER_PORT).
#                          Setting this enables local-mode auth so the session
#                          API key is validated internally instead of against the
#                          OpenHands cloud API.
#   AUTOMATION_KV_SECRET  – Encrypts automation KV state and signs scoped KV
#                          tokens. Unless set, it is pinned in
#                          automation-kv-secret.txt on first start (seeded with
#                          the session API key) and kept from then on, so
#                          rotating the session key does not change it.
#   FILE_STORE             – Storage backend for automation tarballs (default: local).
#                          Without this the automation backend may fall back to
#                          S3/GCS which fails without cloud credentials.
#   LOCAL_STORAGE_PATH     – Directory for local file storage (default: ~/.openhands/storage)
#   AUTOMATION_BASE_URL    – Publicly-reachable base URL for the automation
#                          service, used in callback URLs and injected into
#                          sandboxes (default: http://127.0.0.1:$PORT).
#                          Override in production when the external URL differs.
#   AUTOMATION_WORKSPACE_BASE – Directory for automation run workspaces
#                          (default: ~/.openhands/workspaces)
#   AGENT_CANVAS_MAX_FILE_MB  – Maximum size (MB) of a single chat
#                          attachment, enforced by the frontend before it is
#                          uploaded (default: 3 from config/defaults.json).
#   AGENT_CANVAS_MAX_TOTAL_MB – Maximum combined size (MB) of all chat
#                          attachments in one message (default: 3).
#                          Both are injected into the served frontend as
#                          window.__AGENT_CANVAS_MAX_FILE_SIZE__ /
#                          __AGENT_CANVAS_MAX_TOTAL_SIZE__ (bytes), so a
#                          pre-built bundle can be reconfigured without
#                          rebuilding.
#   Any agent-server or automation env vars are passed through.
# ═══════════════════════════════════════════════════════════════════════════════
set -uo pipefail

log() { printf '[agent-canvas] %s\n' "$*"; }
log_error() { printf '[agent-canvas] ERROR: %s\n' "$*" >&2; }

# ── Load centralized defaults (generated from config/defaults.json at build) ─
# shellcheck source=/dev/null
if [ -f /opt/agent-canvas/defaults.env ]; then
  # shellcheck disable=SC1091
  . /opt/agent-canvas/defaults.env
fi

PORT="${PORT:-${CONFIG_PROXY_PORT:-8000}}"
AGENT_SERVER_PORT="${AGENT_SERVER_PORT:-${CONFIG_AGENT_SERVER_PORT:-18000}}"
AUTOMATION_PORT="${AUTOMATION_PORT:-${CONFIG_AUTOMATION_PORT:-18001}}"

# The bundled editor is reached through a path prefix on the proxy port rather
# than a published port of its own. The same prefix has to reach agent-server
# (it launches openvscode-server with --server-base-path and advertises the
# prefix from /api/vscode/url) and the static-server route table below, or the
# advertised URL and the route serving it disagree.
#
# Two env var names reach the same setting: OH_VSCODE_PORT / OH_VSCODE_BASE_PATH
# are agent-server's own documented variables, which a deployment may already
# set and which this entrypoint passes through like any other OH_* var, while
# VSCODE_PORT / VSCODE_BASE_PATH are this image's aliases. They collapse to one
# effective pair here, before anything reads them — resolving them
# independently would let `OH_VSCODE_BASE_PATH=/editor` move the editor without
# moving the route, leaving the button pointing at a path the proxy never
# serves.
# >>> vscode-config: this block is extracted and executed by
# >>> __tests__/scripts/docker-vscode-route-sync.test.ts — keep the markers.
# The canvas mount is resolved here rather than alongside the ports above
# because the collision guard below compares the two prefixes: keeping both
# inside the extracted block is what lets that comparison be tested against the
# real defaults instead of only against values a test injects.
AGENT_CANVAS_BASE_PATH="${AGENT_CANVAS_BASE_PATH:-${CONFIG_CANVAS_BASE_PATH:-/canvas}}"
VSCODE_PORT="${OH_VSCODE_PORT:-${VSCODE_PORT:-${CONFIG_VSCODE_PORT:-8001}}}"
VSCODE_BASE_PATH="${OH_VSCODE_BASE_PATH:-${VSCODE_BASE_PATH:-${CONFIG_VSCODE_BASE_PATH:-/vscode}}}"

# Chat-attachment size limits (MB). The static server converts these to the
# injected byte-based window globals. The frontend validates client-side before
# upload; keep the values at or below what the WebSocket layer can carry for
# base64-embedded images (4MB frame cap → ~2.5MB binary) — see config/defaults.json.
MAX_FILE_SIZE_MB="${AGENT_CANVAS_MAX_FILE_MB:-${CONFIG_MAX_FILE_SIZE_MB:-3}}"
MAX_TOTAL_SIZE_MB="${AGENT_CANVAS_MAX_TOTAL_MB:-${CONFIG_MAX_TOTAL_SIZE_MB:-3}}"

# Accept "editor", "/editor" and "/editor/" alike: agent-server strips the
# slashes when it builds the advertised URL, the static-server route table
# needs the leading one, so settle on one spelling rather than one per use site.
normalize_base_path() {
  local p="$1"
  while [ "${p#/}" != "$p" ]; do p="${p#/}"; done
  while [ "${p%/}" != "$p" ]; do p="${p%/}"; done
  printf '/%s' "$p"
}
VSCODE_BASE_PATH="$(normalize_base_path "$VSCODE_BASE_PATH")"
if [ "$VSCODE_BASE_PATH" = "/" ]; then
  log_error "VSCODE_BASE_PATH resolved to the site root — that would route the whole origin to the editor instead of the canvas. Set a prefix such as /vscode."
  exit 1
fi

# The canvas mount gets the same treatment, for the same reason and with the
# same function. static-server normalizes whatever `--base-path` it is handed
# (`canvas` and `/canvas/` both mount at `/canvas`), so comparing a normalized
# editor prefix against a raw canvas one below would let `AGENT_CANVAS_BASE_PATH=canvas`
# with `OH_VSCODE_BASE_PATH=/canvas` past the collision guard and then land both
# on `/canvas` — where the editor route, registered after the SPA mount, takes
# the application over. Normalizing here rather than at the comparison keeps the
# value passed to `--base-path` further down identical to the one guarded.
AGENT_CANVAS_BASE_PATH="$(normalize_base_path "$AGENT_CANVAS_BASE_PATH")"

# static-server keys its route table by prefix and the editor route is
# registered last, so a prefix that collides with an earlier route silently
# replaces it rather than failing: OH_VSCODE_BASE_PATH=/api would send every
# API call to the editor port. Reject collisions and anything that is not a
# plain single-segment path — '=' would be mis-split by the --route parser
# (it cuts at the first '='), and whitespace, '?', '#' or '..' have no
# meaningful reading as a route prefix.
VSCODE_PATH_SEGMENT="${VSCODE_BASE_PATH#/}"
case "$VSCODE_PATH_SEGMENT" in
  */*)
    log_error "VSCODE_BASE_PATH must be a single path segment (got '$VSCODE_BASE_PATH'). Use a prefix such as /vscode."
    exit 1
    ;;
  .|..)
    log_error "VSCODE_BASE_PATH must not be a relative path segment (got '$VSCODE_BASE_PATH'). Use a prefix such as /vscode."
    exit 1
    ;;
  *[!A-Za-z0-9._-]*)
    log_error "VSCODE_BASE_PATH may only contain letters, digits, '.', '_' and '-' (got '$VSCODE_BASE_PATH'). Use a prefix such as /vscode."
    exit 1
    ;;
esac
for reserved in /api /sockets /server_info /alive /health /ready /docs /redoc /openapi.json "${AGENT_CANVAS_BASE_PATH:-}"; do
  if [ -n "$reserved" ] && [ "$VSCODE_BASE_PATH" = "$reserved" ]; then
    log_error "VSCODE_BASE_PATH '$VSCODE_BASE_PATH' collides with an existing route and would take it over. Set a different prefix, such as /vscode."
    exit 1
  fi
done

# The port ends up in a proxy target URL, so a non-numeric value fails at the
# first editor request instead of at startup. Catch it here.
case "$VSCODE_PORT" in
  ''|*[!0-9]*)
    log_error "VSCODE_PORT must be a number (got '$VSCODE_PORT')."
    exit 1
    ;;
esac

export OH_VSCODE_PORT="$VSCODE_PORT"
export OH_VSCODE_BASE_PATH="$VSCODE_BASE_PATH"
# The single route string every static-server instance registers. Derived from
# the exported pair above so the advertised URL and the route cannot diverge.
VSCODE_ROUTE="${VSCODE_BASE_PATH}=http://127.0.0.1:${VSCODE_PORT}"
# <<< vscode-config

# Persistence paths — keep settings, conversations, bash history under a
# single well-known directory that the VOLUME directive exposes.
OPENHANDS_DIR="${HOME}/.openhands"
STATE_DIR="${OPENHANDS_DIR}/${CONFIG_STATE_SUBDIR:-agent-canvas}"
export OH_PERSISTENCE_DIR="${OH_PERSISTENCE_DIR:-${OPENHANDS_DIR}}"
# >>> docker-browser-tools
if [ "${VITE_ENABLE_BROWSER_TOOLS:-}" = "false" ]; then
  export OH_ENABLE_BROWSER="${OH_ENABLE_BROWSER:-false}"
fi
# <<< docker-browser-tools
export OH_CONVERSATIONS_PATH="${OH_CONVERSATIONS_PATH:-${OPENHANDS_DIR}/${CONFIG_CONVERSATIONS:-agent-canvas/conversations}}"
export OH_BASH_EVENTS_DIR="${OH_BASH_EVENTS_DIR:-${OPENHANDS_DIR}/${CONFIG_BASH_EVENTS:-agent-canvas/bash_events}}"

# OH_SECRET_KEY is required for settings/secrets encryption. Without it the
# agent-server refuses to return encrypted secrets → conversation creation
# fails with a 503.  Auto-generate and persist (just like the session API key)
# so the image never runs with a known default.
SECRET_KEY_FILE="${STATE_DIR}/secret-key.txt"
if [ -z "${OH_SECRET_KEY:-}" ]; then
  if [ -f "$SECRET_KEY_FILE" ]; then
    OH_SECRET_KEY="$(cat "$SECRET_KEY_FILE")"
  else
    OH_SECRET_KEY="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    mkdir -p "$(dirname "$SECRET_KEY_FILE")"
    printf '%s' "$OH_SECRET_KEY" > "$SECRET_KEY_FILE"
    chmod 600 "$SECRET_KEY_FILE"
    log "Generated OH_SECRET_KEY (persisted to $SECRET_KEY_FILE)"
  fi
fi
export OH_SECRET_KEY

# API key — generate one if not provided so the image doesn't run wide-open
# by default. LOCAL_BACKEND_API_KEY is the single user-facing env var.
# Persisted so restarts reuse the same key.
API_KEY_FILE="${STATE_DIR}/api-key.txt"

if [ -z "${LOCAL_BACKEND_API_KEY:-}" ] && [ -z "${OH_SESSION_API_KEYS_0:-}" ]; then
  if [ -f "$API_KEY_FILE" ]; then
    LOCAL_BACKEND_API_KEY="$(cat "$API_KEY_FILE")"
  else
    LOCAL_BACKEND_API_KEY="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
    mkdir -p "$(dirname "$API_KEY_FILE")"
    printf '%s' "$LOCAL_BACKEND_API_KEY" > "$API_KEY_FILE"
    chmod 600 "$API_KEY_FILE"
    log "Generated API key (persisted to $API_KEY_FILE)"
  fi
  export OH_SESSION_API_KEYS_0="$LOCAL_BACKEND_API_KEY"
fi

# Both backends share the same API key value and the same `X-Session-API-Key`
# header for authentication.  Default OPENHANDS_AUTOMATION_API_KEY to the
# API key so a single credential secures the whole stack.
EFFECTIVE_SESSION_KEY="${OH_SESSION_API_KEYS_0:-${LOCAL_BACKEND_API_KEY:-}}"
if [ -z "$EFFECTIVE_SESSION_KEY" ]; then
  log "ERROR: No session API key available — cannot configure automation auth"
  exit 1
fi
export OPENHANDS_AUTOMATION_API_KEY="${OPENHANDS_AUTOMATION_API_KEY:-${EFFECTIVE_SESSION_KEY}}"
export AUTOMATION_LOCAL_API_KEY="${AUTOMATION_LOCAL_API_KEY:-${EFFECTIVE_SESSION_KEY}}"
export AUTOMATION_AGENT_SERVER_API_KEY="${AUTOMATION_AGENT_SERVER_API_KEY:-${EFFECTIVE_SESSION_KEY}}"
# >>> automation-kv-secret
# AUTOMATION_KV_SECRET encrypts automation KV state, so it must not change
# under existing data. Earlier versions passed the session key on every start,
# so rotating that key silently changed the encryption key too (#18299). Pin
# the session key in effect the first time -- the value those versions used --
# and keep it. scripts/dev-with-automation.mjs pins to the same file and reads
# it the same way: trailing line breaks are not part of the key, and a blank
# file counts as not pinned yet.
# The pin follows the same contract as the npm launcher: a regular file, not
# a symbolic link; an existing pin is made private (0600) before it is used;
# it is created without replacing anything (a fully written temporary file is
# hard-linked into place), so concurrent first starts all end up with the one
# value that won; a blank pin is replaced by one start holding
# automation-kv-secret.txt.lock, while the others wait for its value; and
# every failure stops the start.
# Deleting the file does not rotate the KV key; it strands existing KV state.
AUTOMATION_KV_SECRET_FILE="${STATE_DIR}/automation-kv-secret.txt"
AUTOMATION_KV_SECRET_LOCK="${AUTOMATION_KV_SECRET_FILE}.lock"
_kv_locked=""
kv_secret_fail() {
  [ -z "$_kv_locked" ] || rmdir "$AUTOMATION_KV_SECRET_LOCK" 2>/dev/null
  log_error "$1"
  exit 1
}
if [ -z "${AUTOMATION_KV_SECRET:-}" ]; then
  mkdir -p "$STATE_DIR" ||
    kv_secret_fail "Cannot create $STATE_DIR for automation-kv-secret.txt"
  # Up to about 5 seconds of waiting on another start that replaces a blank pin.
  for ((_kv_attempt = 0; _kv_attempt < 50; _kv_attempt++)); do
    if [ -L "$AUTOMATION_KV_SECRET_FILE" ]; then
      kv_secret_fail "$AUTOMATION_KV_SECRET_FILE is a symbolic link; refusing to use it"
    fi
    if [ -e "$AUTOMATION_KV_SECRET_FILE" ]; then
      [ -f "$AUTOMATION_KV_SECRET_FILE" ] ||
        kv_secret_fail "$AUTOMATION_KV_SECRET_FILE is not a regular file"
      _kv_mode="$(stat -c '%a' "$AUTOMATION_KV_SECRET_FILE" 2>/dev/null ||
        stat -f '%Lp' "$AUTOMATION_KV_SECRET_FILE")" ||
        kv_secret_fail "Cannot read the permissions of $AUTOMATION_KV_SECRET_FILE"
      [[ "$_kv_mode" =~ ^[0-7]+$ ]] ||
        kv_secret_fail "Cannot read the permissions of $AUTOMATION_KV_SECRET_FILE"
      if (( 8#$_kv_mode & 8#077 )); then
        chmod 600 "$AUTOMATION_KV_SECRET_FILE" ||
          kv_secret_fail "Cannot make $AUTOMATION_KV_SECRET_FILE private (chmod 600)"
      fi
      PINNED_KV_SECRET="$(cat "$AUTOMATION_KV_SECRET_FILE")" ||
        kv_secret_fail "Cannot read $AUTOMATION_KV_SECRET_FILE"
      while [[ "$PINNED_KV_SECRET" == *[$'\r\n'] ]]; do
        PINNED_KV_SECRET="${PINNED_KV_SECRET%?}"
      done
      if [[ "$PINNED_KV_SECRET" == *[![:space:]]* ]]; then
        AUTOMATION_KV_SECRET="$PINNED_KV_SECRET"
        break
      fi
      # A blank pin counts as not pinned yet. One start takes the lock and
      # reads the pin again under it; the others wait for the new value.
      if [ -z "$_kv_locked" ]; then
        if mkdir "$AUTOMATION_KV_SECRET_LOCK" 2>/dev/null; then
          _kv_locked=1
        else
          sleep 0.1
        fi
        continue
      fi
      # Still blank under the lock: rename a fully written file over it, so
      # the pin never goes missing for the starts that are waiting.
      _kv_tmp="$(mktemp "${AUTOMATION_KV_SECRET_FILE}.XXXXXX")" ||
        kv_secret_fail "Cannot write to $STATE_DIR"
      if ! printf '%s' "$EFFECTIVE_SESSION_KEY" > "$_kv_tmp" ||
        ! mv -f "$_kv_tmp" "$AUTOMATION_KV_SECRET_FILE"; then
        rm -f "$_kv_tmp"
        kv_secret_fail "Cannot replace the blank $AUTOMATION_KV_SECRET_FILE"
      fi
      AUTOMATION_KV_SECRET="$EFFECTIVE_SESSION_KEY"
      log "Pinned AUTOMATION_KV_SECRET (persisted to $AUTOMATION_KV_SECRET_FILE)"
      break
    fi
    # mktemp creates the file with mode 0600.
    _kv_tmp="$(mktemp "${AUTOMATION_KV_SECRET_FILE}.XXXXXX")" ||
      kv_secret_fail "Cannot write to $STATE_DIR"
    if ! printf '%s' "$EFFECTIVE_SESSION_KEY" > "$_kv_tmp"; then
      rm -f "$_kv_tmp"
      kv_secret_fail "Cannot write $_kv_tmp"
    fi
    if ln "$_kv_tmp" "$AUTOMATION_KV_SECRET_FILE" 2>/dev/null; then
      rm -f "$_kv_tmp"
      AUTOMATION_KV_SECRET="$EFFECTIVE_SESSION_KEY"
      log "Pinned AUTOMATION_KV_SECRET (persisted to $AUTOMATION_KV_SECRET_FILE)"
      break
    fi
    rm -f "$_kv_tmp"
    # Another start pinned first: read its value on the next pass. Anything
    # else is a failure to persist.
    [ -e "$AUTOMATION_KV_SECRET_FILE" ] || [ -L "$AUTOMATION_KV_SECRET_FILE" ] ||
      kv_secret_fail "Cannot create $AUTOMATION_KV_SECRET_FILE"
  done
  [ -n "${AUTOMATION_KV_SECRET:-}" ] ||
    kv_secret_fail "Could not pin AUTOMATION_KV_SECRET in $AUTOMATION_KV_SECRET_FILE (if no other start is running, remove $AUTOMATION_KV_SECRET_LOCK)"
  if [ -n "$_kv_locked" ]; then
    rmdir "$AUTOMATION_KV_SECRET_LOCK"
    _kv_locked=""
  fi
fi
export AUTOMATION_KV_SECRET
# <<< automation-kv-secret
export OPENHANDS_REMOTE_WS_READY_REQUIRED="${OPENHANDS_REMOTE_WS_READY_REQUIRED:-false}"
if [ -z "${AUTOMATION_POSTHOG_API_KEY:-}" ]; then
  if [ -n "${VITE_POSTHOG_API_KEY:-}" ]; then
    export AUTOMATION_POSTHOG_API_KEY="$VITE_POSTHOG_API_KEY"
  elif [ "${VITE_DO_NOT_TRACK:-}" != "1" ]; then
    export AUTOMATION_POSTHOG_API_KEY="${CONFIG_POSTHOG_API_KEY:-}"
  fi
fi
if [ -n "${AUTOMATION_POSTHOG_API_KEY:-}" ]; then
  export AUTOMATION_POSTHOG_HOST="${AUTOMATION_POSTHOG_HOST:-${VITE_POSTHOG_HOST:-${CONFIG_POSTHOG_HOST:-}}}"
fi

# Configure product analytics for the agent-server. The SDK uses its own
# OH_TELEMETRY_* variables, so mirror the same Canvas/PostHog defaults used by
# the frontend and automation backend while preserving explicit operator
# overrides. Consent stays in persisted settings, where the backend/UI owns it.
if [ "${VITE_DO_NOT_TRACK:-}" = "1" ]; then
  export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"
fi

if [ -z "${OH_TELEMETRY_POSTHOG_API_KEY:-}" ]; then
  if [ -n "${VITE_POSTHOG_API_KEY:-}" ]; then
    export OH_TELEMETRY_POSTHOG_API_KEY="$VITE_POSTHOG_API_KEY"
  elif [ "${DO_NOT_TRACK:-}" != "1" ]; then
    export OH_TELEMETRY_POSTHOG_API_KEY="${CONFIG_POSTHOG_API_KEY:-}"
  fi
fi

if [ -z "${OH_TELEMETRY_EXPORTER:-}" ] && [ -n "${OH_TELEMETRY_POSTHOG_API_KEY:-}" ]; then
  export OH_TELEMETRY_EXPORTER="posthog"
fi

if [ "${OH_TELEMETRY_EXPORTER:-}" = "posthog" ] && [ -n "${OH_TELEMETRY_POSTHOG_API_KEY:-}" ]; then
  export OH_TELEMETRY_POSTHOG_HOST="${OH_TELEMETRY_POSTHOG_HOST:-${VITE_POSTHOG_HOST:-${CONFIG_POSTHOG_HOST:-}}}"
fi

# AGENT_SERVER_URL — needed by automation sandbox callbacks.
export AGENT_SERVER_URL="${AGENT_SERVER_URL:-http://127.0.0.1:${AGENT_SERVER_PORT}}"

# AUTOMATION_AGENT_SERVER_URL — the URL the automation service uses to reach
# the agent-server REST API (tarball upload, bash dispatch, auth key minting).
# When set, ServiceSettings.is_local_mode returns True, enabling local API key
# authentication. Without this, the automation server falls back to validating
# keys against the OpenHands cloud API (app.all-hands.dev), which returns 401
# for locally-generated session keys.
export AUTOMATION_AGENT_SERVER_URL="${AUTOMATION_AGENT_SERVER_URL:-http://127.0.0.1:${AGENT_SERVER_PORT}}"

# Keep the legacy canvas_ui_tool module importable when the agent-server restores
# conversations whose persisted metadata still references its module qualname.
# It is also imported at startup below (--import-modules) so its builtin
# FinishTool registration lets automation runs resolve the tool on their
# remote conversations (see the note at the bottom of tools/canvas_ui_tool.py).
export OH_EXTRA_PYTHON_PATH="${OH_EXTRA_PYTHON_PATH:-/opt/agent-canvas/tools}"

# Read public skills from the extensions package the frontend was built with,
# so Canvas and the agent-server see the same catalog and nothing is cloned.
export EXTENSIONS_PATH="${EXTENSIONS_PATH:-/opt/agent-canvas/node_modules/@openhands/extensions}"
AGENT_SERVER_IMPORT_MODULES="canvas_ui_tool"

# Both backends default LOG_LEVEL to INFO, at which level they emit
# library-internal startup chatter — the automation service alone prints
# several hundred lines of SQLAlchemy ORM model-mapping output — that buries
# the service URLs and readiness lines a user actually needs. Default to
# WARNING so a plain `docker run` is readable, while an explicit LOG_LEVEL
# still raises verbosity for troubleshooting.
#
# >>> backend-log-level: extracted by __tests__/scripts/docker-log-level.test.ts
# >>> — keep the markers.
export LOG_LEVEL="${LOG_LEVEL:-WARNING}"
export AUTOMATION_LOG_LEVEL="${AUTOMATION_LOG_LEVEL:-${LOG_LEVEL}}"

# Narrow a logging level to something Uvicorn's `--log-level` accepts.
#
# The app reads LOG_LEVEL through Python's logging module, which also
# understands the aliases WARN/FATAL/NOTSET and numeric levels; Uvicorn
# validates the flag with a click.Choice over the names below and exits 2 on
# anything else, which would take the automation service down. Unrepresentable
# values fall back to `info`, Uvicorn's own default.
uvicorn_log_level() {
  local _uvicorn_level
  _uvicorn_level="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "${_uvicorn_level}" in
    critical | error | warning | info | debug | trace) printf '%s' "${_uvicorn_level}" ;;
    warn) printf 'warning' ;;
    fatal) printf 'critical' ;;
    '' | *[!0-9]*) printf 'info' ;;
    *)
      # Strip leading zeros so `030` reads as 30. Only two digits can fall
      # below 50, so anything longer is clamped — `[` compares as a signed
      # integer and errors out on a value it cannot hold.
      while [ "${#_uvicorn_level}" -gt 1 ] && [ "${_uvicorn_level#0}" != "${_uvicorn_level}" ]; do
        _uvicorn_level="${_uvicorn_level#0}"
      done
      # Python's NOTSET is 0 and means "inherit", not "most verbose".
      if [ "${_uvicorn_level}" = "0" ]; then printf 'info'
      elif [ "${#_uvicorn_level}" -gt 2 ]; then printf 'critical'
      elif [ "${_uvicorn_level}" -ge 50 ]; then printf 'critical'
      elif [ "${_uvicorn_level}" -ge 40 ]; then printf 'error'
      elif [ "${_uvicorn_level}" -ge 30 ]; then printf 'warning'
      elif [ "${_uvicorn_level}" -ge 20 ]; then printf 'info'
      elif [ "${_uvicorn_level}" -ge 10 ]; then printf 'debug'
      else printf 'trace'
      fi
      ;;
  esac
}
# <<< backend-log-level

# Track child PIDs so we can clean up on exit.
PIDS=()

cleanup() {
  log "Shutting down..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  exit 0
}
trap cleanup EXIT SIGINT SIGTERM

# ── 1. Start Agent Server ────────────────────────────────────────────────────
log "Starting agent-server on port $AGENT_SERVER_PORT..."

if command -v openhands-agent-server >/dev/null 2>&1; then
  # Binary build (production image)
  openhands-agent-server --port "$AGENT_SERVER_PORT" \
    --import-modules "$AGENT_SERVER_IMPORT_MODULES" &
elif [ -x /agent-server/.venv/bin/python ]; then
  # Source build (development image)
  /agent-server/.venv/bin/python -m openhands.agent_server --port "$AGENT_SERVER_PORT" \
    --import-modules "$AGENT_SERVER_IMPORT_MODULES" &
else
  log_error "Cannot find agent-server binary or source venv."
  exit 1
fi
PIDS+=($!)

# ── 2. Start Automation Server ───────────────────────────────────────────────
log "Starting automation server on port $AUTOMATION_PORT..."

# File storage — use local filesystem unless the user has configured cloud
# storage.  Without FILE_STORE=local the automation backend may fall back
# to a cloud provider (S3/GCS) which will fail without credentials, causing
# tarball-based presets (preset/prompt, preset/plugin) to silently error.
export FILE_STORE="${FILE_STORE:-local}"
export LOCAL_STORAGE_PATH="${LOCAL_STORAGE_PATH:-${OPENHANDS_DIR}/storage}"
mkdir -p "$LOCAL_STORAGE_PATH"

# AUTOMATION_BASE_URL — the publicly-reachable base URL for the automation
# service.  Appended to callback URLs and injected into each sandbox as
# AUTOMATION_API_URL.  Defaults to the unified ingress.
export AUTOMATION_BASE_URL="${AUTOMATION_BASE_URL:-http://127.0.0.1:${PORT}}"

# AUTOMATION_WORKSPACE_BASE — where automation runs unpack tarballs.
export AUTOMATION_WORKSPACE_BASE="${AUTOMATION_WORKSPACE_BASE:-${OPENHANDS_DIR}/workspaces}"
mkdir -p "$AUTOMATION_WORKSPACE_BASE"

# Default to SQLite so the automation server works out of the box without
# an external PostgreSQL instance. Users can override AUTOMATION_DB_URL to
# point at a real Postgres for production deployments.
if [ -z "${AUTOMATION_DB_URL:-}" ]; then
  AUTOMATION_DB_FILE="${OPENHANDS_DIR}/${CONFIG_AUTOMATION_DB:-automation/automations.db}"
  mkdir -p "$(dirname "$AUTOMATION_DB_FILE")"
  export AUTOMATION_DB_URL="sqlite+aiosqlite:///${AUTOMATION_DB_FILE}"
  log "Using SQLite database: $AUTOMATION_DB_URL"
fi

# The automation server uses uvicorn. Set AUTOMATION_PORT via its CLI.
if command -v uvicorn >/dev/null 2>&1; then
  uvicorn openhands.automation.app:app \
    --host 0.0.0.0 \
    --port "$AUTOMATION_PORT" \
    --log-level "$(uvicorn_log_level "${LOG_LEVEL}")" &
  PIDS+=($!)
elif python -c "import openhands.automation" 2>/dev/null; then
  python -m uvicorn openhands.automation.app:app \
    --host 0.0.0.0 \
    --port "$AUTOMATION_PORT" \
    --log-level "$(uvicorn_log_level "${LOG_LEVEL}")" &
  PIDS+=($!)
else
  log "WARNING: Automation server not found, skipping."
fi

# ── 3. Wait for backends to be ready ─────────────────────────────────────────
wait_for_port() {
  local port=$1 name=$2 max_wait=${3:-30}
  local elapsed=0
  while ! (echo >/dev/tcp/127.0.0.1/"$port") 2>/dev/null; do
    sleep 1
    elapsed=$((elapsed + 1))
    if [ "$elapsed" -ge "$max_wait" ]; then
      log "WARNING: $name on port $port did not become ready within ${max_wait}s"
      return 1
    fi
  done
  log "$name is ready on port $port"
}

wait_for_port "$AGENT_SERVER_PORT" "Agent Server" 60 &
WAIT_PID1=$!
wait_for_port "$AUTOMATION_PORT" "Automation Server" 60 &
WAIT_PID2=$!
wait "$WAIT_PID1" "$WAIT_PID2"

# ── 4. Start static server (frontend + proxy) ────────────────────────────────
log "Starting frontend + proxy on port $PORT..."

# Describe the local runtime services so the frontend can populate the agent's
# <RUNTIME_SERVICES> system-prompt block (without it the agent does not know how
# to reach the local automation backend and falls back to the cloud API). These
# URLs are runtime config (overridable at `docker run`), so build the JSON here
# from the sandbox-facing URLs the entrypoint already exports. static-server.mjs
# appends it to /server_info as runtime_services and also injects the legacy
# window global for older frontend bundles.
RUNTIME_SERVICES_INFO="$(node /opt/agent-canvas/runtime-services-info.mjs \
  --mode docker \
  --agent-host-alias 127.0.0.1 \
  --agent-server-url "$AGENT_SERVER_URL" \
  --automation-url "$AUTOMATION_BASE_URL")"

# EFFECTIVE_SESSION_KEY is set above from LOCAL_BACKEND_API_KEY or the persisted api-key.txt.
# Binding every interface is required so Docker published ports can reach the
# process: :: (dual-stack), or 0.0.0.0 on kernels without IPv6, where binding ::
# fails with EAFNOSUPPORT. Because the container cannot tell whether the host
# published that port on loopback or every interface, session-key injection
# stays disabled unless the operator explicitly opts in.
STATIC_SERVER_BIND_HOST="::"
if [ ! -e /proc/net/if_inet6 ]; then
  log "IPv6 is not available; binding the frontend to 0.0.0.0."
  STATIC_SERVER_BIND_HOST="0.0.0.0"
fi
# >>> docker-session-key-policy: extracted by the regression test below.
STATIC_SERVER_SESSION_KEY_ARGS=()
if [ "${AGENT_CANVAS_ALLOW_LAN_SESSION_KEY:-false}" = "true" ]; then
  log "WARNING: Embedding the session API key in frontend HTML; publish port $PORT on host loopback only."
  STATIC_SERVER_SESSION_KEY_ARGS+=(--allow-lan-session-key)
fi
# <<< docker-session-key-policy
# Pass the key in the child environment, not its process arguments.
SESSION_API_KEY="$EFFECTIVE_SESSION_KEY" node /opt/agent-canvas/static-server.mjs \
  --port "$PORT" \
  --host "$STATIC_SERVER_BIND_HOST" \
  "${STATIC_SERVER_SESSION_KEY_ARGS[@]}" \
  --dir /opt/agent-canvas/frontend \
  --base-path "$AGENT_CANVAS_BASE_PATH" \
  --runtime-services-info "$RUNTIME_SERVICES_INFO" \
  --route "/api/automation=http://127.0.0.1:${AUTOMATION_PORT}" \
  --route "/api=http://127.0.0.1:${AGENT_SERVER_PORT}" \
  --route "/server_info=http://127.0.0.1:${AGENT_SERVER_PORT}" \
  --route "/sockets=http://127.0.0.1:${AGENT_SERVER_PORT}" \
  --route "/alive=http://127.0.0.1:${AGENT_SERVER_PORT}" \
  --route "/health=http://127.0.0.1:${AGENT_SERVER_PORT}" \
  --route "/ready=http://127.0.0.1:${AGENT_SERVER_PORT}" \
  --route "/docs=http://127.0.0.1:${AGENT_SERVER_PORT}" \
  --route "/redoc=http://127.0.0.1:${AGENT_SERVER_PORT}" \
  --route "/openapi.json=http://127.0.0.1:${AGENT_SERVER_PORT}" \
  --route "$VSCODE_ROUTE" \
  --vscode-base-path "$VSCODE_BASE_PATH" \
  --no-referrer-prefix "$VSCODE_BASE_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --max-total-size-mb "$MAX_TOTAL_SIZE_MB" &
STATIC_PID=$!
PIDS+=("$STATIC_PID")

# ── 5. (Optional) Public-mode static server ─────────────────────────────────
# When PUBLIC_MODE_PORT is set, start a second static-server instance that
# serves the same frontend WITHOUT injecting the session key into the HTML
# (--auth-required). This is used by auth-mode E2E tests to verify the
# ApiKeyEntryScreen gate, key rotation recovery, etc.
#
# Neither the editor route nor --vscode-base-path is registered here, and the
# pair is deliberate: the route is what would serve the editor, and the flag is
# what tells the frontend this origin can. Omitting only the route would leave
# the control rendering and falling through to the SPA, because the agent-server
# it shares with the main instance still reports the editor as available.
#
# --auth-required only
# controls whether the session key is injected into the served HTML; the
# dispatcher matches routes before it reaches that flag, so proxied paths are
# not gated by it. The routes above are safe on that footing because
# agent-server enforces the session key itself, but the editor's own
# credential is the connection token agent-server puts in the query string —
# and agent-server derives that token from session_api_keys[0], so it is the
# same secret that authenticates /api. Registering the route here would put
# that secret in a browser-navigable URL on the origin that exists precisely
# to test the unauthenticated case, where it would persist in history and
# leak by Referer from the workbench's own subresources.
#
# The token's scope is upstream's to fix and is tracked in
# OpenHands/software-agent-sdk#4317; if the editor gets a credential of its own,
# this exclusion and the --no-referrer-prefix below can both be revisited.
#
# MARS_WEB=0 keeps the DigitalOcean bridge on the main instance only. This one
# binds every interface with no key, so an opted-in container would otherwise
# warn on every start, or, with MARS_WEB_KEY set, host a second bridge with its
# own sign-in state.
if [ -n "${PUBLIC_MODE_PORT:-}" ]; then
  log "Starting public-mode frontend on port $PUBLIC_MODE_PORT (--auth-required)..."
  MARS_WEB=0 node /opt/agent-canvas/static-server.mjs \
    --port "$PUBLIC_MODE_PORT" \
    --host "$STATIC_SERVER_BIND_HOST" \
    --dir /opt/agent-canvas/frontend \
    --base-path "$AGENT_CANVAS_BASE_PATH" \
    --auth-required \
    --runtime-services-info "$RUNTIME_SERVICES_INFO" \
    --max-file-size-mb "$MAX_FILE_SIZE_MB" \
    --max-total-size-mb "$MAX_TOTAL_SIZE_MB" \
    --route "/api/automation=http://127.0.0.1:${AUTOMATION_PORT}" \
    --route "/api=http://127.0.0.1:${AGENT_SERVER_PORT}" \
    --route "/server_info=http://127.0.0.1:${AGENT_SERVER_PORT}" \
    --route "/sockets=http://127.0.0.1:${AGENT_SERVER_PORT}" \
    --route "/alive=http://127.0.0.1:${AGENT_SERVER_PORT}" \
    --route "/health=http://127.0.0.1:${AGENT_SERVER_PORT}" \
    --route "/ready=http://127.0.0.1:${AGENT_SERVER_PORT}" \
    --route "/docs=http://127.0.0.1:${AGENT_SERVER_PORT}" \
    --route "/redoc=http://127.0.0.1:${AGENT_SERVER_PORT}" \
    --route "/openapi.json=http://127.0.0.1:${AGENT_SERVER_PORT}" &
  PIDS+=($!)
fi

log "All services started. Unified entry point: http://0.0.0.0:${PORT}/"

# Keep the container alive while the static-server (ingress) is running.
# Backend crashes (agent-server, automation) are tolerated — the proxy
# returns 502 for downed routes, matching the non-Docker path where each
# service is an independent host process.
#
# Pattern: `sleep & wait $!` makes `wait` (a bash builtin) the foreground
# operation.  Unlike a bare `sleep`, the builtin `wait` is interrupted
# immediately when a trapped signal (SIGTERM/SIGINT) arrives, so cleanup()
# fires without delay.  cleanup() calls `exit 0` to terminate after the
# trap returns.  The loop re-checks the static-server PID every 10 s so the
# container exits promptly if the ingress process dies on its own.
while kill -0 "$STATIC_PID" 2>/dev/null; do
  sleep 10 & wait $!
done
log_error "Static server (PID $STATIC_PID) exited"
exit 1
