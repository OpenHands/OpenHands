"""
OAuth 2.0 Device Authorization Grant (RFC 8628) implementation.

Provides endpoints for the agent-canvas device flow:
  POST /oauth/device/authorize   — request a device code
  POST /oauth/device/token       — poll for an access token
  POST /oauth/device/verify-authenticated — confirm authorization from browser
  GET  /oauth/device/verify       — HTML page for the user to authorize
"""

import secrets
import string
import time
from dataclasses import dataclass

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

# ---------------------------------------------------------------------------
# In-memory store for pending device authorizations
# ---------------------------------------------------------------------------

@dataclass
class PendingDeviceAuth:
    device_code: str
    user_code: str
    created_at: float
    authorized: bool = False
    denied: bool = False
    # Filled once the user confirms
    access_token: str | None = None

DEVICE_STORE: dict[str, PendingDeviceAuth] = {}  # keyed by device_code
USER_CODE_INDEX: dict[str, str] = {}  # user_code -> device_code

DEVICE_CODE_EXPIRY = 300  # 5 minutes


def _generate_device_code() -> str:
    return secrets.token_urlsafe(32)


def _generate_user_code() -> str:
    chars = string.ascii_uppercase + string.digits
    while True:
        code = "".join(secrets.choice(chars) for _ in range(8))
        if code not in USER_CODE_INDEX:
            return code


def _cleanup_expired() -> None:
    now = time.time()
    expired = [
        dc for dc, entry in DEVICE_STORE.items()
        if now - entry.created_at > DEVICE_CODE_EXPIRY
    ]
    for dc in expired:
        entry = DEVICE_STORE.pop(dc)
        USER_CODE_INDEX.pop(entry.user_code, None)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/oauth", tags=["OAuth Device Flow"])


@router.post("/device/authorize")
async def device_authorize(request: Request):
    """
    RFC 8628 §3.1 — Device Authorization Request.

    Agent Canvas calls this to obtain a device_code and user_code.
    """
    _cleanup_expired()

    device_code = _generate_device_code()
    user_code = _generate_user_code()

    DEVICE_STORE[device_code] = PendingDeviceAuth(
        device_code=device_code,
        user_code=user_code,
        created_at=time.time(),
    )
    USER_CODE_INDEX[user_code] = device_code

    # Build the verification URI pointing back to this OIO server.
    # Agent Canvas opens this URL in a popup for the user to confirm.
    host = request.headers.get("host", "localhost:3000")
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    # Default to https for known cloud domains even if proxy forgets X-Forwarded-Proto
    if proto != "https" and host.startswith("oio.ai-1"):
        proto = "https"
    scheme = "https" if proto == "https" else "http"
    base = f"{scheme}://{host}"
    verification_uri = f"{base}/oauth/device/verify"
    verification_uri_complete = f"{verification_uri}?user_code={user_code}"

    return JSONResponse({
        "device_code": device_code,
        "user_code": user_code,
        "verification_uri": verification_uri,
        "verification_uri_complete": verification_uri_complete,
        "expires_in": DEVICE_CODE_EXPIRY,
        "interval": 3,
    })


@router.post("/device/token")
async def device_token(request: Request):
    """
    RFC 8628 §3.4 — Client Credential Request (Token polling).

    Agent Canvas polls this endpoint until the user has confirmed
    authorization in their browser.
    """
    _cleanup_expired()

    body = await request.form()
    grant_type = body.get("grant_type")
    device_code = body.get("device_code")

    if grant_type != "urn:ietf:params:oauth:grant-type:device_code":
        raise HTTPException(status_code=400, detail="invalid_grant")

    entry = DEVICE_STORE.get(device_code)
    if entry is None:
        # Device code expired or unknown
        return JSONResponse(
            status_code=400,
            content={"error": "expired_token"},
        )

    if entry.denied:
        return JSONResponse(
            status_code=400,
            content={"error": "access_denied"},
        )

    if not entry.authorized:
        # User has not confirmed yet — continue polling
        return JSONResponse(
            status_code=400,
            content={"error": "authorization_pending", "interval": 3},
        )

    # Authorized — return access token
    return JSONResponse({
        "access_token": entry.access_token,
        "token_type": "Bearer",
    })


@router.post("/device/verify-authenticated")
async def device_verify_authenticated(request: Request):
    """
    Confirm device authorization from the browser popup.

    Agent Canvas device-verify page calls this with the user_code to
    mark the pending device auth as approved and generate an API key.
    """
    form = await request.form()
    user_code = form.get("user_code", "")

    device_code = USER_CODE_INDEX.get(user_code)
    if not device_code:
        raise HTTPException(status_code=404, detail="user_code not found")

    entry = DEVICE_STORE.get(device_code)
    if entry is None:
        raise HTTPException(status_code=404, detail="device_code expired")

    # Generate an API key that agent-canvas will use as X-Session-API-Key
    api_key = "sk-" + secrets.token_urlsafe(32)

    entry.authorized = True
    entry.access_token = api_key

    # Clean up index
    USER_CODE_INDEX.pop(user_code, None)

    return JSONResponse({"success": True})


# ---------------------------------------------------------------------------
# HTML verify page — serves a self-contained authorization confirmation page
# ---------------------------------------------------------------------------

_VERIFY_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Confirm Authorization · OpenHands</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #0d1117;
         color: #c9d1d9; display: flex; align-items: center; justify-content: center;
         min-height: 100vh; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px;
          padding: 32px; max-width: 440px; width: 100%; text-align: center; }
  h1 { font-size: 1.4rem; margin-bottom: 8px; color: #f0f6fc; }
  p.sub { font-size: 0.9rem; color: #8b949e; margin-bottom: 24px; }
  .code { font-family: monospace; font-size: 1.6rem; letter-spacing: 0.3em;
          background: #0d1117; padding: 12px; border-radius: 8px;
          border: 1px solid #30363d; margin-bottom: 20px; color: #f0f6fc; }
  .warn { background: #3b2f00; border-left: 3px solid #d29922; text-align: left;
          padding: 10px 14px; border-radius: 4px; font-size: 0.82rem;
          color: #e3b341; margin-bottom: 24px; }
  .buttons { display: flex; gap: 12px; }
  button { flex: 1; padding: 10px; border-radius: 6px; font-size: 0.95rem;
           cursor: pointer; border: none; }
  .btn-cancel { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; }
  .btn-cancel:hover { background: #30363d; }
  .btn-authorize { background: #238636; color: #fff; }
  .btn-authorize:hover { background: #2ea043; }
  .result { padding: 24px; }
  .result svg { width: 48px; height: 48px; margin-bottom: 12px; }
  .success { color: #3fb950; }
  .error { color: #f85149; }
  .hidden { display: none; }
  .spinner { border: 3px solid #30363d; border-top-color: #58a6ff;
             border-radius: 50%; width: 32px; height: 32px;
             animation: spin 0.8s linear infinite; margin: 0 auto 16px; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="card">
  <!-- Form view -->
  <div id="form-view">
    <h1>Confirm Authorization</h1>
    <p class="sub">An application is requesting access to your OpenHands account.</p>
    <div class="code" id="user-code-display"></div>
    <div class="warn">
      ⚠️ Do not share this code. Only enter it if you initiated this request.
    </div>
    <div class="buttons">
      <button class="btn-cancel" onclick="window.close()">Cancel</button>
      <button class="btn-authorize" onclick="authorize()">Authorize</button>
    </div>
  </div>

  <!-- Loading view -->
  <div id="loading-view" class="hidden">
    <div class="spinner"></div>
    <p class="sub">Processing…</p>
  </div>

  <!-- Result view -->
  <div id="result-view" class="hidden">
    <div class="result" id="result-content"></div>
  </div>
</div>

<script>
  const userCode = new URLSearchParams(window.location.search).get('user_code') || '';

  document.getElementById('user-code-display').textContent = userCode;

  // Base URL for API calls (same origin as this page)
  const apiBase = window.location.origin;

  function showLoading() {
    document.getElementById('form-view').classList.add('hidden');
    document.getElementById('loading-view').classList.remove('hidden');
  }
  function showResult(html) {
    document.getElementById('loading-view').classList.add('hidden');
    document.getElementById('result-content').innerHTML = html;
    document.getElementById('result-view').classList.remove('hidden');
  }

  async function authorize() {
    if (!userCode) {
      showResult('<p class="error">No user code found.</p>');
      return;
    }
    showLoading();
    try {
      const res = await fetch(apiBase + '/oauth/device/verify-authenticated', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: 'user_code=' + encodeURIComponent(userCode),
      });
      if (res.ok) {
        showResult(
          '<p class="success" style="font-size:2rem">✓</p>' +
          '<h1 class="success">Authorized!</h1>' +
          '<p class="sub">You may close this window.</p>'
        );
      } else {
        showResult(
          '<p class="error" style="font-size:2rem">✗</p>' +
          '<h1 class="error">Authorization Failed</h1>' +
          '<p class="sub">Please try again.</p>'
        );
      }
    } catch {
      showResult(
        '<p class="error" style="font-size:2rem">✗</p>' +
        '<h1 class="error">Error</h1>' +
        '<p class="sub">Could not connect to server.</p>'
      );
    }
  }
</script>
</body>
</html>
"""


@router.get("/device/verify", response_class=HTMLResponse)
async def device_verify_page(
    user_code: str | None = Query(default=None),
):
    """
    Self-contained HTML page for the user to confirm device authorization.
    Opened in a popup by agent-canvas during the device flow.
    """
    return HTMLResponse(_VERIFY_TEMPLATE)
