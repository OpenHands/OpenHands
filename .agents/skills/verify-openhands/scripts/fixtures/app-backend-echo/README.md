# Real app-backend bridge fixture

This Linux amd64 fixture uses the actual SDK app-backend manager and Canvas
`host.appBackendView.mount`. It serves a fixed HTTP marker and a bounded WebSocket
echo with subprotocol `qa.echo.v1`. It inherits no environment, returns only
credential-presence booleans, and never echoes arbitrary WebSocket input. It needs
Python at `/usr/bin/python3` on the Agent Server host. No model or scheduler runs.

Use only a run that you own and control serially. The standard SDK ingress must
be a separate browser origin from Canvas. The helper changes only that run's
saved `OH_APP_BACKEND_PUBLIC_URL`, backs up its prior presence/value privately,
and prints no environment values. It neither restarts services nor drives a page.
Merely exporting the field is insufficient because the CLI filters launcher
environment variables.

```sh
FIXTURE="$PWD/.agents/skills/verify-openhands/scripts/fixtures/app-backend-echo"
python3 "$FIXTURE/configure-ingress.py" --run "$OH_VERIFY_RUN"
control-openhands restart
control-openhands doctor
control-openhands api GET /server_info --pick capabilities
control-openhands api GET /server_info --pick app_backend_ingress_url
```

Require `canvas_app_backend_bridge_v1` and the actual owned Agent Server URL.
In Apps → Add app install this directory and trust it. Read
`GET /api/canvas-extensions/installed/qa-bridge-echo/backend`; use its exact
`revision` in the prepare and start POST bodies (`--write`). Once ready, open
the app's actual rail link and inspect `testid=qa-bridge-result` plus a screenshot.
HTTP must return `QA_HTTP_7731` without forwarded control key/cookie. WebSocket
must negotiate `qa.echo.v1` and return `QA_ECHO_7731`; connection failure is not a
pass. The direct backend can provide a separate positive protocol control.
Inspect actual proxied Date/Server headers without publishing session URLs.
Navigating away should dispose the admitted session. Its deliberately plain
preformatted diagnostic output is not a responsive-layout specification.

Stop the owned backend, uninstall only this fixture, restore the field and
restart/doctor the same owned run. Verify its prior ingress/capability state and
that the backend subprocess is gone. Restore even after a failed drive:

```sh
python3 "$FIXTURE/configure-ingress.py" --run "$OH_VERIFY_RUN" --restore
control-openhands restart
control-openhands doctor
```

The tarball contains the executable `backend.py`. Its SHA256 is recorded in
`canvas-extension.json`; the readable source and archive must remain identical.
Do not include local settings, tokens, cookies or model credentials in this
fixture or published receipts.
