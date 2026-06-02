# OpenHands local-mode on WSL — repair & persistence notes

Repaired and persisted on **WSL #2 (BACKEND_PORT=3090)**. Apply the same to **WSL #1 (BACKEND_PORT=3080)** or any other distro by following this doc end-to-end.

---

## 0. Per-instance differences

Adjust **only these two variables** when applying on a different WSL:

| Variable | WSL #1 | WSL #2 (this one) |
|---|---|---|
| `BACKEND_PORT` | `3080` | `3090` |
| `OPENHANDS_EXTERNAL_URL` | `http://localhost:3080` | `http://localhost:3090` |

Everything else (patches, paths, secrets, systemd unit, LLM config) is **identical**.

> ⚠️ WSL2 mirrored networking shares `localhost` across distros — only one WSL can listen on a given port at a time. Pick distinct backend ports per distro.

---

## 1. Root causes found

| # | Symptom | Cause | Fix |
|---|---|---|---|
| 1 | Backend won't start: `No module named 'openhands.sdk.utils.redact'` | Stale Poetry env (openhands-sdk 1.5.2 vs pyproject pin 1.23.1) | `poetry lock && poetry sync` |
| 2 | Frontend crash: `Cannot read properties of undefined (reading 'ENABLE_BILLING')` | Stale prebuilt SPA at backend port; uppercase `FEATURE_FLAGS` from old code vs lowercase `feature_flags` from new backend | Rebuild frontend (`npm run build`) so backend serves a current SPA |
| 3 | `Sandbox failed to start within 120s` — child agent-server log says nothing | `_get_process_status` returns `STARTING` for any psutil status other than `STATUS_RUNNING`. Asyncio uvicorn idles in `sleeping` → status loops forever | Patch to treat anything not stopped/zombie as RUNNING |
| 4 | Child agent-server never binds | `base_port=8000` (default) is squatted on WSL by something invisible from inside the distro (mirrored networking) | Patch default `base_port: 8000 → 18000` |
| 5 | Conversation runs, then `litellm.AuthenticationError: Incorrect API key provided: None` | LLM not configured / Azure config wrong | Set Azure profile correctly (see §4) |
| 6 | `Azure OpenAI Responses API is enabled only for api-version 2025-03-01-preview and later` | SDK hardcoded fallback `api_version="2024-12-01-preview"` overrides env vars | Patch SDK fallback to `2025-04-01-preview` (auto-reapplied via sitecustomize so it survives `poetry sync`) |
| 7 | Backend won't restart on WSL reboot | No init system wiring | systemd `--user` unit + `loginctl enable-linger` |

---

## 2. Prerequisites on the target WSL

```bash
# Node 22 (for frontend build). Skip if `npm -v` already works.
which npm || (curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt install -y nodejs)
node -v && npm -v          # expect v22.x and 10.x

# Poetry must be on PATH for the user. Usually ~/.local/bin/poetry
which poetry || echo "Install poetry first: curl -sSL https://install.python-poetry.org | python3 -"

# Optional sanity
python3 --version          # >= 3.12 matches the openhands-ai-* venv
```

You should already have the repo cloned at `~/ohws/OpenHands`. If not:
```bash
mkdir -p ~/ohws && cd ~/ohws
git clone https://github.com/<your-fork>/OpenHands.git   # or upstream + your fork as a second remote
cd OpenHands
```

---

## 3. Sync Python dependencies

```bash
cd ~/ohws/OpenHands
poetry lock
poetry sync                # may take 2-5 min
poetry run python -c "import openhands.sdk.utils.redact; print('redact OK')"
```

Expect: `openhands-sdk==1.23.1` (or whatever pyproject pins) installed, `redact OK`.

---

## 4. Apply the two repo-tracked code patches

These edit files already in git, so they belong on a branch in your fork.

```bash
cd ~/ohws/OpenHands
git checkout -b local-mode-fixes
```

Edit `openhands/app_server/sandbox/process_sandbox_service.py` — two changes:

### 4a. `_get_process_status` — accept `sleeping` as RUNNING

```diff
             if process.is_running():
                 status = process.status()
-                if status == psutil.STATUS_RUNNING:
-                    return SandboxStatus.RUNNING
-                elif status == psutil.STATUS_STOPPED:
+                if status == psutil.STATUS_STOPPED:
                     return SandboxStatus.PAUSED
+                elif status in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
+                    return SandboxStatus.MISSING
                 else:
-                    return SandboxStatus.STARTING
+                    # RUNNING covers psutil 'running' AND 'sleeping' (typical for
+                    # an asyncio uvicorn process waiting on epoll). The /alive
+                    # check in _process_to_sandbox_info verifies real readiness.
+                    return SandboxStatus.RUNNING
             else:
                 return SandboxStatus.MISSING
```

### 4b. `ProcessSandboxServiceInjector.base_port` 8000 → 18000

```diff
     base_port: int = Field(
-        default=8000, description='Base port number for agent servers'
+        default=18000, description='Base port number for agent servers'
     )
```

### Commit (skip husky hooks since `pre-commit` may not be installed)

```bash
git add openhands/app_server/sandbox/process_sandbox_service.py
git commit --no-verify -m "local-mode: base_port 18000, treat sleeping process as RUNNING

base_port 8000 is squatted by WSL mirrored networking; 18000 is free.
Treat psutil 'sleeping' as RUNNING (asyncio uvicorn idles in epoll),
otherwise sandbox status loops forever and conversation times out at 120s."
```

### Push to your fork (not upstream)

```bash
git remote -v
# add YOUR fork if missing; replace sgireddy-nh with your GH user
git remote add myfork https://github.com/<your-gh-user>/OpenHands.git 2>/dev/null || true
git push -u myfork local-mode-fixes
```

---

## 5. SDK auto-patch — survives `poetry sync`

The SDK's `openhands/sdk/llm/llm.py` hardcodes `api_version = "2024-12-01-preview"` as a fallback when the model starts with `azure/`. Azure's `/responses` endpoint (used by gpt-5 family) requires `≥2025-03-01-preview`. The SDK lives **inside the venv** so any edit vanishes on `poetry sync`. Use a sitecustomize that re-applies it on every Python start.

### 5a. Patch module
```bash
mkdir -p ~/ohws/OpenHands/openhands_local_patches
cat > ~/ohws/OpenHands/openhands_local_patches/__init__.py <<'PY'
"""Auto-applied local patches for OpenHands SDK on this WSL.
Imported via sitecustomize.py at every Python startup."""
import logging, re, inspect
log = logging.getLogger(__name__)

def _patch_azure_api_version_default():
    try:
        from openhands.sdk.llm import llm as _llm
        src_path = inspect.getsourcefile(_llm)
        if not src_path:
            return
        with open(src_path) as f:
            s = f.read()
        new = re.sub(r'"2024-12-01-preview"', '"2025-04-01-preview"', s)
        if new != s:
            with open(src_path, "w") as f:
                f.write(new)
            log.warning(
                "Patched %s: azure api_version default -> 2025-04-01-preview",
                src_path,
            )
    except Exception as e:
        log.warning("Could not patch azure api_version default: %s", e)

_patch_azure_api_version_default()
PY
```

### 5b. sitecustomize loader (auto-runs on every `python` invocation)
```bash
cat > ~/ohws/OpenHands/sitecustomize.py <<'PY'
try:
    import openhands_local_patches  # noqa: F401
except Exception as _e:
    import sys
    print(f"[sitecustomize] local patches skipped: {_e}", file=sys.stderr)
PY
```

Both files must be on `PYTHONPATH` when uvicorn starts — the launcher script in §7 sets that.

### 5c. Verify (one-time, manual)
```bash
~/ohws/OpenHands/oh-run.sh &        # (we'll create this in §7)
# Or just:
PYTHONPATH=~/ohws/OpenHands python3 -c "import sitecustomize; print('ok')"
grep 2025-04-01-preview ~/.cache/pypoetry/virtualenvs/openhands-ai-*/lib/python3.12/site-packages/openhands/sdk/llm/llm.py
```

---

## 6. Persisted environment

Two files outside the repo (avoid committing secrets):

```bash
# Random key for OpenHands at-rest secret encryption (so saved API keys survive restarts)
[ -f ~/.openhands.secret ] || (head -c32 /dev/urandom | xxd -p -c64 > ~/.openhands.secret)
chmod 600 ~/.openhands.secret

# Azure key (one line, no quotes)
nano ~/.azure_key                    # paste your Azure OpenAI key
chmod 600 ~/.azure_key
```

(Optionally mirror the env block into `~/.bashrc` so interactive shells also see it. The systemd-managed backend uses the launcher in §7, not `.bashrc`.)

---

## 7. Launcher script

> ⚠️ Edit `BACKEND_PORT` to match this WSL (`3080` or `3090`).

```bash
cat > ~/ohws/oh-run.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

# ---- env ----
export RUNTIME=local
export BACKEND_PORT=3090                          # <-- 3080 on WSL #1
export AZURE_API_KEY="$(cat $HOME/.azure_key)"
export AZURE_API_BASE='https://dev-ai-gpt5-resource.services.ai.azure.com'
export AZURE_API_VERSION='2025-04-01-preview'
export OPENAI_API_VERSION='2025-04-01-preview'
export OH_SECRET_KEY="$(cat $HOME/.openhands.secret)"
export SANDBOX_VOLUMES="$HOME/ohws:/workspace:rw"
export OPENHANDS_EXTERNAL_URL="http://localhost:${BACKEND_PORT}"

cd "$HOME/ohws/OpenHands"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"          # loads sitecustomize.py + openhands_local_patches/
exec poetry run uvicorn openhands.server.listen:app --host 0.0.0.0 --port "${BACKEND_PORT}"
EOF
chmod +x ~/ohws/oh-run.sh
```

Smoke test manually:
```bash
~/ohws/oh-run.sh                                  # ctrl-C after seeing "Application startup complete"
```

---

## 8. Build the frontend SPA (so `http://localhost:$BACKEND_PORT/` shows the UI)

```bash
cd ~/ohws/OpenHands/frontend
npm install
npm run build
ls build/client/index.html                        # must exist
```

The backend mounts `frontend/build/` at `/` only if that path exists.

If `build/client/index.html` doesn't exist after build, check `npm run build` output for errors.

---

## 9. systemd user service — auto-start across reboots

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/openhands.service <<'EOF'
[Unit]
Description=OpenHands local-mode backend
After=network-online.target

[Service]
Type=simple
ExecStart=%h/ohws/oh-run.sh
Restart=on-failure
RestartSec=5
StandardOutput=append:%h/ohws/oh-be.log
StandardError=append:%h/ohws/oh-be.log

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now openhands
loginctl enable-linger "$USER"                    # keep user services alive without an open shell
systemctl --user status openhands --no-pager | head -15
```

**Expected**: `Active: active (running)` with a `uvicorn` process.

Logs accumulate in `~/ohws/oh-be.log`. Live tail: `journalctl --user -fu openhands`.

---

## 10. Configure the LLM profile (Azure OpenAI gpt-5 family)

After backend is up, do this once via API (use the right `$BACKEND_PORT`):

```bash
PORT=3090                                         # 3080 on WSL #1
api_key="$(cat ~/.azure_key)"

curl -sS -X POST http://127.0.0.1:$PORT/api/v1/settings/profiles/gpt-5.5 \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"azure/gpt-5.5\",
    \"base_url\": \"https://dev-ai-gpt5-resource.services.ai.azure.com\",
    \"api_version\": \"2025-04-01-preview\",
    \"api_key\": \"$api_key\"
  }"

curl -sS -X POST http://127.0.0.1:$PORT/api/v1/settings/profiles/gpt-5.5/activate

# Verify
curl -sS http://127.0.0.1:$PORT/api/v1/settings | python3 -c '
import json,sys
d=json.load(sys.stdin)["agent_settings"]["llm"]
print({k:d.get(k) for k in ["model","base_url","api_version"]})'
```

Expected output:
```
{'model': 'azure/gpt-5.5', 'base_url': 'https://dev-ai-gpt5-resource.services.ai.azure.com', 'api_version': '2025-04-01-preview'}
```

> The model **must** start with `azure/`. Without that prefix LiteLLM routes to vanilla OpenAI, ignores your `base_url`, and demands an OPENAI key.

### Independent credential sanity check (skips OpenHands)
```bash
curl -sS -X POST \
  "https://dev-ai-gpt5-resource.services.ai.azure.com/openai/responses?api-version=2025-04-01-preview" \
  -H "Content-Type: application/json" -H "api-key: $api_key" \
  -d '{"model":"gpt-5.5","input":[{"role":"user","content":"hi"}],"max_output_tokens":16}' \
  | head -c 400; echo
```
A `"status":"completed"` response means key + endpoint + deployment + version are all valid.

---

## 11. Reboot test (the real "permanent" verification)

```powershell
# In Windows PowerShell
wsl --shutdown
```

Wait 10 seconds, reopen the WSL terminal, then:

```bash
systemctl --user is-active openhands              # active
ss -tlnp | grep :$BACKEND_PORT                    # uvicorn listening
curl -sS http://127.0.0.1:$BACKEND_PORT/api/v1/web-client/config | head -c 80; echo
grep -n 2025-04-01-preview ~/.cache/pypoetry/virtualenvs/openhands-ai-*/lib/python3.12/site-packages/openhands/sdk/llm/llm.py
```

Open `http://localhost:$BACKEND_PORT/`, hard-refresh (Ctrl+Shift+R) to bust any cached SPA, send "hello". You should get a real reply.

---

## 12. Operational cheatsheet

| Task | Command |
|---|---|
| Status | `systemctl --user status openhands` |
| Restart | `systemctl --user restart openhands` |
| Live logs | `journalctl --user -fu openhands` |
| File log | `tail -f ~/ohws/oh-be.log` |
| Newest sandbox child log | `ls -t /tmp/openhands-sandboxes/ \| head -1 \| xargs -I{} tail -50 /tmp/openhands-sandboxes/{}/.openhands-agent-server.log` |
| Kill orphan agent-servers | `pkill -9 -f openhands.agent_server` |
| After `poetry sync` | `systemctl --user restart openhands` (sitecustomize re-patches SDK automatically) |
| Reset DB / settings | `rm -rf ~/.openhands/`  *(loses all conversations)* |

---

## 13. Troubleshooting

### "Sandbox failed to start within 120s"
- Check children are alive: `ss -tlnp \| grep -E ':1800[0-9]'`
- If alive but timeout still fires → `_get_process_status` patch missing → re-apply §4a
- If no child even spawned → port conflict on 18000+ → check `ss -tlnp` for what's listening, or bump `base_port` higher

### `ENABLE_BILLING` undefined in browser
- Backend port serves the SPA from `frontend/build/`. If that dir is stale or missing, rebuild it: §8.
- Hard-refresh browser (Ctrl+Shift+R) or test in Incognito to bypass cache.

### `Incorrect API key provided: None`
- LLM profile missing `azure/` prefix, or `api_version` empty, or `api_key` missing.
- Re-apply §10.

### `Azure OpenAI Responses API is enabled only for api-version …`
- SDK auto-patch not active. Check:
  - `grep 2025-04-01-preview ~/.cache/pypoetry/virtualenvs/openhands-ai-*/lib/python3.12/site-packages/openhands/sdk/llm/llm.py`
  - If that grep fails → sitecustomize didn't load → ensure `PYTHONPATH` includes `~/ohws/OpenHands` (it's set in `oh-run.sh`).

### systemd unit exits with code 127
- `poetry` not in PATH. The launcher script in §7 sets `PATH="$HOME/.local/bin:...` explicitly — make sure your `poetry` actually lives there: `which poetry`. Adjust the PATH line if it's elsewhere.

### Stale `GITHUB_TOKEN` 403 noise
- UI Settings → Git Providers → clear / re-add. Cosmetic; doesn't break conversations.

### Frontend build complains about node version
- Need Node 22+: `nvm install 22 && nvm use 22` (no sudo) or the apt install in §2.

---

## 14. Apply checklist (per WSL)

- [ ] §2: `npm`, `poetry`, Python 3.12 available
- [ ] §3: `poetry sync` succeeds, `redact OK`
- [ ] §4: two patches committed on branch `local-mode-fixes`, pushed to fork
- [ ] §5: `openhands_local_patches/__init__.py` + `sitecustomize.py` present
- [ ] §6: `~/.openhands.secret` + `~/.azure_key` exist, `chmod 600`
- [ ] §7: `~/ohws/oh-run.sh` exists, `chmod +x`, **BACKEND_PORT matches this WSL**
- [ ] §8: `frontend/build/client/index.html` exists
- [ ] §9: `systemctl --user is-active openhands` → `active`, `loginctl enable-linger` done
- [ ] §10: profile activated, settings curl shows `azure/gpt-5.5` + `2025-04-01-preview`
- [ ] §11: post-reboot verification passes
