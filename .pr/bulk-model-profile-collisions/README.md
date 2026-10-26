# Bulk model profile name collision

Reproduced on macOS with Agent Canvas 1.33.0, source revision `7a41641c6d8bcf3e2e3c19c08b778781bcf954c2`, Node 24.21.0, npm 11.12.1, real Agent Server 1.58.1, and automation pinned to 1.25.1. No backend responses were mocked, and no model inference or provider credentials were needed for this supported keyless profile-creation path.

## Baseline reproduction

After installing dependencies with `npm ci`, launch the source checkout with the repository verification CLI. It runs `bin/agent-canvas.mjs`, the production launcher used by the npm package, with isolated state and ports.

```sh
export PATH="$PWD/.agents/skills/verify-openhands/scripts:$PATH"
export OH_VERIFY_HOME=/tmp/openhands-qa-bulk
control-openhands launch --new --print-run --run-id qa-model-collision-baseline
```

The run directory was `/tmp/openhands-qa-bulk/qa-model-collision-baseline`. The installed Chrome fallback was used because the pinned Playwright browser was unavailable:

```sh
export CONTROL_OPENHANDS_BROWSER='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
export OH_VERIFY_RUN=/tmp/openhands-qa-bulk/qa-model-collision-baseline
control-openhands browser start
control-openhands doctor
control-openhands onboard --skip
control-openhands browser goto /settings/llm
control-openhands api GET /api/profiles
control-openhands browser click testid=add-models-from-provider
control-openhands browser select testid=add-models-provider azure
control-openhands browser check 'testid=add-models-check-azure/eu/gpt-5.1'
control-openhands browser check 'testid=add-models-check-azure/global/gpt-5.1'
control-openhands browser text testid=add-models-submit
control-openhands browser click testid=add-models-submit
control-openhands browser toasts --history
control-openhands api GET /api/profiles
control-openhands stop
```

`doctor` passed. The initial profile list was empty. The submit button read **Add 2 profiles**, and the success toast read **Added 2 profiles**. The resulting list contained only `gpt-5.1`, pointing to `azure/global/gpt-5.1`; the first model had been overwritten.

Both catalog entries are present among the first 100 Azure rows in the real backend catalog. The frontend derives the same profile name from both IDs and saves both through the upsert-by-name API.

![Two selected profiles before submission](baseline-selected.png)

![Success reports two additions, but only the second model remains](baseline-after-save.png)

The screenshots are unmodified captures from the running Canvas. Only these inspected images and these reproduction notes are published; private run state and browser data are excluded.
