# Isolated runtime with control-openhands

Use the maintained [verification skill](../SKILL.md) and its
[feature map](feature-map/README.md) for current selectors and driving recipes.
The original manual launcher and Playwright smoke helper have been superseded by
`control-openhands`; the 58-check inventory under `features/` remains historical.

## Launch and check the exact checkout

Require Node >=24, the committed npm dependencies, uv/uvx, and a supported Chromium.
Run from a dedicated, clean worktree. The CLI uses private state, generated keys,
free loopback ports, and a build-input marker; it never rewrites a marker to bless
an unrelated build or resets the user's checkout.

```sh
npm ci --ignore-scripts --no-audit --no-fund
export PATH="$PWD/.agents/skills/verify-openhands/scripts:$PATH"
control-openhands launch --help
export OH_VERIFY_RUN=$(control-openhands launch --new --print-run)
control-openhands doctor
control-openhands onboard --skip
```

Set `CONTROL_OPENHANDS_BROWSER` to an installed Chromium executable when the pinned
Playwright browser is unavailable. Use `launch --public` plus `login` when the
API-key entry screen is the check. Skipped onboarding is not onboarding-success
evidence. Record the full source SHA and the actual backend versions from launch
and doctor. Configure a provider only when model execution is in the authorized
scope and budget; otherwise mark dependent checks blocked.

## Drive and retain evidence

Choose the affected families from the maintained map. Its recipes use the same
CLI for UI actions, safe fixture arrangement, network observation, and evidence.
For example, inspect the real secrets page before following its create/delete
recipe:

```sh
control-openhands browser goto /settings/secrets
control-openhands browser snapshot
control-openhands browser testids
control-openhands browser screenshot --feature F14.open --name secrets
```

A route rendering is not proof that a feature works. Follow every required user
entry point, record expected and actual results, and inspect the captured images.
Use `browser record` for ordering or timing. API writes arrange preconditions and
never replace a UI proof. Do not use route interception or scripted model replies
as live evidence. Run doctor after surprising failures before blaming the app.

Keep BASE and TARGET in separate worktrees and isolated runs. With several runs,
pass each exact run directory via `--run`; never guess a shared current instance.
Do not advance an accepted baseline after partial or failed verification.

## Stop only the owned instance

```sh
control-openhands evidence report
control-openhands stop --purge-private
```

The CLI validates process ownership, stops its browser and launcher tree, verifies
that owned ports close, and retains evidence. Review artifacts before publishing;
never publish private browser storage, keys, logs, or a whole run directory.
