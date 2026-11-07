# Device pairing verification

## Stack refresh validation

The pairing change is reapplied to the current third-stack head `45fb8ef22505cc76d6ce5c35f3bb4761e7f41077`, preserving its newer OAuth, credential storage, WebSocket limits, and explicit `MARS_WEB=1` opt-in. This supersedes the Windows-only validation below for the refreshed code. Linux, Node 24.22.1; npm 11.12.1 was used for the lock update with `--min-release-age=7`. Registry metadata dates qrcode.react 4.2.0 to 2024-12-11; the lock update adds only that existing pairing dependency.

Pairing now uses the stack's authoritative `getMarsBridgeHeaders()` policy. A regression through the real create-invitation HTTP request demonstrates that the active host key wins over an earlier stale key for the same origin: the old pairing helper fails, and the refreshed implementation passes. The pairing route follows the existing app layout and stays a top-level route, preserving the verification driver's existing route-discovery behavior without editing the driver.

- Six focused suites: 157 passed (`device-pairing`, API/web-server bridge, tunnel bridge, Managed Agents view, and direct-call guard).
- Root suite: 23 passed, 4 failed; an unmodified worktree at the stack head has 22 passed and the same four Cloud-auth failures. The additional unauthenticated pairing route test passes.
- Typecheck, full source Prettier check, relevant-file ESLint, translation completeness, and `git diff --check` pass. Relevant ESLint reports two existing arbitrary-class warnings in the pairing dialog.
- Production app and library builds pass with `RAYON_NUM_THREADS=2 NODE_OPTIONS=--max-old-space-size=2048`. Earlier unrestricted/native-parallel builds were killed with exit 137. Whole-source ESLint was also killed with exit 137, so no complete `npm run lint` pass is claimed for this refresh.
- Read-only verification-map selection: route discovery now passes; the remaining coverage test still reports the new `/pair` route as unmapped. No verify-* specification, fixture, or driver has been changed.

No full repository test run, real DigitalOcean account, live model, cross-device QR scan, or new visual capture was performed for this refresh. The PR remains draft pending independent verification-map maintenance, human testing notes, current CI review, and live visual evidence. The existing `HUMAN:` PR section is untouched.

## Historical validation before the stack refresh

Base: OpenHands/OpenHands PR #18707, d6203bb0b39aa0e2c63901ab73c01482736e12a8.
Environment: Windows, Node 24.14.1, npm 11.11.0. No DigitalOcean account, OAuth approval, paid model, or DigitalOcean session was used.

## Passing checks

- npm run lint: typecheck, ESLint (existing warnings), and formatting pass.
- npm run build and npm run build:lib pass.
- npm run check-translation-completeness passes for all 16 locales.
- Focused Vitest run: 113 tests passed across the browser pairing, web client, web server, tunnel bridge, Managed Agents, and direct-call guard suites.
- The separate root regression (pairing landing page before auth/onboarding): 1 passed.
- Server regressions were run first and failed before implementing pairing. They now cover one-time and concurrent redemption, five-minute expiry, cancel/replacement/sign-out, cross-origin refusal, disabled pairing without a host key, and the admitted cookie surviving a new bridge instance.

## Real browser and server run

1. Served the production build with scripts/static-server.mjs on loopback, port 18811, --auth-required, telemetry disabled, an isolated MARS_STATE_DIR and a disposable MARS_WEB_KEY. The key was not injected into HTML.
2. Created an invitation through the real authenticated POST /mars/pairing/create endpoint as test arrangement.
3. Opened its /pair fragment link in the in-app browser. The real page displayed the full-access explanation and Connect this device. Its address was already /pair with no fragment.
4. Clicked Connect this device. The browser navigated to / without a key-entry step. The onboarding backend form had a non-empty password field; no key was typed into it.
5. Restarted the static server with the same key and reloaded the browser. Its saved credential was still present. A real openhands-agent-server 1.56.0 was started at 18812, with /api, /server_info and /sockets routed to it. Direct checks through 18811 returned 401 without the key and 200 for authenticated settings; the Agent Server log also recorded 200 settings, profile and schema requests from the browser. No LLM conversation was started.

This validates browser admission and the existing host-auth transport. It does not prove live DigitalOcean restore, OAuth renewal, a real QR scan on an iPad, or the issuer's signed-in DigitalOcean view.

The in-app browser's screenshot operation returned Unable to capture screenshot. No screenshot or video has been fabricated. Visual release evidence is still needed. The repository control-openhands launcher also failed on Windows (directory symlink privilege, then its isolated launcher reported uvx unavailable). Manual production server commands above are distinct from a passing control-openhands run. Tabbit's CLI registration was unavailable; no user's browser was restarted or reconfigured.

## Full-suite comparison and remaining gates

- Head's full npm test run: 9235 passed, 25 failed, 36 skipped, 7 todo. This run discovered files before the new component suite was added; its focused result is reported above.
- Unmodified parent archive, same machine and installed dependencies: 9234 passed, 23 failed, 36 skipped, 7 todo. An additional targeted parent run reproduced the Cloud-onboarding lazy-load timeout, adapter, Vite and POSIX-permission failures.
- Most failures are shared Windows/POSIX assumptions or parent regressions. Two verification-map tests additionally reject the new /pair route because it has no feature-map entry. AGENTS.md requires implementors to leave verify-* skills and maps unchanged and report this coverage gap for separate maintenance. A second Cloud-onboarding timeout occurred under full-suite load on head but passed in a focused rerun; the first timeout reproduces on parent. The archive has no Git metadata, so its map-baseline check differs too.
- Keep the PR draft pending Linux CI, independent feature-map maintenance, human testing notes, and live visual evidence. The earlier three draft PRs are dependencies; this PR alone does not close #18704.
