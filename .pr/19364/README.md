# Expected shutdown versus supervised failure

Baseline: `57cd74aaafc5254fa7760db4b444bdd9468f5b7f`.
Product fix: `53f553136eaeefd380f3b79c8add3dfca023e0af`.

The actual `bin/agent-canvas.mjs` entrypoint ran with Node 24.21.0, Agent Server 1.59.0 and automation 1.26.1 in Ubuntu WSL. The comparison reused the same isolated launcher environment and private state created by `control-openhands`; no service was mocked. It required no model configuration or inference.

After the UI and all four services were reachable, the validation driver sent SIGKILL to the launcher's own ingress child four times, waiting for each replacement. It captured the real launcher exit code and probed the ingress, frontend, Agent Server and automation ports. This completed twice on each version:

- Main logs a static restart after the give-up message and reports intentional backend termination as exit-code errors.
- The fix logs Stopping for those services without restarting static or reporting their expected exits as crashes.
- Both versions exit 1 and leave all four ports closed.

`observations.json` records these measurements. `base.log` and `fixed.log` are real process output with ANSI color escapes removed and isolated credential/state details redacted; the original logs remain in the local audit directory. The browser videos show the actual running Canvas during fault injection and its unavailability after shutdown. The internal supervision difference is established by the process logs and measurements, rather than inferred from the unchanged UI.

Reproduce using the issue's four-ingress-crash recipe. For automated coverage, run `npx vitest run __tests__/scripts/dev-with-automation.test.ts`. The focused Windows run passes 105 tests, with three POSIX cases skipped; it covers normal restarts, healthy-peer shutdown and cancellation of an already-pending restart. The existing POSIX automation-startup failure test also gains a stop-log assertion.

Full Windows run: 10,022 passed, 60 failed, 45 pending and 7 todo. All 60 failing names reproduce on unchanged current main; see `windows-main-comparison.json`. Application and library builds pass. Full lint reproduces main's 14 formatting errors in four unchanged files; upstream main CI also fails those checks. Required CI has not been waived.
