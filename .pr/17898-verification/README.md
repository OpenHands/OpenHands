# Do Not Track initialization verification

AI-assisted implementation, separate design/task review and independent final agent review. No human testing or review is asserted by this artifact.

Both base `35dc8aafaad2d883cf0c6629dea857eb2235895e` and implementation `7aeaa19be14b847ffded31477e45a3f41893ef2e` were production-built with Vite DNT enabled and tested against real AgentServer 1.50.1. Runtime and browser DNT were not additionally injected, so the baseline bug remained observable. Fresh Chrome contexts recorded initial load and reload, requests and storage before navigation; all external requests were blocked and no model executed.

| Observation across two loads | Base | Fixed |
| --- | --- | --- |
| Attempted PostHog configuration/flags requests | 6 | 0 |
| New PostHog persistence | present | absent |
| Successful real-backend responses | 39 | 41 |
| Page errors | 0 | 0 |

The screenshots establish that both builds reach the real application. The accompanying `runtime-summary.json` records the network/storage difference. `before.webm` and `after.webm` show the actual application runs. Sixteen unrelated external resource attempts occur in each version and are recorded separately from analytics; optional automation services were not started.

Validation: 58 telemetry/bootstrap tests pass; canonical lint exits 0 with zero errors and 376 repository warnings; production and ESM/CJS/declaration library builds pass. The complete suite exits 1 with 8,108 passing, 3 failing and 7 todo tests. All three failures also reproduce on the unchanged base with the same dependency environment: the New York epoch-year assertion and two Bash 3.2 nounset/empty-array test-helper assertions. This is not described as an all-green full suite.

Recorded production build command:

```sh
VITE_DO_NOT_TRACK=1 VITE_MOCK_API=false node node_modules/@react-router/dev/bin.js build
```

The selected suites can be rerun with the normal project entry point:

```sh
npm test -- __tests__/services/telemetry.test.ts __tests__/services/telemetry-bootstrap.test.ts --maxWorkers=1
npm run lint
npm run build:lib
```

For app reproduction, serve the production build through the official static launcher and a real local Agent Server. Use a fresh browser profile, observe the Network tab filtered to the PostHog host, and reload. Keep build-time DNT as the only opt-out while comparing the baseline; adding a runtime opt-out would mask that baseline. Inspect local/session storage for PostHog keys. No conversation or paid model invocation is required.

The source SHA256 during the complete runtime and build evidence is `46f8d7c66ed87f6e419fc4b6b7725a3143ec66a566b8ba94c1519c200b1502bd`, unchanged after implementation commit hooks. Full raw traces and failure logs are retained locally; this curated public bundle excludes environments and private runtime state. Remove temporary `.pr` review artifacts before merge on this fork.
