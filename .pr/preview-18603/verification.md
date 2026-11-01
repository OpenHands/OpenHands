# Re-review, 2026-10-23

## Canvas corrections

Code commit `5dcd0ef61d81` adds named connection/download errors, image decoding failures, and a localized Rich fallback for absolute HTML/Markdown that cannot resolve sibling assets. Plain source stays available. Merged current main at `6be67b820bc3` with a test-only conflict resolved by retaining both suites; resulting source `723cd2d87539`.

The new failure cases reproduced before the correction. Final merged source: 62 focused tests passed across file content, viewer, workspace sessions, Files routes, mutation counters, and the direct API-call guard. Typecheck and production build passed; translation coverage passed. Final changed-file lint passed with 0 errors and 12 existing warnings. No Agent Server endpoint or filesystem permissions changed.

## Remaining merge blocker: published client browser cache

A separate Chromium context called the exact published TypeScript `FileClient.downloadFile` 1.58.1 against real Agent Server 1.56.0 in an isolated production Canvas stack. An owned PNG had an old modification time. After the first read, the probe replaced its bytes and updated its modification time. A second typed-client read returned the original bytes from the browser disk cache; an independent uncached read returned the new bytes. An invalid-key control likewise reused browser-cached bytes while the uncached server request correctly returned 401. This is a client-cache problem, not the server accepting the invalid key.

No Playwright routing, response mocks, cache overrides, DigitalOcean resource changes, or LLM calls were used. `cache-probe-result.json` records hashes and CDP cache flags without keys or response bodies. The probe owns and removes its temporary fixture. This is API/browser-cache evidence, not a completed Files UI before/after demonstration.

Reproduce against a disposable run started with `control-openhands launch`:

```sh
node .pr/preview-18603/cache-probe.mjs "$OH_VERIFY_RUN" /tmp/cache-probe-result.json
```

The smallest dependency is a published SDK TypeScript-client change that disables browser caching for file downloads, followed by a Canvas version bump. A server-only header change cannot repair already-fresh cached entries and would require a server compatibility boundary. No matching released/open fix was found during this review. Keep this PR draft until the dependency and running-Files evidence are complete.

Sources: [FileClient 1.58.1](https://github.com/OpenHands/software-agent-sdk/blob/v1.58.1/clients/typescript/src/client/file-client.ts#L54), [HTTP transport](https://github.com/OpenHands/software-agent-sdk/blob/v1.58.1/clients/typescript/src/client/http-client.ts#L231), [Agent Server 1.56.0](https://github.com/OpenHands/software-agent-sdk/blob/v1.56.0/openhands-agent-server/openhands/agent_server/file_router.py#L160).
