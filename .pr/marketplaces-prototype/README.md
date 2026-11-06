# Marketplaces prototype: browser evidence

Canvas `main` at fc9f801f0e plus this branch, run with

```
VITE_FRONTEND_PORT=3177 VITE_BACKEND_BASE_URL=http://127.0.0.1:8000 \
VITE_SESSION_API_KEY=marketplaces-dev npm run dev:mock
```

and driven by `drive.mjs` (headless Chromium through the repo's Playwright,
`node drive.mjs http://127.0.0.1:3177 <out-dir>` from the repo root after
copying the script somewhere under `node_modules/.cache/`). The script seeds the
onboarding/backend localStorage keys, dismisses the telemetry dialog, and
asserts each step; the screenshots are what it saw.

| Screenshot | What it shows |
| --- | --- |
| 01-list | Two marketplaces grouped by scope; the default one counts 79 MCP servers, 73 skills, 9 plugins, 3 Apps, 26 automation templates from the bundled `@openhands/extensions` 0.41.0 |
| 02-default-marketplace | One page with all 190 entries, every kind together |
| 03-search-slack | Search narrows to 6 entries across kinds |
| 04-installed-filter | After "Add to MCP servers" (github) and Install (add-javadoc): the Installed filter |
| 05-update-available | Refresh moved the marketplace commit; the tracked install shows "Update available", Update clears it |
| 06-acme-untrusted | An untrusted personal marketplace: read error listed, "Runs code" entries cannot be installed, "Also in openhands-extensions" on the duplicate skill |
| 07-acme-trusted-installed | Trusted toggled on: the plugin installs and gets an enable toggle |
| 08-add-form | Add marketplace form (source, name, ref, token secret, trusted) |
| 09-superpowers | `github:obra/superpowers` added and opened: 4 entries |
| 10-add-private-error | A private source without a token is refused and the form stays open |
| 11-list-after-remove | superpowers removed; after a reload the acme trust flag survives |

No agent-server is involved: the `/api/marketplaces` endpoints exist only in
`src/mocks/marketplaces-handlers.ts`. The 502s in the console are mock mode
proxying endpoints this branch does not mock (plugins, skills) to the
nonexistent agent-server, as on `main`.
