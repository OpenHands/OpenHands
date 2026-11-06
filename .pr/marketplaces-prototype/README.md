# Marketplaces prototype: browser evidence

Canvas `main` at fc9f801f0e plus this branch, run with

```
VITE_FRONTEND_PORT=3177 VITE_BACKEND_BASE_URL=http://127.0.0.1:8000 \
VITE_SESSION_API_KEY=marketplaces-dev npm run dev:mock
```

and driven by `drive.mjs` (headless Chromium through the repo's Playwright;
copy the script under `node_modules/.cache/` and run
`node <copy> http://127.0.0.1:3177 <out-dir>` from the repo root). The script
seeds the onboarding/backend localStorage keys, dismisses the telemetry dialog,
and asserts each step; the screenshots are what it saw.

Layout: no Customize rail. The band at the top picks a marketplace and manages
it (trust, refresh, remove; add is in the page header). Tabs under it switch
between the kinds; below, the search bar, the facet menu and the cards follow
the Skills page. One card component serves every kind, shaped like the MCP
library and Skills cards: the icon slot shows the Skills badge, the MCP
integration logo, the template's manifest icon (or the logos of the
integrations it needs), or the kind's icon; the circled +/check control at the
top right installs an entry and switches an installed one on and off.

| Screenshot | What it shows |
| --- | --- |
| v2-01-default | The default marketplace: 190 entries from the bundled `@openhands/extensions` 0.41.0, tabs with counts, facets, MCP cards with their logos |
| v2-02-automation-tab | Automation templates tab (26) with manifest icons and `Needs:` pills |
| v2-03-installed-update | Status facet "Installed" after installing github; Refresh moved the commit and the tracked install shows "Update available" |
| v2-04-acme-untrusted | Picker switched to acme-internal: read error listed, Remove marketplace shown, "Runs code" entries blocked until trusted, "Also in openhands-extensions" on the duplicate skill |
| v2-05-acme-trusted | Trusted toggled on: the plugin installs and gets an enable toggle |
| v2-06-add-form | Add marketplace form (source, name, ref, token secret, trusted) |
| v2-07-superpowers | `github:obra/superpowers` added and selected: 4 entries |
| v2-08-picker-open | superpowers removed; the picker lists the remaining two |

No agent-server is involved: the `/api/marketplaces` endpoints exist only in
`src/mocks/marketplaces-handlers.ts`. The 502s in the console are mock mode
proxying endpoints this branch does not mock (plugins, skills) to the
nonexistent agent-server, as on `main`.
