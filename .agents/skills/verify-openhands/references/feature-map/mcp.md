# F06 — MCP catalog, configuration and connectivity

Install and test an integration through the real settings flow. Source:
[MCP route](../../../../src/routes/mcp.tsx),
[MCP UI](../../../../src/components/features/mcp-page),
[MCP spec](../../../../specs/mcp-settings.md).

## Sub-features

- `F06.catalog`: search/filter, installed/library states and server details.
- `F06.credential-layout`: install fields and native password concealment on desktop/mobile.
- `F06.lifecycle`: install/edit/remove a reviewed server; persistence and confirmation.
- `F06.connection`: actual connection test and tool listing, including failure feedback.
- `F06.external-auth`: real account/OAuth and save-as-secret behavior, when authorized.

## How to get to it (user POV)

Customize → MCP Servers or `/mcp`; select a marketplace card or Add custom server.
Reach an installed server's editor separately from the marketplace install dialog.

## Driving it with Playwright

Preconditions: approved local MCP package for a credential-free integration (Time
is a useful candidate); explicit test-account permission for external services.
Review executable package/source before installation; enabling MCP executes code.

- **Catalog/layout:** inspect `mcp-marketplace-grid`; click
  `mcp-marketplace-card-github`, fill `mcp-install-field-api_key` with a dummy and
  assert native `type="password"`. Use the [smoke helper](../scripts/smoke.mjs)
  for both viewport captures; do not click Install with the dummy credential.
- **Install:** choose the reviewed Time entry, inspect the command/configuration,
  use `mcp-install-submit`; expect the named `mcp-server-item` in
  `mcp-installed-list`, then reload to prove persistence.
- **Test/edit/remove:** open the installed server's `mcp-custom-editor`, use
  `mcp-test-connection` and assert a real successful tool list in the result—not
  just that the server appears installed. Change a safe option, reload, then delete
  with `mcp-custom-editor-delete` / `confirm-button`; verify absence after reload.
- **Errors/external auth:** exercise a deliberately unreachable local test endpoint
  and inspect useful error text. For authorized integrations, verify actual account
  access and whether save-as-secret stores the intended entry; clean up the fixture.

## Gotchas

A catalog card does not prove connectivity. Mock-LLM specs sometimes intercept MCP
endpoints: reuse selectors, never those fake responses. Never install every
marketplace integration as a blanket smoke test or use a production account by
inference. Preserve native input styling; no magenta screenshot overlays.
