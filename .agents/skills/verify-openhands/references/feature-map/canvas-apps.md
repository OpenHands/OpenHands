# F08 — Canvas apps and extension pages

Install a reviewed extension, explicitly trust it, and render its contributed page.
Source: [apps route](../../../../src/routes/canvas-extensions.tsx),
[extension route](../../../../src/routes/canvas-extension-page.tsx),
[manual guide](../../../../docs/CANVAS_EXTENSIONS_TESTING.md),
[bundled demo](../../../../src/fixtures/canvas-extensions/demo-page).

## Sub-features

- `F08.catalog`: empty/installed list and add dialog.
- `F08.lifecycle`: install, explicit trust/enable, disable and uninstall.
- `F08.page`: contributed navigation/direct route and unavailable fallback.

## How to get to it (user POV)

Customize → Apps (`/apps`); Add app, then the installed app's enable action.
Open its contributed entry and `/extensions/demo-page/hello` directly.

## Driving it with Playwright

Preconditions: a real backend with canvas-extension endpoints. Review the bundled
fixture and use its absolute path in this checkout; do not install arbitrary code.

- **Install:** click `canvas-extensions-add-button`; fill
  `add-canvas-extension-source-input` with the absolute bundled demo path and submit
  `add-canvas-extension-submit`. Verify installed but disabled state after reload.
- **Trust:** invoke Enable, inspect the “Enable trusted app” confirmation, and approve
  only this reviewed fixture. Verify enabled state and its new navigation entry.
- **Page:** follow that entry and the direct extension URL separately. Assert the
  real “Hello from an extension” main content, not merely an iframe/container.
  Capture desktop/mobile and check browser errors.
- **Disable/remove:** disable the app, verify its unavailable fallback and removed
  navigation, then uninstall and verify the list after reload.

## Gotchas

The mock-LLM extension spec can serve a fixture endpoint when its backend pin lacks
extension support. Do not copy that interception: missing real endpoints are a
blocked capability, not live success. Enabling an extension is code execution;
installation alone is not trust or proof that its UI renders.
