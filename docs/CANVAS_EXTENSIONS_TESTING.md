# Canvas Extensions manual testing

Canvas Extensions can be exercised locally before the Agent Server implements
the `/api/canvas-extensions` endpoints. Canvas's existing MSW development mode
contains an in-memory implementation of the API and serves the checked-in demo
extension bundle through the same frontend service and runtime used in a real
deployment.

This path is for frontend development only. It does not test Agent Server
installation, filesystem validation, persistence, authentication, or Git
resolution.

## Start the mock frontend

From the repository root, run:

```sh
VITE_FRONTEND_PORT=3102 \
VITE_BACKEND_BASE_URL=http://127.0.0.1:8000 \
VITE_SESSION_API_KEY=canvas-extension-dev \
npm run dev:mock
```

Port `3102` avoids the `3001` Vite process used by the normal local stack. The
backend URL only gives Canvas a local backend identity; MSW intercepts the
extension requests in the browser. The mock also covers the settings and server
information probes needed to mark that backend healthy, so the Agent Server
does not need the extension endpoints and does not need to be running.

Open <http://localhost:3102/extensions>. Do not use the normal ingress URL at
`http://localhost:8000` for this test because its `/api` traffic goes directly
to the unmodified Agent Server rather than through the mock browser session.

If the browser profile already contains incompatible backend or onboarding
state, use a private window or clear local storage for `localhost:3102` and
reload.

## Install and enable the fixture

1. In **Customize -> Extensions**, select **Add extension**.
2. Enter this exact source:

   ```text
   src/fixtures/canvas-extensions/demo-page
   ```

3. Leave **Ref** and **Repository path** empty, then select **Install**.
4. Confirm that **Demo page** appears disabled. Installation must not execute
   the bundle or add its navigation item.
5. Turn on the extension and accept the trusted-code confirmation.
6. Confirm that **Extension demo** appears in the main left rail.
7. Open it and verify the page says **Hello from a Canvas Extension**.
8. Visit `/extensions/demo-page/hello/nested` directly and verify the page
   renders `Nested extension path: nested`.

## Lifecycle checks

- **Disable:** turn the extension off. Its rail item should disappear, and its
  route should no longer render the contributed page.
- **Re-enable:** turn it on again. The item and page should return without a
  Canvas restart.
- **Uninstall:** select **Uninstall** and confirm. The inventory and rail item
  should become empty.
- **Reload:** reload the page and confirm the installation and enablement are
  retained for this browser tab. The mock uses session storage and clears when
  you uninstall it or end the browser session.

## Test an extension edit

Edit
`src/fixtures/canvas-extensions/demo-page/extension.js`, restart the mock
frontend if Vite does not rebuild the raw fixture import automatically, then
uninstall and reinstall the fixture. This allows page mounting, cleanup,
subrouting, and use of the host API to be developed before backend support is
available.

The fixture must remain a self-contained browser ES module: it may not rely on
bare package imports or additional output chunks.

## What still requires the Agent Server

Repeat this flow against `http://localhost:8000/extensions` after the backend
contract lands. That test must additionally verify:

- Git and backend-local-path installation;
- immutable revision resolution;
- manifest, traversal, symlink, and entrypoint validation;
- persistence across Agent Server and browser restarts;
- session-authenticated bundle delivery;
- isolation when switching between active backends.

The backend contract and acceptance criteria are documented in
[`specs/canvas-extensions.md`](../specs/canvas-extensions.md).
