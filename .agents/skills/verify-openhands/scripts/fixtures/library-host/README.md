# Library host fixture

A minimal Vite 8 browser host that imports `@openhands/agent-canvas` by bare
package name, with React 19.3.0 peers, and renders a button inside
`AgentServerUIRoot`. `#outside` is a host element outside the Canvas scope, and
`#module-error` shows any native `window` error. There is no `require` shim and
no mocked module response.

Copy this directory to a scratch directory outside the checkout. Replace
`REPLACE_WITH_LIBRARY_BUILD_DIR` in `package.json` with the checkout copy where
`npm run build:lib` ran (`$L`), then run `npm install --ignore-scripts` and
`QA_CANVAS_PACKAGE_DIR=$L npx vite --port <owned-port>`.
[F26](../../../references/feature-map/F26-runtime-variants.md) (`F26.lib-host-app`) drives it.
