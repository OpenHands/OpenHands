import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// Minimal config so `vite-node` can run the live ACP e2e script outside the
// app's full Vite/React-Router pipeline. We only need the `#/*` → `src/*` path
// alias (the app resolves it via tsconfig-paths, which vite-node doesn't load)
// and to inline the typescript-client so its ESM resolves the same way Vitest
// configures it.
const srcDir = fileURLToPath(new URL("../../../src", import.meta.url));

export default defineConfig({
  // `agent-server-adapter.ts` reads this Vite `define` (injected by the app's
  // vite.config.ts). Without it the module throws `__EXTENSIONS_SKILLS_DIR__ is
  // not defined` the moment the e2e imports it. Empty string is the same value
  // library builds use, and the adapter falls back to "public" when it is falsy.
  define: {
    __EXTENSIONS_SKILLS_DIR__: JSON.stringify(""),
  },
  resolve: {
    alias: [{ find: /^#\//, replacement: `${srcDir}/` }],
  },
  ssr: {
    noExternal: ["@openhands/typescript-client"],
  },
});
