import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// Minimal config so `vite-node` can run the live ACP e2e script outside the
// app's full Vite/React-Router pipeline. We only need the `#/*` → `src/*` path
// alias (the app resolves it via tsconfig-paths, which vite-node doesn't load)
// and to inline the typescript-client so its ESM resolves the same way Vitest
// configures it.
const srcDir = fileURLToPath(new URL("../../../src", import.meta.url));

export default defineConfig({
  resolve: {
    alias: [{ find: /^#\//, replacement: `${srcDir}/` }],
  },
  ssr: {
    noExternal: ["@openhands/typescript-client"],
  },
});
