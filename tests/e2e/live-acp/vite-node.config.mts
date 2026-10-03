import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// Minimal config so `vite-node` can run the live ACP e2e script outside the
// app's full Vite/React-Router pipeline. We only need the `#/*` → `src/*` path
// alias (the app resolves it via tsconfig-paths, which vite-node doesn't load)
// and to inline the typescript-client so its ESM resolves the same way Vitest
// configures it.
const srcDir = fileURLToPath(new URL("../../../src", import.meta.url));

export default defineConfig({
  define: {
    // The production Vite config injects this compile-time constant. Live ACP
    // scripts run outside that pipeline and do not load extension skills.
    __EXTENSIONS_SKILLS_DIR__: JSON.stringify(""),
  },
  resolve: {
    alias: [{ find: /^#\//, replacement: `${srcDir}/` }],
  },
  ssr: {
    noExternal: ["@openhands/typescript-client"],
  },
});
