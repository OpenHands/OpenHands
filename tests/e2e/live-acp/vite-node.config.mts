import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// Minimal config so `vite-node` can run the live ACP e2e script outside the
// app's full Vite/React-Router pipeline. We only need the `#/*` → `src/*` path
// alias (the app resolves it via tsconfig-paths, which vite-node doesn't load)
// and to inline the typescript-client so its ESM resolves the same way Vitest
// configures it.
const srcDir = fileURLToPath(new URL("../../../src", import.meta.url));

const _require = createRequire(import.meta.url);
let extensionsSkillsDir = "";
try {
  extensionsSkillsDir = resolve(
    dirname(_require.resolve("@openhands/extensions/package.json")),
    "skills",
  );
} catch {
  extensionsSkillsDir = "";
}

export default defineConfig({
  define: {
    __EXTENSIONS_SKILLS_DIR__: JSON.stringify(extensionsSkillsDir),
  },
  resolve: {
    alias: [{ find: /^#\//, replacement: `${srcDir}/` }],
  },
  ssr: {
    noExternal: ["@openhands/typescript-client"],
  },
});
