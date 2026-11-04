import { defineConfig } from "vite";
import { resolve } from "node:path";
import { realpathSync } from "node:fs";
if (!process.env.QA_CANVAS_PACKAGE_DIR) throw new Error("Set QA_CANVAS_PACKAGE_DIR to the built library checkout");
const packageDirectory = resolve(process.env.QA_CANVAS_PACKAGE_DIR);
export default defineConfig({
  server: { host: "127.0.0.1", strictPort: true, fs: { allow: [process.cwd(), packageDirectory, realpathSync(resolve(packageDirectory, "node_modules"))] } },
  optimizeDeps: { include: ["@openhands/agent-canvas"] },
  resolve: { dedupe: ["react", "react-dom", "react-router"] },
});
