/// <reference types="vitest" />
/// <reference types="vite-plugin-svgr/client" />
import { defineConfig, loadEnv } from "vite";
import viteTsconfigPaths from "vite-tsconfig-paths";
import svgr from "vite-plugin-svgr";
import { reactRouter } from "@react-router/dev/vite";
import { configDefaults } from "vitest/config";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig(({ mode }) => {
  const {
    VITE_BACKEND_HOST = "127.0.0.1:3000",
    VITE_USE_TLS = "false",
    VITE_FRONTEND_PORT = "3001",
    VITE_INSECURE_SKIP_VERIFY = "false",
  } = loadEnv(mode, process.cwd());

  const USE_TLS = VITE_USE_TLS === "true";
  const INSECURE_SKIP_VERIFY = VITE_INSECURE_SKIP_VERIFY === "true";
  const PROTOCOL = USE_TLS ? "https" : "http";
  const WS_PROTOCOL = USE_TLS ? "wss" : "ws";

  const API_URL = `${PROTOCOL}://${VITE_BACKEND_HOST}/`;
  const WS_URL = `${WS_PROTOCOL}://${VITE_BACKEND_HOST}/`;
  const FE_PORT = Number.parseInt(VITE_FRONTEND_PORT, 10);

  return {
    plugins: [
      !process.env.VITEST && reactRouter(),
      viteTsconfigPaths(),
      svgr(),
      tailwindcss(),
    ],
    optimizeDeps: {
      include: [
        // Pre-bundle ALL dependencies to prevent runtime optimization and page reloads
        // These are discovered during initial app load:
        "posthog-js",
        "@tanstack/react-query",
        "react-hot-toast",
        "i18next",
        "i18next-http-backend",
        "i18next-browser-languagedetector",
        "react-i18next",
        "axios",
        "@uidotdev/usehooks",
        "react-icons/fa6",
        "react-icons/fa",
        "clsx",
        "tailwind-merge",
        "@heroui/react",
        "lucide-react",
        "@microlink/react-json-view",
        "socket.io-client",
        // These are discovered when launching conversations:
        "react-icons/vsc",
        "react-icons/lu",
        "react-icons/di",
        "react-icons/io5",
        "react-icons/io", // Added to prevent runtime optimization
        "@monaco-editor/react",
        "react-textarea-autosize",
        "react-markdown",
        "remark-gfm",
        "remark-breaks",
        "react-syntax-highlighter",
        "react-syntax-highlighter/dist/esm/styles/prism",
        "react-syntax-highlighter/dist/esm/styles/hljs",
        // Terminal dependencies - added to prevent runtime optimization
        "@xterm/addon-fit",
        "@xterm/xterm",
        "@xterm/xterm/css/xterm.css",
      ],
    },
    server: {
      port: FE_PORT,
      host: true,
      allowedHosts: true,
      proxy: {
        "/api": {
          target: API_URL,
          changeOrigin: true,
          secure: !INSECURE_SKIP_VERIFY,
        },
        "/ws": {
          target: WS_URL,
          ws: true,
          changeOrigin: true,
          secure: !INSECURE_SKIP_VERIFY,
        },
        "/socket.io": {
          target: WS_URL,
          ws: true,
          changeOrigin: true,
          secure: !INSECURE_SKIP_VERIFY,
          // rewriteWsOrigin: true,
        },
      },
      watch: {
        ignored: ["**/node_modules/**", "**/.git/**"],
      },
    },
    ssr: {
      noExternal: ["react-syntax-highlighter"],
    },
    clearScreen: false,
    test: {
      environment: "jsdom",
      setupFiles: ["vitest.setup.ts"],
      exclude: [...configDefaults.exclude, "tests"],
      coverage: {
        reporter: ["text", "json", "html", "lcov", "text-summary"],
        reportsDirectory: "coverage",
        include: ["src/**/*.{ts,tsx}"],
      },
    },
  };
});
