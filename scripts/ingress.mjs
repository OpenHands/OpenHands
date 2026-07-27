#!/usr/bin/env node
/**
 * Standalone Ingress / Reverse Proxy
 *
 * A minimal HTTP reverse proxy that routes requests to multiple backends
 * based on URL path. Completely independent of any backend implementation.
 *
 * Usage:
 *   node scripts/ingress.mjs [options]
 *   node scripts/ingress.mjs --port 8000 --route "/api/automation=http://localhost:18001" --route "/api=http://localhost:18000" --default "http://localhost:3001"
 *
 * Environment variables:
 *   INGRESS_PORT          - Port to listen on (default: 8000)
 *   INGRESS_ROUTES        - JSON object of path prefix -> backend URL
 *   INGRESS_DEFAULT       - Default backend for unmatched routes
 *
 * Route matching:
 *   - Routes are matched by longest prefix first
 *   - More specific routes take precedence (e.g., /api/automation before /api)
 */

import { createServer } from "node:http";
import process from "node:process";
import { pathToFileURL } from "node:url";

import {
  createProxyHandlers,
  createRouter,
  isBenignSocketError,
} from "./proxy-utils.mjs";

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    port: 8000,
    routes: {},
    defaultBackend: null,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "-p":
      case "--port":
        config.port = parseInt(args[++i], 10);
        break;
      case "-r":
      case "--route":
        // Format: "/path=http://host:port"
        const [path, url] = args[++i].split("=");
        config.routes[path] = url;
        break;
      case "-d":
      case "--default":
        config.defaultBackend = args[++i];
        break;
      case "-h":
      case "--help":
        showHelp();
        process.exit(0);
    }
  }

  return config;
}

function showHelp() {
  console.log(`
Standalone Ingress / Reverse Proxy

Routes HTTP requests to multiple backends based on URL path prefix.

USAGE:
  node scripts/ingress.mjs [options]

OPTIONS:
  -p, --port <port>           Port to listen on (default: 8000)
  -r, --route <path=url>      Add a route (can be repeated)
  -d, --default <url>         Default backend for unmatched routes
  -h, --help                  Show this help

ENVIRONMENT VARIABLES:
  INGRESS_PORT                Port to listen on
  INGRESS_ROUTES              JSON object: {"path": "url", ...}
  INGRESS_DEFAULT             Default backend URL

EXAMPLES:
  # Basic setup with agent server and automation
  node scripts/ingress.mjs \\
    --port 8000 \\
    --route "/api/automation=http://localhost:18001" \\
    --route "/api=http://localhost:18000" \\
    --route "/sockets=http://localhost:18000" \\
    --default "http://localhost:3001"

  # Using environment variables
  INGRESS_PORT=8000 \\
  INGRESS_ROUTES='{"/ api/automation":"http://localhost:18001","/api":"http://localhost:18000"}' \\
  INGRESS_DEFAULT="http://localhost:3001" \\
  node scripts/ingress.mjs

ROUTE MATCHING:
  Routes are sorted by path length (longest first), so more specific
  routes like /api/automation will match before /api.
`);
}

function buildConfig(args, env = process.env) {
  let routes = { ...args.routes };

  // Merge env routes
  if (env.INGRESS_ROUTES) {
    try {
      const envRoutes = JSON.parse(env.INGRESS_ROUTES);
      routes = { ...routes, ...envRoutes };
    } catch (e) {
      console.error("Failed to parse INGRESS_ROUTES:", e.message);
    }
  }

  return {
    port: args.port || parseInt(env.INGRESS_PORT, 10) || 8000,
    routes,
    defaultBackend: args.defaultBackend || env.INGRESS_DEFAULT || null,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Server
// ═══════════════════════════════════════════════════════════════════════════

export function startIngress(config) {
  const route = createRouter(config.routes, config.defaultBackend);
  const proxy = createProxyHandlers({ label: `ingress:${config.port}` });
  const uninstallDiagnostics = proxy.installDiagnostics();

  const server = createServer((req, res) => {
    const backend = route(req.url ?? "/");

    if (!backend) {
      res.writeHead(503);
      res.end("No backend configured for this route");
      return;
    }

    proxy.proxyHttp(req, res, backend);
  });

  // Handle WebSocket upgrades
  server.on("upgrade", (req, socket, head) => {
    const backend = route(req.url ?? "/");

    if (!backend) {
      socket.destroy();
      return;
    }

    proxy.proxyWebSocket(req, socket, head, backend);
  });

  // Built-in protection against malformed client requests that can otherwise
  // bubble up as unhandled errors on the underlying TCP socket.
  server.on("clientError", (err, socket) => {
    if (!isBenignSocketError(err)) {
      console.error("Client error:", err.message);
    }
    if (socket.writable) {
      socket.end("HTTP/1.1 400 Bad Request\r\n\r\n");
    } else {
      socket.destroy();
    }
  });
  server.on("close", uninstallDiagnostics);

  server.listen(config.port, () => {
    console.log("");
    console.log(
      "╔═══════════════════════════════════════════════════════════════╗",
    );
    console.log(
      "║  Ingress Proxy                                                ║",
    );
    console.log(
      "╠═══════════════════════════════════════════════════════════════╣",
    );
    console.log(
      `║  Listening on: http://localhost:${config.port}/`.padEnd(66) + "║",
    );
    console.log(
      "╠═══════════════════════════════════════════════════════════════╣",
    );
    console.log(
      "║  Routes:                                                      ║",
    );

    const sortedRoutes = Object.entries(config.routes).sort(
      ([a], [b]) => b.length - a.length,
    );
    for (const [path, backend] of sortedRoutes) {
      const line = `    ${path} → ${backend}`;
      console.log(`║  ${line.padEnd(61)}║`);
    }

    if (config.defaultBackend) {
      const line = `    * (default) → ${config.defaultBackend}`;
      console.log(`║  ${line.padEnd(61)}║`);
    }

    console.log(
      "║                                                               ║",
    );
    console.log(
      "╚═══════════════════════════════════════════════════════════════╝",
    );
    console.log("");
  });

  return server;
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

const isMainModule =
  process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;

if (isMainModule) {
  const args = parseArgs();
  const config = buildConfig(args);

  if (Object.keys(config.routes).length === 0 && !config.defaultBackend) {
    console.error(
      "Error: No routes configured. Use --route or --default options.",
    );
    console.error("Run with --help for usage information.");
    process.exit(1);
  }

  startIngress(config);

  // Handle graceful shutdown
  process.on("SIGINT", () => {
    console.log("\nShutting down...");
    process.exit(0);
  });

  process.on("SIGTERM", () => {
    console.log("\nShutting down...");
    process.exit(0);
  });
}
