#!/usr/bin/env node
// Tiny static server for build/ — per-request fs lookup, SPA fallback to
// index.html. Replacement for sirv-cli, whose boot scan served 404s for
// freshly rebuilt assets on this machine.
import { createServer } from "node:http";
import { statSync, readFileSync, existsSync } from "node:fs";
import { join, extname, normalize } from "node:path";

const ROOT = process.argv[2] ?? join(process.cwd(), "build");
const PORT = Number(process.env.STATIC_PORT ?? 18300);
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  ".woff": "font/woff",
  ".map": "application/json",
  ".txt": "text/plain",
  ".webmanifest": "application/manifest+json",
};

createServer((req, res) => {
  try {
    const url = new URL(req.url, "http://127.0.0.1");
    let path = decodeURIComponent(url.pathname);
    let file = join(ROOT, normalize(path).replace(/^([/\\])+/, ""));
    if (!file.startsWith(ROOT)) file = join(ROOT, "index.html");
    if (!existsSync(file) || statSync(file).isDirectory()) {
      // SPA fallback: extension-less routes get index.html; missing assets 404.
      if (extname(path) === "" || path === "/") file = join(ROOT, "index.html");
      else {
        res.writeHead(404, { "Content-Type": "text/plain" });
        return res.end("not found");
      }
    }
    const body = readFileSync(file);
    res.writeHead(200, {
      "Content-Type": MIME[extname(file).toLowerCase()] ?? "application/octet-stream",
      "Cache-Control": "no-store",
    });
    res.end(body);
  } catch (error) {
    res.writeHead(500, { "Content-Type": "text/plain" });
    res.end(String(error));
  }
}).listen(PORT, "127.0.0.1", () => {
  console.log(`[static] serving ${ROOT} on http://127.0.0.1:${PORT}`);
});
