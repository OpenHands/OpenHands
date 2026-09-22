/**
 * Reproduces the shared-origin cloud host: the enterprise app owns `/` and
 * Canvas is mounted at `/canvas`. Serving both from one origin is what makes a
 * bare `/conversations/...` link resolve into the wrong app.
 *
 * Also serves the MSW worker at the origin root, because MSW registers its
 * service worker with `/` scope regardless of the SPA base path.
 *
 *   node .pr/shared-origin-server.mjs --port 13001 --dir build-fixed
 */
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize } from "node:path";

const args = process.argv.slice(2);
const getArg = (flag, fallback) => {
  const i = args.indexOf(flag);
  return i === -1 ? fallback : args[i + 1];
};

const port = Number.parseInt(getArg("--port", "13001"), 10);
const dir = getArg("--dir", "build-fixed");

const MIME = {
  ".css": "text/css",
  ".html": "text/html",
  ".ico": "image/x-icon",
  ".js": "text/javascript",
  ".json": "application/json",
  ".map": "application/json",
  ".mjs": "text/javascript",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".webmanifest": "application/manifest+json",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".xml": "application/xml",
};

const enterpriseStub = `<!DOCTYPE html><html><head><title>Enterprise</title></head>
<body><h1 id="app">Enterprise app root</h1>
<p>This app owns <code>/</code>. A Canvas link without its base path lands here.</p>
</body></html>`;

async function sendFile(res, filePath) {
  const body = await readFile(filePath);
  res.writeHead(200, {
    "content-type": MIME[extname(filePath)] ?? "application/octet-stream",
  });
  res.end(body);
}

const server = createServer(async (req, res) => {
  const urlPath = decodeURIComponent(
    new URL(req.url, "http://localhost").pathname,
  );

  // MSW's service worker always registers at the origin root.
  if (urlPath === "/mockServiceWorker.js") {
    return sendFile(res, join(dir, "mockServiceWorker.js"));
  }

  // The enterprise app owns the origin root.
  if (urlPath === "/" || urlPath === "") {
    res.writeHead(200, { "content-type": "text/html" });
    return res.end(enterpriseStub);
  }

  if (urlPath === "/canvas" || urlPath.startsWith("/canvas/")) {
    const rel = normalize(urlPath.slice("/canvas".length)).replace(/^\/+/, "");
    const candidate = join(dir, rel);
    try {
      if ((await stat(candidate)).isFile()) return await sendFile(res, candidate);
    } catch {
      // fall through to SPA fallback
    }
    return sendFile(res, join(dir, "index.html"));
  }

  // Every other path belongs to the enterprise app, which is also an SPA and
  // therefore answers unknown routes with its own shell rather than a 404.
  res.writeHead(200, { "content-type": "text/html" });
  res.end(enterpriseStub);
});

server.listen(port, () => {
  console.log(
    `shared-origin server on http://localhost:${port}/  (enterprise at /, canvas at /canvas, dir=${dir})`,
  );
});