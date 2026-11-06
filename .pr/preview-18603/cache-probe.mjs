import { createRequire } from "node:module";
import { readFile, writeFile, mkdtemp, utimes, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
const repoRoot = fileURLToPath(new URL("../../", import.meta.url));
const require = createRequire(join(repoRoot, "package.json"));
const run = process.argv[2];
const output = process.argv[3];
if (!run || !output)
  throw new Error(
    "Usage: node .pr/preview-18603/cache-probe.mjs <isolated verification run> <result.json>",
  );
const { chromium } = require("playwright");
const { build } = require("esbuild");
const { baseUrl: base } = JSON.parse(
  await readFile(join(run, "run.json"), "utf8"),
);
if (!["localhost", "127.0.0.1", "[::1]"].includes(new URL(base).hostname))
  throw new Error("Only an isolated loopback verification run is supported");
const key = (await readFile(join(run, "private/session-key"), "utf8")).trim();
const dir = await mkdtemp(join(tmpdir(), "do-18603-cache-owned-"));
const path = join(dir, "cache-probe.png");
const first = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=",
  "base64",
);
const second = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
  "base64",
);
await writeFile(path, first);
await utimes(
  path,
  new Date("2020-01-01T00:00:00Z"),
  new Date("2020-01-01T00:00:00Z"),
);
const bundle = await build({
  stdin: {
    contents:
      'export { FileClient } from "@openhands/typescript-client/clients";',
    resolveDir: repoRoot,
  },
  bundle: true,
  write: false,
  format: "iife",
  globalName: "cacheProbeSDK",
  platform: "browser",
  external: ["node:*"],
  logLevel: "silent",
});
const browser = await chromium.launch({ headless: true });
try {
  const context = await browser.newContext();
  const page = await context.newPage();
  const cdp = await context.newCDPSession(page);
  await cdp.send("Network.enable");
  const network = [];
  const cacheIds = new Set();
  cdp.on("Network.requestServedFromCache", (e) => cacheIds.add(e.requestId));
  cdp.on("Network.responseReceived", (e) => {
    if (!e.response.url.includes("/api/file/download")) return;
    const h = Object.fromEntries(
      Object.entries(e.response.headers).filter(([name]) =>
        [
          "cache-control",
          "last-modified",
          "etag",
          "content-type",
          "content-disposition",
        ].includes(name.toLowerCase()),
      ),
    );
    network.push({
      requestId: e.requestId,
      status: e.response.status,
      fromDiskCache: e.response.fromDiskCache ?? false,
      fromServiceWorker: e.response.fromServiceWorker ?? false,
      headers: h,
    });
  });
  await page.goto(base + "/health");
  await page.addScriptTag({ content: bundle.outputFiles[0].text });
  const download = () =>
    page.evaluate(
      async ({ base, key, path }) => {
        const file = new window.cacheProbeSDK.FileClient({
          host: base,
          apiKey: key,
        });
        try {
          const buffer = await file.downloadFile(path);
          return [...new Uint8Array(buffer)];
        } finally {
          file.close();
        }
      },
      { base, key, path },
    );
  const a = Buffer.from(await download());
  await writeFile(path, second);
  await utimes(path, new Date(), new Date());
  const b = Buffer.from(await download());
  const invalid = await page.evaluate(
    async ({ base, path }) => {
      try {
        const file = new window.cacheProbeSDK.FileClient({
          host: base,
          apiKey: "invalid-cache-probe",
        });
        return {
          ok: true,
          bytes: [...new Uint8Array(await file.downloadFile(path))],
        };
      } catch (error) {
        return { ok: false, status: error.status };
      }
    },
    { base, path },
  );
  const directInvalid = await fetch(
    base + "/api/file/download?path=" + encodeURIComponent(path),
    { headers: { "X-Session-API-Key": "invalid-cache-probe" } },
  );
  const direct = await fetch(
    base + "/api/file/download?path=" + encodeURIComponent(path),
    { headers: { "X-Session-API-Key": key } },
  );
  const truth = Buffer.from(await direct.arrayBuffer());
  const hash = (v) => createHash("sha256").update(v).digest("hex");
  const result = {
    base,
    client: JSON.parse(
      await readFile(
        join(
          repoRoot,
          "node_modules/@openhands/typescript-client/package.json",
        ),
        "utf8",
      ),
    ).version,
    firstHash: hash(a),
    secondHash: hash(b),
    actualNewBytesHash: hash(truth),
    expectedNewBytesHash: hash(second),
    staleSecondRead: a.equals(b) && !b.equals(truth),
    invalidKeyBrowserRead: {
      ok: invalid.ok,
      status: invalid.status,
      oldBytesReturned: invalid.bytes
        ? Buffer.from(invalid.bytes).equals(a)
        : false,
    },
    invalidKeyUncachedStatus: directInvalid.status,
    network: network.map((e) => ({
      ...e,
      servedFromCache: cacheIds.has(e.requestId),
    })),
    requestPolicy:
      "No routing, no global headers, cache left enabled; exact bundled FileClient.downloadFile in own Chromium context.",
  };
  await writeFile(output, JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser.close();
  await rm(dir, { recursive: true, force: true });
}
