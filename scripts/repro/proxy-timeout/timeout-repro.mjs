// Exercise the real helper with an unresponsive upstream and a short deadline.
// The ingress CLI uses 120s; the helper option keeps this check quick.
import assert from "node:assert/strict";
import { once } from "node:events";
import { createServer, get } from "node:http";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const root = resolve(process.argv[2] || ".");
console.log(
  JSON.stringify({
    time: new Date().toISOString(),
    node: process.version,
    cwd: process.cwd(),
    root,
    argv: process.argv,
    pid: process.pid,
  }),
);
const { createProxyHandlers } = await import(
  pathToFileURL(resolve(root, "scripts/proxy-utils.mjs"))
);
const backend = createServer(() => {});
backend.listen(0, "127.0.0.1");
await once(backend, "listening");
const handlers = createProxyHandlers({ timeout: 300, proxyTimeout: 0 });
const server = createServer((req, res) =>
  handlers.proxyHttp(req, res, `http://127.0.0.1:${backend.address().port}`),
);
server.listen(0, "127.0.0.1");
await once(server, "listening");
try {
  const started = performance.now();
  const code = await new Promise((done, reject) => {
    const req = get(`http://127.0.0.1:${server.address().port}/hang`, {
      agent: false,
    });
    const deadline = setTimeout(() => {
      req.destroy();
      reject(new Error("downstream deadline did not fire"));
    }, 5000);
    req.once("error", (err) => {
      clearTimeout(deadline);
      done(err.code);
    });
    req.once("response", (res) => {
      clearTimeout(deadline);
      res.resume();
      reject(new Error("unexpected response"));
    });
  });
  const elapsedMs = Math.round(performance.now() - started);
  console.log(JSON.stringify({ timeoutMs: 300, elapsedMs, code }));
  assert.equal(code, "ECONNRESET");
  assert.ok(elapsedMs >= 250 && elapsedMs < 5000);
} finally {
  server.closeAllConnections();
  backend.closeAllConnections();
  await Promise.all([
    new Promise((done) => server.close(done)),
    new Promise((done) => backend.close(done)),
  ]);
}
