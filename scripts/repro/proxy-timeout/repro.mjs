// node repro.mjs /path/to/checkout-or-installed-package clean|warning
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { Agent, createServer, get } from "node:http";
import { resolve } from "node:path";

const root = resolve(process.argv[2] || ".");
const expected = process.argv[3] || "clean";
assert.ok(["clean", "warning"].includes(expected));
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
const backend = createServer((_req, res) => res.end("ok"));
backend.listen(0, "127.0.0.1");
await once(backend, "listening");
const reserve = createServer();
reserve.listen(0, "127.0.0.1");
await once(reserve, "listening");
const port = reserve.address().port;
await new Promise((done) => reserve.close(done));
const args = [
  resolve(root, "scripts/ingress.mjs"),
  "--port",
  String(port),
  "--default",
  `http://127.0.0.1:${backend.address().port}`,
];
console.log("spawn", JSON.stringify([process.execPath, ...args]));
const child = spawn(process.execPath, args, {
  cwd: root,
  stdio: ["ignore", "pipe", "pipe"],
});
const exited = once(child, "close");
let output = "";
let errors = "";
child.stdout.on("data", (chunk) => {
  output += chunk;
  process.stdout.write(chunk);
});
child.stderr.on("data", (chunk) => {
  errors += chunk;
  process.stderr.write(chunk);
});
const agent = new Agent({ keepAlive: true, maxSockets: 1 });
const sockets = new Set();
const deadline = setTimeout(() => child.kill("SIGKILL"), 15_000);
try {
  while (!output.includes("Listening on:")) {
    assert.equal(child.exitCode, null, "ingress exited before listening");
    await Promise.race([
      once(child.stdout, "data"),
      exited.then(() => {
        throw new Error("ingress exited");
      }),
    ]);
  }
  for (let i = 0; i < 20; i++) {
    await new Promise((done, reject) => {
      const req = get(
        `http://127.0.0.1:${port}/repro-${i}`,
        { agent },
        (res) => {
          let body = "";
          res.on("data", (chunk) => {
            body += chunk;
          });
          res.on("error", reject);
          res.on("end", () => {
            try {
              assert.equal(res.statusCode, 200);
              assert.equal(body, "ok");
              done();
            } catch (error) {
              reject(error);
            }
          });
        },
      );
      req.on("socket", (socket) => sockets.add(socket));
      req.on("error", reject);
    });
  }
  assert.equal(sockets.size, 1, "must reuse one actual TCP connection");
  // Flush child stderr by awaiting process exit before inspecting warnings.
  child.kill("SIGTERM");
  await exited;
  const warning = errors.includes("MaxListenersExceededWarning");
  console.log(
    JSON.stringify({
      requests: 20,
      connections: sockets.size,
      warning,
      expected,
    }),
  );
  assert.equal(warning, expected === "warning");
} finally {
  clearTimeout(deadline);
  if (child.exitCode === null && child.signalCode === null) {
    child.kill("SIGKILL");
    await exited;
  }
  agent.destroy();
  backend.closeAllConnections();
  await new Promise((done) => backend.close(done));
}
