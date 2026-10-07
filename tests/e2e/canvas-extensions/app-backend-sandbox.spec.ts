import { createHash } from "node:crypto";
import http, { type IncomingMessage } from "node:http";
import type { Duplex } from "node:stream";
import { test, expect } from "@playwright/test";

const SESSION_COOKIE = "app_session=browser-test";
let appOrigin: string;

type RequestRecord = {
  kind: "http" | "websocket";
  path: string;
  origin: string | null;
  cookie: string | null;
};

const requests: RequestRecord[] = [];
const sockets = new Set<Duplex>();
let appServer: http.Server;

function recordRequest(kind: RequestRecord["kind"], request: IncomingMessage) {
  requests.push({
    kind,
    path: request.url ?? "",
    origin: request.headers.origin ?? null,
    cookie: request.headers.cookie ?? null,
  });
}

test.beforeAll(async () => {
  appServer = http.createServer((request, response) => {
    recordRequest("http", request);
    if (request.url === "/app") {
      response.setHeader("content-type", "text/html");
      response.end(`<!doctype html><script>
        const results = {};
        Promise.all([
          fetch('/api', { method: 'POST', credentials: 'include' }).then(async response => {
            results.fetch = { ok: response.ok, body: await response.text() };
          }),
          new Promise(resolve => {
            const socket = new WebSocket('${appOrigin.replace("http", "ws")}/socket');
            socket.onopen = () => { results.websocket = 'open'; socket.close(); resolve(); };
            socket.onerror = () => { results.websocket = 'error'; resolve(); };
          }),
          new Promise(resolve => {
            const worker = new Worker('/worker.js');
            worker.onmessage = event => { results.worker = event.data; worker.terminate(); resolve(); };
            worker.onerror = event => { results.worker = 'error:' + event.message; resolve(); };
          }),
        ]).then(() => parent.postMessage(results, '*'));
      </script>`);
      return;
    }
    if (request.url === "/api") {
      response.end("api-ok");
      return;
    }
    if (request.url === "/worker.js") {
      response.setHeader("content-type", "application/javascript");
      response.end("postMessage('worker-ok')");
      return;
    }
    response.writeHead(404).end();
  });
  appServer.on("upgrade", (request, socket) => {
    sockets.add(socket);
    socket.once("close", () => sockets.delete(socket));
    recordRequest("websocket", request);
    const key = request.headers["sec-websocket-key"];
    if (typeof key !== "string") {
      socket.destroy();
      return;
    }
    const accept = createHash("sha1")
      .update(`${key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`)
      .digest("base64");
    socket.write(
      `HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`,
    );
  });
  await new Promise<void>((resolve) =>
    appServer.listen(0, "127.0.0.1", resolve),
  );
  const address = appServer.address();
  if (!address || typeof address === "string") {
    throw new Error("App backend browser test failed to bind a TCP port.");
  }
  appOrigin = `http://localhost:${address.port}`;
});

test.afterAll(async () => {
  for (const socket of sockets) socket.destroy();
  await new Promise<void>((resolve, reject) =>
    appServer.close((error) => (error ? reject(error) : resolve())),
  );
});

test("the isolated app ingress retains origin for authenticated browser APIs", async ({
  browserName,
  context,
  page,
}) => {
  test.skip(
    browserName !== "chromium",
    "App sandbox browser behavior is verified in Chromium.",
  );
  requests.length = 0;
  await context.addCookies([
    {
      name: "app_session",
      value: "browser-test",
      url: appOrigin,
      sameSite: "Lax",
    },
  ]);
  await page.goto("/");
  const resultsPromise = page.evaluate(
    () =>
      new Promise<Record<string, unknown>>((resolve) => {
        window.addEventListener("message", (event) => resolve(event.data), {
          once: true,
        });
      }),
  );
  await page.evaluate(
    ({ appOrigin }) => {
      const frame = document.createElement("iframe");
      frame.sandbox.add(
        "allow-forms",
        "allow-modals",
        "allow-popups",
        "allow-same-origin",
        "allow-scripts",
      );
      frame.src = `${appOrigin}/app`;
      document.body.append(frame);
    },
    { appOrigin: appOrigin },
  );

  await expect(resultsPromise).resolves.toEqual({
    fetch: { ok: true, body: "api-ok" },
    websocket: "open",
    worker: "worker-ok",
  });
  const browserApiRequests = requests
    .filter(({ path }) => ["/api", "/socket", "/worker.js"].includes(path))
    .sort((left, right) => left.path.localeCompare(right.path));
  expect(browserApiRequests).toEqual([
    {
      kind: "http",
      path: "/api",
      origin: appOrigin,
      cookie: SESSION_COOKIE,
    },
    {
      kind: "websocket",
      path: "/socket",
      origin: appOrigin,
      cookie: SESSION_COOKIE,
    },
    {
      kind: "http",
      path: "/worker.js",
      origin: null,
      cookie: SESSION_COOKIE,
    },
  ]);
});
