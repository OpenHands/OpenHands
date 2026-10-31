// The Canvas server's /mars routes, with a stand-in for the DigitalOcean
// tunnel: session "sess_e2e" is "connected" to a local, keyless agent-server.
import { createServer } from "node:http";
import { createMarsWebBridge } from "../../scripts/mars-web-bridge.mjs";

const guestPort = Number(process.env.GUEST_PORT);
const bridge = {
  getAuthState: () => ({ connections: [], active: null }),
  getTunnel: (sessionId) =>
    sessionId === "sess_e2e" ? { sessionId, status: "connected", localPort: guestPort } : undefined,
  openTunnel: async ({ sessionId }) => ({ sessionId, status: "connected", host: `http://127.0.0.1:${guestPort}`, localPort: guestPort }),
  dispose: async () => {},
};
const marsWeb = createMarsWebBridge({ bridge, env: {}, key: process.env.CANVAS_KEY, log: console.warn });
const server = createServer((req, res) => {
  console.log(`HTTP ${req.method} ${req.url} key=${req.headers["x-session-api-key"] ? "yes" : "no"}`);
  if (!marsWeb.handleHttp(req, res)) res.writeHead(404).end();
});
server.on("upgrade", (req, socket, head) => {
  console.log(`UPGRADE ${req.url} key-header=${req.headers["x-session-api-key"] ? "yes" : "no"} cookie=${req.headers.cookie ? "yes" : "no"}`);
  if (!marsWeb.handleUpgrade(req, socket, head)) socket.destroy();
});
server.listen(Number(process.env.PROXY_PORT), "127.0.0.1", () => console.log("proxy ready"));
