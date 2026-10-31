// A stand-in for DigitalOcean's harness API and port-forward tunnel, for a
// screenshot only: one OpenHands agent with one ready session, whose tunnel
// pipes to a local, keyless agent-server (as the guest runs). Logs each call.
import { createServer } from "node:http";
import { connect } from "node:net";
import { WebSocketServer } from "ws";

const PORT = Number(process.env.FAKE_PORT);
const GUEST = Number(process.env.GUEST_PORT);
const now = new Date().toISOString();
const session = {
  session_id: "sess_standin",
  name: "stand-in-session",
  status: "SESSION_STATUS_READY",
  config_id: "cfg_standin",
  agent_kind: "AGENT_KIND_UNSPECIFIED",
  created_at: now,
  last_event_at: now,
};
const config = { id: "cfg_standin", name: "standin-agent", updated_at: now };
const routes = {
  "GET /v2/agents/sessions": { sessions: [session], next_page_token: "" },
  "GET /v2/agents/sessions/sess_standin": { session },
  "GET /v2/agents/configs": { configs: [config], next_page_token: "" },
  "GET /v2/agents/configs/cfg_standin": { config: { ...config, manifest: { agent: "openhands", template: "openhands" } } },
  "GET /v2/agents/configs/cfg_standin/sessions": { sessions: [session], next_page_token: "" },
  "GET /v2/agents/sessions/sandbox/sizes": { sizes: [{ slug: "mars-2vcpu-4gb", vcpus: 2, memory_mb: 4096 }] },
  "POST /v2/agents/sessions/sess_standin/resume": {},
  "POST /v2/agents/sessions/sess_standin/pause": {},
  "GET /v2/account": { account: { email: "you@example.com", team: { uuid: "team-standin", name: "Stand-in team" } } },
};
const server = createServer((req, res) => {
  const path = new URL(req.url, "http://x").pathname;
  const body = routes[`${req.method} ${path}`];
  console.log(`HTTP ${req.method} ${path} -> ${body ? 200 : 404}`);
  res.writeHead(body ? 200 : 404, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body ?? { message: "not in the stand-in" }));
});
const wss = new WebSocketServer({ noServer: true });
server.on("upgrade", (req, socket, head) => {
  const match = /^\/v2\/agents\/sessions\/sess_standin\/port-forward\/(\d+)$/.exec(new URL(req.url, "http://x").pathname);
  console.log(`TUNNEL ${req.url} -> ${match ? "guest" : "refused"}`);
  if (!match) return socket.destroy();
  wss.handleUpgrade(req, socket, head, (ws) => {
    const guest = connect(GUEST, "127.0.0.1");
    guest.on("data", (chunk) => ws.readyState === ws.OPEN && ws.send(chunk));
    guest.on("close", () => ws.close());
    guest.on("error", () => ws.close(4002));
    ws.on("message", (data) => guest.write(data));
    ws.on("close", () => guest.destroy());
  });
});
server.listen(PORT, "127.0.0.1", () => console.log(`stand-in DigitalOcean on ${PORT}, guest ${GUEST}`));
