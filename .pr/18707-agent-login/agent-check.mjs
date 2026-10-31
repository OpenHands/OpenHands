// Event-socket logins an agent can use against a connected DO session.
//   node .pr/18707-agent-login/agent-check.mjs <canvasUrl> <key> <sessionId>
import WebSocket from "ws";

const [base, key, session] = process.argv.slice(2);
const url = `${base.replace(/^http/, "ws")}/mars/sessions/${session}/sockets/events/00000000000000000000000000000000`;

function socket(label, { header, firstMessage, waitMs = 15000 }) {
  return new Promise((resolve) => {
    const started = Date.now();
    let settled = false;
    const ws = new WebSocket(url, { headers: header ? { "X-Session-API-Key": header } : {} });
    const done = (outcome) => {
      if (settled) return;
      settled = true;
      console.log(`${label}: ${outcome} after ${((Date.now() - started) / 1000).toFixed(1)}s`);
      try { ws.terminate(); } catch {}
      resolve();
    };
    ws.on("unexpected-response", (_req, res) => done(`refused at upgrade HTTP ${res.statusCode}`));
    ws.on("open", () => { if (firstMessage) ws.send(JSON.stringify(firstMessage)); });
    // 4004 comes from the guest Agent Server (no such conversation): the bridge let it through.
    ws.on("close", (code, reason) => done(`closed ${code}${reason?.length ? ` (${reason})` : ""}`));
    ws.on("error", (e) => done(`error ${e.message}`));
    setTimeout(() => done("still open"), waitMs);
  });
}

await socket("key in header", { header: key });
await socket("wrong key in header", { header: "wrong-key" });
await socket("SDK first frame, right key", { firstMessage: { type: "auth", session_api_key: key } });
await socket("SDK first frame, wrong key", { firstMessage: { type: "auth", session_api_key: "wrong-key" } });
await socket("no header, no first frame (silent)", {});
