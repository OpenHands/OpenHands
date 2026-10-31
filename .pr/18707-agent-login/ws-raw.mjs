// Raw WebSocket client for #18707 review repros (no URL validation, any path).
//   node ws-raw.mjs <port> <path> <payload-kind> [key]
// payload-kind: auth  -> {"type":"auth","session_api_key":key}
//               big:N -> N MiB of junk text, no key
import { connect } from "node:net";
import { randomBytes } from "node:crypto";

const [port, path, kind, key = ""] = process.argv.slice(2);
const sock = connect(Number(port), "127.0.0.1");
const started = Date.now();
const t = () => `${((Date.now() - started) / 1000).toFixed(1)}s`;

function frame(text) {
  const payload = Buffer.from(text);
  const len = payload.length;
  const head = len < 126 ? Buffer.from([0x81, 0x80 | len])
    : len < 65536 ? Buffer.from([0x81, 0x80 | 126, len >> 8, len & 255])
    : (() => { const b = Buffer.alloc(10); b[0] = 0x81; b[1] = 0x80 | 127; b.writeBigUInt64BE(BigInt(len), 2); return b; })();
  const mask = randomBytes(4);
  for (let i = 0; i < len; i += 1) payload[i] ^= mask[i % 4];
  return Buffer.concat([head, mask, payload]);
}

let upgraded = false;
sock.on("connect", () => {
  sock.write(`GET ${path} HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n` +
    `Sec-WebSocket-Key: ${randomBytes(16).toString("base64")}\r\nSec-WebSocket-Version: 13\r\n\r\n`);
});
sock.on("data", (buf) => {
  if (!upgraded) {
    const status = buf.toString("latin1").split("\r\n")[0];
    console.log(`[${t()}] upgrade response: ${status}`);
    if (!status.includes("101")) return sock.end();
    upgraded = true;
    const text = kind === "auth" ? JSON.stringify({ type: "auth", session_api_key: key })
      : "x".repeat(Number(kind.split(":")[1]) * 1024 * 1024);
    sock.write(frame(text));
    console.log(`[${t()}] sent ${kind} frame (${text.length} bytes)`);
    return;
  }
  if ((buf[0] & 0x0f) === 0x8 && buf.length >= 4) console.log(`[${t()}] server closed with code ${buf.readUInt16BE(2)}`);
});
sock.on("close", () => { console.log(`[${t()}] socket closed`); process.exit(0); });
sock.on("error", (e) => { console.log(`[${t()}] error ${e.code}`); process.exit(0); });
setTimeout(() => { console.log(`[${t()}] still open after 15s`); sock.destroy(); }, 15000);
