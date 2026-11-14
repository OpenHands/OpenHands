#!/usr/bin/env node
// Device-flow login helper for PR #18913 real-cloud evidence.
//
// Runs the OAuth 2.0 device flow against the real OpenHands Cloud, prints the
// verification URL + user code, waits for the user to approve in their
// browser, then writes the resulting backend session to `.pr/cloud-session.json`
// for `evidence-driver.mjs` to consume. The in-flight `device_code` is saved to
// `.pr/cloud-session.pending.json` so a crashed poll can be resumed.
//
// Run:
//   node .pr/device-flow-login.mjs [--host https://app.all-hands.dev]
//
// Emits `.pr/cloud-session.json`:
//   { "host": "https://app.all-hands.dev", "apiKey": "...", "orgId": "...", "name": "OpenHands Cloud" }

import { writeFileSync, readFileSync, existsSync, unlinkSync } from "node:fs";
import { join } from "node:path";
import { EnvHttpProxyAgent } from "undici";

// The sandbox routes outbound traffic through a local proxy (HTTPS_PROXY);
// node's fetch (undici) ignores it by default, so route explicitly.
const dispatcher = new EnvHttpProxyAgent();

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, arg, i, arr) => {
    if (arg.startsWith("--")) acc.push([arg.slice(2), arr[i + 1]]);
    return acc;
  }, []),
);
const HOST = (args.host ?? "https://app.all-hands.dev").replace(/\/+$/, "");
const OUT = join(".pr", "cloud-session.json");
const PENDING = join(".pr", "cloud-session.pending.json");

// Same coarse client headers the Canvas sends (see src/api/client-source.ts).
const CLIENT_HEADERS = {
  "Content-Type": "application/json",
  "X-OpenHands-Client": "agent_canvas",
  "X-OpenHands-Client-Version": "0.0.0-evidence",
};

// Resume an in-flight flow if the previous run crashed mid-poll.
let flow = null;
if (existsSync(PENDING)) {
  try {
    flow = JSON.parse(readFileSync(PENDING, "utf8"));
    console.log(`[device-flow] 恢复未完成的设备流（${flow.user_code}）...`);
  } catch {
    flow = null;
  }
}

async function startFlow() {
  const res = await fetch(`${HOST}/oauth/device/authorize`, {
    method: "POST",
    headers: CLIENT_HEADERS,
    dispatcher,
  });
  if (!res.ok) {
    throw new Error(`device authorize failed: ${res.status} ${await res.text()}`);
  }
  const body = await res.json();
  const required = ["device_code", "user_code", "verification_uri"];
  for (const key of required) {
    if (!body[key]) throw new Error(`authorize response missing "${key}"`);
  }
  return { ...body, host: HOST };
}

async function pollToken(deviceCode) {
  const params = new URLSearchParams({
    grant_type: "urn:ietf:params:oauth:grant-type:device_code",
    device_code: deviceCode,
  });
  try {
    const res = await fetch(`${HOST}/oauth/device/token`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: params.toString(),
      dispatcher,
    });
    if (res.ok) {
      const body = await res.json();
      if (body.access_token) return body.access_token;
    }
  } catch {
    // Network blip — keep polling, never crash the run.
  }
  return null;
}

async function fetchOrgId(apiKey) {
  const res = await fetch(`${HOST}/api/organizations`, {
    headers: { Authorization: `Bearer ${apiKey}` },
    dispatcher,
  });
  if (!res.ok) {
    throw new Error(`organizations failed: ${res.status} ${await res.text()}`);
  }
  const body = await res.json();
  return body.current_org_id ?? body.items?.[0]?.id ?? null;
}

if (!flow) {
  flow = await startFlow();
  writeFileSync(PENDING, JSON.stringify(flow, null, 2) + "\n");
}

const url =
  flow.verification_uri_complete ??
  `${flow.verification_uri}?user_code=${encodeURIComponent(flow.user_code)}`;
const interval = Math.max(2, Math.min(flow.interval ?? 5, 10));

console.log("\n=== OpenHands Cloud 设备流登录 ===\n");
console.log(`1. 打开这个链接（浏览器）并登录 GitHub：\n   ${url}\n`);
console.log(`2. 如果没自动带入，手动输入验证码：${flow.user_code}\n`);
console.log(`3. 授权后回到这里，我会自动轮询取 token（${interval}s 间隔）...\n`);

const deadline = Date.now() + (flow.expires_in ?? 600) * 1000;
let apiKey = null;
process.stdout.write("等待授权");
while (Date.now() < deadline) {
  apiKey = await pollToken(flow.device_code);
  if (apiKey) break;
  process.stdout.write(".");
  await new Promise((r) => setTimeout(r, interval * 1000));
}
process.stdout.write("\n");

if (!apiKey) {
  console.error("\n✗ 设备流超时/未授权，请重跑。");
  process.exit(1);
}

console.log("✓ 已拿到 API key，正在读取组织信息...");
const orgId = await fetchOrgId(apiKey);

const session = {
  host: HOST,
  apiKey,
  orgId,
  name: "OpenHands Cloud",
};
writeFileSync(OUT, JSON.stringify(session, null, 2) + "\n");
if (existsSync(PENDING)) unlinkSync(PENDING);
console.log(`\n✓ 会话已写入 ${OUT}`);
console.log(JSON.stringify({ host: session.host, orgId: session.orgId }, null, 2));
