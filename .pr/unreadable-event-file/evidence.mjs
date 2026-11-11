// Evidence for #16806: what Canvas shows for a conversation whose event log
// has an unreadable file. No LLM is involved.
//
//   1. npm run dev
//   2. node .pr/unreadable-event-file/evidence.mjs seed http://127.0.0.1:8000 <session key>
//      (prints the conversation id; 40 user messages, the agent never runs)
//   3. Stop the stack, empty one event file of that conversation, e.g.
//        : > ~/.openhands/agent-canvas/dev_conversations/<id without dashes>/events/event-00041-*.json
//      and start the stack again (a running server answers from memory).
//   4. node .pr/unreadable-event-file/evidence.mjs shot http://127.0.0.1:8000 <session key> <id> after.png
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

const [command, origin, key, ...rest] = process.argv.slice(2);

if (command === "seed") {
  const call = async (method, route, body) => {
    const response = await fetch(`${origin}${route}`, {
      method,
      headers: { "Content-Type": "application/json", "X-Session-API-Key": key },
      body: JSON.stringify(body),
    });
    if (!response.ok) throw new Error(`${route}: ${response.status}`);
    return response.json();
  };
  const { id } = await call("POST", "/api/conversations", {
    workspace: { working_dir: fs.mkdtempSync(path.join(os.tmpdir(), "seed-")) },
    agent_settings: {
      llm: { usage_id: "seed", model: "openai/never-called", api_key: "none" },
    },
    autotitle: false,
  });
  for (let i = 0; i < 40; i += 1) {
    await call("POST", `/api/conversations/${id}/events`, {
      role: "user",
      content: [{ type: "text", text: `message ${i}` }],
      run: false,
    });
  }
  console.log(id);
} else if (command === "shot") {
  const [conversationId, output] = rest;
  const { chromium } = await import("playwright");
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1360, height: 900 },
  });
  await context.addInitScript((apiKey) => {
    const backend = {
      id: "local",
      name: "local",
      host: window.location.origin,
      apiKey,
      kind: "local",
    };
    const active = JSON.stringify({ backendId: "local", orgId: null });
    localStorage.setItem("openhands-onboarded", "1");
    localStorage.setItem(
      "openhands-sidebar-onboarding-checklist-dismissed",
      "true",
    );
    localStorage.setItem("openhands-backends", JSON.stringify([backend]));
    localStorage.setItem("openhands-active-backend", active);
    sessionStorage.setItem("openhands-active-backend", active);
  }, key);
  const page = await context.newPage();
  await page.goto(`${origin}/conversations/${conversationId}`);
  const consent = page.getByRole("button", { name: /confirm preferences/i });
  await consent.click({ timeout: 9000 }).catch(() => {});
  await page.waitForTimeout(9000);
  await page.screenshot({ path: output });
  await browser.close();
} else {
  console.error("usage: evidence.mjs seed|shot <origin> <session key> ...");
  process.exit(1);
}
