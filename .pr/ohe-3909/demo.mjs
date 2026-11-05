// Drives the mock-API Canvas and opens the "Archived Project" conversation
// (mock id 4, sandbox MISSING) after rewriting the mock handler module so that
// the conversation's history ends the way an automation run's does: a blocked
// Stop hook, the SDK's "[Stop hook feedback]" message, the agent's retry and a
// final successful Stop hook whose `error` is null.
//
// Usage (from the repo root, with `npm run dev:mock -- --port <port>` running):
//   node .pr/ohe-3909/demo.mjs http://localhost:<port> <outDir> <before|after>
import { chromium } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const [baseUrl, outDir, label] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });

const HAIKU = "Ink on careful maps,\nthe river laughs and wanders.\nWe redraw the shore.";
const FEEDBACK =
  "[Stop hook feedback] The task appears complete, but automation runs must end by calling the finish tool. Please call the finish tool now with the final task outcome.";
const CONTEXT =
  "The task appears complete, but automation runs must end by calling the finish tool. Please call the finish tool now with the final task outcome.";
const at = (s) => new Date(Date.UTC(2026, 10, 3, 20, 7, 12 + s)).toISOString();
const msg = (id, source, role, text, s) => ({
  id,
  timestamp: at(s),
  source,
  llm_message: { role, content: [{ type: "text", text }] },
  activated_skills: [],
  extended_content: [],
});
const hook = (id, s, extra = {}) => ({
  id,
  timestamp: at(s),
  source: "hook",
  kind: "HookExecutionEvent",
  hook_event_type: "Stop",
  hook_command: "python -c 'import json, os, sys; ...finish_tool_used...'",
  success: true,
  blocked: false,
  exit_code: 0,
  reason: null,
  tool_name: null,
  action_id: null,
  message_id: null,
  stdout: "",
  stderr: "",
  error: null,
  additional_context: null,
  hook_input: { reason: "agent_finished" },
  ...extra,
});
const EVENTS = [
  hook("h-session-start", 0, { hook_event_type: "SessionStart", hook_input: null }),
  msg("m-user", "user", "user", "Write a short haiku about plans", 1),
  msg("m-agent-1", "agent", "assistant", HAIKU, 10),
  hook("h-stop-blocked", 11, {
    blocked: true,
    success: false,
    exit_code: 2,
    reason: "finish tool was not used",
    additional_context: CONTEXT,
    stdout: JSON.stringify({ decision: "deny", reason: "finish tool was not used", additionalContext: CONTEXT }),
  }),
  msg("m-feedback", "environment", "user", FEEDBACK, 12),
  msg("m-agent-2", "agent", "assistant", HAIKU, 20),
  hook("h-post-finish", 21, { hook_event_type: "PostToolUse", tool_name: "finish", hook_input: null }),
  hook("h-stop-ok", 22),
];

const browser = await chromium.launch({ channel: "chrome", headless: true });
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
await context.addInitScript(() => {
  window.localStorage.setItem("analytics-consent", "false");
  window.localStorage.setItem("openhands-telemetry-consent", "denied");
  window.localStorage.setItem("openhands-telemetry-first-use", "true");
  window.localStorage.setItem("openhands-onboarded", "1");
  window.localStorage.setItem(
    "openhands-backends",
    JSON.stringify([{ id: "default-local", name: "Local", host: window.location.origin, apiKey: "", kind: "local" }]),
  );
  window.localStorage.setItem("openhands-active-backend", JSON.stringify({ backendId: "default-local", orgId: null }));
});

let rewrote = false;
await context.route("**/src/mocks/conversation-handlers.ts*", async (route) => {
  const response = await route.fetch();
  let body = await response.text();
  const marker = /const CONVERSATION_EVENTS\s*=\s*\{/;
  if (!marker.test(body)) throw new Error("CONVERSATION_EVENTS marker not found in served module");
  body = body.replace(marker, `const CONVERSATION_EVENTS = { "4": ${JSON.stringify(EVENTS)},`);
  rewrote = true;
  await route.fulfill({ response, body, headers: { ...response.headers(), "content-length": String(Buffer.byteLength(body)) } });
});

// The mock settings leave the analytics choice unanswered, which pops the
// telemetry consent modal over the transcript; answer it in the mock itself.
await context.route("**/src/mocks/settings-handlers.ts*", async (route) => {
  const response = await route.fetch();
  const body = (await response.text()).replace("user_consents_to_analytics: null,", "user_consents_to_analytics: true,");
  await route.fulfill({ response, body, headers: { ...response.headers(), "content-length": String(Buffer.byteLength(body)) } });
});

const page = await context.newPage();
const consoleErrors = [];
page.on("console", (m) => { if (m.type() === "error") consoleErrors.push(m.text().slice(0, 200)); });

const deadline = Date.now() + 120_000;
for (;;) {
  try { await page.goto(`${baseUrl}/conversations/4`, { waitUntil: "domcontentloaded", timeout: 15_000 }); break; }
  catch (e) { if (Date.now() > deadline) throw e; await new Promise((r) => setTimeout(r, 2000)); }
}
const consent = page.getByTestId("confirm-telemetry-preferences");
if (await consent.isVisible({ timeout: 5000 }).catch(() => false)) await consent.click();

await page.getByText("Write a short haiku about plans").first().waitFor({ timeout: 60_000 });
// Give the history effect a beat to run its "newest event" check.
await page.waitForTimeout(800);
const toast = page.getByText("An error occurred. Please try again.");
const toastSeen = (await toast.count()) > 0;
await page.screenshot({ path: join(outDir, `${label}.png`), fullPage: false });
const result = {
  label,
  baseUrl,
  rewroteMockModule: rewrote,
  toastSeen,
  hookRows: await page.getByText(/^Hook: Stop/).count(),
  noteCount: await page.getByTestId("corrective-nudge-message").count(),
  environmentMessageCount: await page.getByTestId("environment-message").count(),
  archivedBanner: await page.getByTestId("archived-conversation-banner").count(),
  consoleErrors: consoleErrors.slice(0, 8),
};
writeFileSync(join(outDir, `${label}.json`), JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
await browser.close();
