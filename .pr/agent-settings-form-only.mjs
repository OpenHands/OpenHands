import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { chromium } from "playwright";
import {
  AgentProfilesClient,
  ProfilesClient,
} from "@openhands/typescript-client/clients";

const canvas = process.env.E2E_CANVAS_URL ?? "http://127.0.0.1:3031";
const mock = process.env.E2E_MOCK_URL ?? "http://127.0.0.1:18699";
const out = process.env.E2E_OUT ?? ".pr";
const options = {
  host: process.env.E2E_SERVER_URL ?? "http://127.0.0.1:18600",
  apiKey: readFileSync(process.env.E2E_KEY_FILE, "utf8").trim(),
};
for (const url of [canvas, mock, options.host]) {
  assert(["localhost", "127.0.0.1"].includes(new URL(url).hostname));
}

await new ProfilesClient(options).saveProfile("mock", {
  llm: {
    model: "openai/mock-test-model",
    base_url: `${mock}/v1`,
    api_key: "mock-test-key",
  },
  include_secrets: true,
});
const agentProfiles = new AgentProfilesClient(options);
for (const name of ["acp-evidence", "oh-evidence"]) {
  await agentProfiles.deleteAgentProfile(name).catch(() => {});
}

const browser = await chromium.launch({
  channel: process.env.E2E_BROWSER_CHANNEL,
});
const page = await browser.newPage({ viewport: { width: 1400, height: 2000 } });

await page.goto(`${canvas}/settings/agent`, { waitUntil: "networkidle" });
const consent = page.getByTestId("confirm-telemetry-preferences");
if (await consent.isVisible().catch(() => false)) await consent.click();
const skip = page.getByTestId("onboarding-skip");
if (await skip.isVisible().catch(() => false)) await skip.click();
if (!/\/settings\/agents$/.test(page.url())) {
  await page.goto(`${canvas}/settings/agent`, { waitUntil: "networkidle" });
}
await page.waitForURL(/\/settings\/agents$/);
console.log(`/settings/agent -> ${new URL(page.url()).pathname}`);

async function pick(testId, option) {
  await page.getByTestId(testId).click();
  await page.getByRole("option", { name: option, exact: true }).click();
}
async function assertFormOnly() {
  const form = page.getByTestId("agent-settings-screen");
  assert.equal(await form.locator("h1, h2, h3").count(), 0);
  assert.equal(await page.getByTestId("agent-save-button").count(), 0);
  assert.equal(await page.getByTestId("save-agent-profile-btn").count(), 1);
}
async function reopen(name) {
  await page.reload({ waitUntil: "networkidle" });
  const row = page.getByTestId("agent-profile-row").filter({ hasText: name });
  await row.getByTestId("agent-profile-menu-trigger").click();
  await page.getByTestId("agent-profile-edit").click();
  await page.getByTestId("agent-settings-screen").waitFor();
  await assertFormOnly();
}

await page.getByTestId("add-agent-profile").click();
await page.getByTestId("agent-profile-name-input").fill("acp-evidence");
await pick("agent-type-selector", "ACP (external subprocess)");
await page.getByTestId("agent-command-input").waitFor();
await assertFormOnly();
await page.getByTestId("save-agent-profile-btn").click();
await page.getByTestId("add-agent-profile").waitFor();
const acp = (await agentProfiles.getAgentProfile("acp-evidence")).profile;
const { acp_server: server, acp_command: command, acp_model: model } = acp;
console.log(`acp-evidence: ${JSON.stringify({ server, command, model })}`);
assert.equal(server, "claude-code");
assert.equal(command ?? null, null);
await reopen("acp-evidence");
await page.screenshot({ path: `${out}/form-only-acp-profile.png` });

await page.getByTestId("back-to-agent-profiles").click();
await page.getByTestId("add-agent-profile").click();
await page.getByTestId("agent-profile-name-input").fill("oh-evidence");
await page.getByTestId("sdk-settings-tool_concurrency_limit").fill("4");
await assertFormOnly();
await page.getByTestId("save-agent-profile-btn").click();
await page.getByTestId("add-agent-profile").waitFor();
const oh = (await agentProfiles.getAgentProfile("oh-evidence")).profile;
const { llm_profile_ref: llm, tool_concurrency_limit: limit, tools } = oh;
console.log(`oh-evidence: ${JSON.stringify({ llm, limit, tools })}`);
assert.equal(llm, "mock");
assert.equal(limit, 4);
await reopen("oh-evidence");
assert.equal(
  await page.getByTestId("sdk-settings-tool_concurrency_limit").inputValue(),
  "4",
);
await page.screenshot({ path: `${out}/form-only-openhands-profile.png` });

await browser.close();
