import { chromium } from "playwright";

const CANVAS = process.env.E2E_CANVAS_URL ?? "http://127.0.0.1:3021";
const MOCK = process.env.E2E_MOCK_URL ?? "http://127.0.0.1:18399";
const post = (path, body = {}) =>
  fetch(MOCK + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

await post("/admin/reset");
await post("/admin/trajectory/register", {
  name: "fin",
  turns: [0, 1, 2].map(() => ({
    tool_call: { name: "finish", arguments: { message: "OK" } },
  })),
});
await post("/admin/trajectory/activate", { name: "fin" });

const browser = await chromium.launch({
  channel: process.env.E2E_BROWSER_CHANNEL,
});
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
let launch = null;
page.on("request", (request) => {
  if (
    request.method() === "POST" &&
    /\/api\/conversations(\?|$)/.test(request.url())
  ) {
    launch = request.postDataJSON();
  }
});
await page.goto(CANVAS, { waitUntil: "networkidle" });
const skip = page.getByText("Skip for now");
if (await skip.isVisible()) await skip.click();
const box = page.locator("textarea, [contenteditable=true]").first();
await box.waitFor({ timeout: 30000 });
await box.click();
await box.fill("finish");
await page.keyboard.press("Enter");
for (let i = 0; i < 60 && !launch; i += 1) await page.waitForTimeout(500);
await page.waitForTimeout(15000);

const requests = await (await fetch(`${MOCK}/admin/requests`)).json();
const calls = Array.isArray(requests) ? requests : (requests.requests ?? []);
const last = calls.at(-1);
console.log(
  JSON.stringify({
    launch_path: launch?.agent_profile_id
      ? "profile"
      : launch?.agent_settings
        ? "agent_settings"
        : launch
          ? "agent"
          : null,
    sends_tools: launch?.agent_settings
      ? "tools" in launch.agent_settings
      : null,
    schema_version: launch?.agent_settings?.schema_version,
    llm_calls: calls.length,
    model_tools: (last?.tools ?? []).map((tool) => tool.function.name),
  }),
);
if (process.env.E2E_SCREENSHOT) {
  await page.screenshot({ path: process.env.E2E_SCREENSHOT });
}
await browser.close();
