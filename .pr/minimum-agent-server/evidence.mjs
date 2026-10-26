// Screenshot Canvas (VITE_MOCK_API=true) connecting to its mock agent-server.
// Usage: node .pr/minimum-agent-server/evidence.mjs <base-url> <out.png>
// For the before/after pair the mock server_info reported version 1.56.0
// (MOCK_AGENT_SERVER_VERSION, local edit), with minimumAgentServer 1.55.0
// (main) and then 1.57.0 (this PR).
import { createRequire } from "node:module";

const require = createRequire(new URL("../../package.json", import.meta.url));
const { chromium } = require("playwright");

const [baseUrl, out] = process.argv.slice(2);
const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: { width: 1280, height: 800 },
  deviceScaleFactor: 2,
});
await page.addInitScript(() => {
  const backend = {
    id: "local",
    name: "Local",
    host: location.origin,
    apiKey: "",
    kind: "local",
  };
  const active = JSON.stringify({ backendId: backend.id, orgId: null });
  localStorage.setItem("openhands-onboarded", "1");
  localStorage.setItem("openhands-backends", JSON.stringify([backend]));
  localStorage.setItem("openhands-active-backend", active);
  sessionStorage.setItem("openhands-active-backend", active);
});
await page.goto(baseUrl);
await page.waitForTimeout(8000);
const consent = page.getByRole("button", { name: "Confirm preferences" });
if (await consent.isVisible()) await consent.click();
await page.waitForTimeout(1500);
const message = page.getByText(/requires agent-server/).first();
console.log(
  JSON.stringify({
    out,
    versionMessage: (await message.isVisible())
      ? await message.textContent()
      : null,
    homeScreen: await page.getByTestId("home-screen").isVisible(),
  }),
);
await page.screenshot({ path: out });
await browser.close();
