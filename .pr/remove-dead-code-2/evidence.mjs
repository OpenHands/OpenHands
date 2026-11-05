// Screenshot of Agent Canvas (mock API) running from this branch.
// Usage: VITE_MOCK_API=true npx react-router dev --port 5199 --strictPort
//        node .pr/remove-dead-code-2/evidence.mjs <out.png>
import { chromium } from "playwright";

const out = process.argv[2] ?? "after.png";
const base = "http://localhost:5199";
const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: 2,
});
await context.addInitScript(() => {
  const backend = {
    id: "local-mock",
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
  localStorage.setItem("openhands-sidebar-onboarding-checklist-dismissed", "true");
});
const page = await context.newPage();
await page.goto(`${base}/conversations/1`);
await page.waitForSelector('[data-testid="chat-interface"]', { timeout: 60000 });
const confirm = page.getByRole("button", { name: "Confirm preferences" });
try {
  await confirm.waitFor({ state: "visible", timeout: 15000 });
  await confirm.click();
} catch {
  // analytics dialog did not appear
}
await page.waitForTimeout(2000);
await page.screenshot({ path: out });
await browser.close();
console.log("wrote", out);
