import { chromium } from "playwright";
const browser = await chromium.launch();
const page = await (await browser.newContext()).newPage();
const failures = [];
page.on("requestfailed", (r) => failures.push(`${r.method()} ${r.url()} :: ${r.failure()?.errorText}`));
page.on("response", (r) => { if (r.status() >= 400) failures.push(`${r.status()} ${r.url()}`); });
await page.goto("http://localhost:3001/", { waitUntil: "domcontentloaded" });
await page.evaluate(() => {
  localStorage.setItem("openhands-onboarded", "1");
  localStorage.setItem("openhands-telemetry-consent", "denied");
});
await page.goto("http://localhost:3001/settings/agents", { waitUntil: "networkidle" });
await page.waitForTimeout(3000);
const txt = await page.locator("body").innerText();
console.log("PAGE SAYS:", txt.includes("Failed to load") ? "Failed to load profiles" : txt.includes("Disconnected") ? "Disconnected" : "(loaded ok)");
console.log("backend registry:", await page.evaluate(() => {
  const out = {};
  for (let i = 0; i < localStorage.length; i++) {
    const k = localStorage.key(i);
    if (/backend|host|api/i.test(k)) out[k] = localStorage.getItem(k)?.slice(0, 200);
  }
  return out;
}));
console.log("FAILURES:", JSON.stringify(failures.slice(0, 6), null, 1));
await browser.close();
