const { chromium } = require("@playwright/test");
const base = process.env.BASE || "http://localhost:3001";
const out = process.env.OUT || "license-banner.png";
const status = JSON.parse(process.env.LICENSE_STATUS);
(async () => {
  const browser = await chromium.launch({ channel: "chrome" });
  const context = await browser.newContext({ viewport: { width: 1280, height: 720 } });
  await context.addInitScript(({ base, status }) => {
    localStorage.setItem("openhands-onboarded", "1");
    localStorage.setItem("openhands-backends", JSON.stringify([
      { id: "locked-cloud", name: "OHE", host: base, apiKey: "cloud-test-api-key", kind: "cloud" },
    ]));
    localStorage.setItem("openhands-active-backend", JSON.stringify({ backendId: "locked-cloud", orgId: "org-1" }));
    localStorage.setItem("analytics-consent", "false");
    localStorage.setItem("openhands-telemetry-consent", "denied");
    localStorage.setItem("openhands-telemetry-first-use", "true");
    const origFetch = window.fetch.bind(window);
    window.__licenseCalls = 0;
    window.fetch = async (input, init) => {
      const url = typeof input === "string" ? input : input.url;
      const json = (body) => new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
      if (new URL(url, location.origin).pathname === "/api/organizations") {
        return json({ items: [{ id: "org-1", name: "Acme Corp", is_personal: false }], current_org_id: "org-1" });
      }
      if (/\/profiles(\?|$)/.test(url)) {
        return json({ profiles: [] });
      }
      if (url.includes("/api/keys/current")) {
        return json({ org_id: "org-1" });
      }
      if (url.includes("/api/license/status")) {
        window.__licenseCalls += 1;
        sessionStorage.setItem("licenseCalls", String(Number(sessionStorage.getItem("licenseCalls") || "0") + 1));
        return new Response(JSON.stringify(status), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      return origFetch(input, init);
    };
  }, { base, status });
  const page = await context.newPage();
  page.on("pageerror", (e) => console.log("pageerror:", (e.stack || String(e)).split("\n").slice(0, 8).join("\n")));
  let consoleErrors = 0;
  page.on("console", (m) => {
    if (process.env.DEBUG && m.type() === "error" && consoleErrors < 6) {
      consoleErrors += 1;
      console.log("console.error:", m.text().slice(0, 700));
    }
  });
  await page.goto(base + "/", { waitUntil: "domcontentloaded" });
  await page.waitForTimeout(8000);
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForTimeout(6000);
  const consent = page.getByTestId("confirm-telemetry-preferences");
  if (await consent.isVisible().catch(() => false)) {
    await consent.click();
    await page.waitForTimeout(1000);
  }
  if (process.env.DEBUG) {
    console.log("dom:", await page.evaluate(() => document.querySelector('[data-testid="license-banner"]')?.outerHTML ?? null));
    console.log("root-layout present:", await page.evaluate(() => !!document.querySelector('[data-testid="root-layout"]')));
    console.log("direct fetch:", await page.evaluate(async () => { const r = await fetch("/api/license/status"); return [r.status, r.headers.get("content-type"), (await r.text()).slice(0, 200)]; }));
    console.log("body text head:", (await page.evaluate(() => document.body.innerText.slice(0, 300))));
  }
  const banner = page.getByTestId("license-banner");
  const visible = await banner.isVisible().catch(() => false);
  console.log("license calls:", await page.evaluate(() => window.__licenseCalls), "total across loads:", await page.evaluate(() => sessionStorage.getItem("licenseCalls")));
  console.log("banner visible:", visible);
  if (visible) {
    console.log("tone:", await banner.getAttribute("data-tone"), "role:", await banner.getAttribute("role"));
    console.log("text:", (await banner.textContent()).trim());
    console.log("link:", await banner.locator("a").getAttribute("href"));
  }
  await page.screenshot({ path: out });
  console.log("screenshot:", out);
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
