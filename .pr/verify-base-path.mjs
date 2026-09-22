import { chromium } from "@playwright/test";

const targets = [
  { label: "PRE-FIX  (port 13002)", base: "http://localhost:13002/canvas/" },
  { label: "POST-FIX (port 13001)", base: "http://localhost:13001/canvas/" },
];

const browser = await chromium.launch();

// The launcher normally seeds a backend from runtime config. Under the mock
// worker no real agent server exists, but the registry still needs an entry or
// the app renders the onboarding gate instead of the conversation UI. The host
// must match the page origin so MSW (same-origin) can intercept the API calls.
const BACKENDS_STORAGE_KEY = "openhands-backends";
const ACTIVE_BACKEND_STORAGE_KEY = "openhands-active-backend";

for (const { label, base } of targets) {
  const context = await browser.newContext();
  await context.addInitScript(
    ([backendsKey, activeKey]) => {
      const backend = {
        id: "default-local",
        name: "Local",
        host: window.location.origin,
        apiKey: "mock-session-key",
        kind: "local",
      };
      window.localStorage.setItem(backendsKey, JSON.stringify([backend]));
      window.localStorage.setItem(
        activeKey,
        JSON.stringify({ backendId: backend.id }),
      );
      // Equivalent to completing/skipping the welcome flow, so the app renders
      // the conversation UI instead of the onboarding wizard.
      window.localStorage.setItem("openhands-onboarded", "true");
    },
    [BACKENDS_STORAGE_KEY, ACTIVE_BACKEND_STORAGE_KEY],
  );
  const page = await context.newPage();
  const errors = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });

  await page.goto(base, { waitUntil: "networkidle" });
  await page.waitForTimeout(2500);

  // Complete/skip the welcome wizard so the conversation UI renders. Keep the
  // visibility probes short — a missing button must not stall on the default
  // 30s action timeout.
  const skipNames = [/skip/i, /confirm/i, /close/i, /next/i, /get started/i];
  for (let i = 0; i < 8; i++) {
    let clicked = false;
    for (const name of skipNames) {
      const btn = page.getByRole("button", { name }).first();
      if (await btn.isVisible({ timeout: 400 }).catch(() => false)) {
        await btn.click({ timeout: 1500 }).catch(() => {});
        clicked = true;
        await page.waitForTimeout(900);
        break;
      }
    }
    if (!clicked) break;
  }
  await page.waitForTimeout(2000);

  // Collect every in-app anchor the SPA renders, ignoring external links.
  const anchors = await page.$$eval("a[href]", (els) =>
    els
      .map((el) => ({
        href: el.getAttribute("href"),
        text: (el.textContent || "").trim().slice(0, 28),
      }))
      .filter((a) => a.href && !/^https?:\/\//i.test(a.href)),
  );

  const inApp = anchors.filter((a) => !/^[a-z]+:/i.test(a.href));
  const wrongBase = inApp.filter((a) => !a.href.startsWith("/canvas"));

  console.log(`\n=== ${label} ===`);
  console.log(`in-app hrefs sampled: ${inApp.length}`);
  console.log(`hrefs MISSING the /canvas prefix: ${wrongBase.length}`);
  for (const a of wrongBase.slice(0, 8)) {
    console.log(`   BUG  href="${a.href}"  (${a.text})`);
  }
  for (const a of inApp
    .filter((x) => x.href.startsWith("/canvas"))
    .slice(0, 4)) {
    console.log(`   OK   href="${a.href}"  (${a.text})`);
  }
  // Resolve where a bare link actually lands: the origin root, i.e. the
  // enterprise app on the shared host.
  const firstBug = wrongBase[0];
  if (firstBug) {
    const resolved = await page.evaluate(
      (h) => new URL(h, window.location.origin).toString(),
      firstBug.href,
    );
    console.log(`   -> bug link resolves to ${resolved} (leaves /canvas)`);
  }
  if (errors.length) {
    console.log(`   console errors: ${errors.slice(0, 2).join(" | ")}`);
  }
  await context.close();
}

await browser.close();