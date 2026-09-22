/**
 * Reproduces the literal OSS-13667 action: cmd/ctrl+click (and middle-click) on
 * the "New Chat" link, which opens a new tab. The new tab's URL decides whether
 * the user lands on Canvas or on the enterprise app that owns the origin root.
 *
 *   node .pr/verify-cmd-click.mjs
 */
import { chromium } from "@playwright/test";

const targets = [
  { label: "PRE-FIX  (port 13002)", base: "http://localhost:13002/canvas/" },
  { label: "POST-FIX (port 13001)", base: "http://localhost:13001/canvas/" },
];

const browser = await chromium.launch();

async function prepare(context, base) {
  await context.addInitScript((targetOrigin) => {
    const backend = {
      id: "default-local",
      name: "Local",
      host: targetOrigin,
      apiKey: "mock-session-key",
      kind: "local",
    };
    window.localStorage.setItem("openhands-backends", JSON.stringify([backend]));
    window.localStorage.setItem(
      "openhands-active-backend",
      JSON.stringify({ backendId: backend.id }),
    );
    window.localStorage.setItem("openhands-onboarded", "true");
  }, new URL(base).origin);

  const page = await context.newPage();
  await page.goto(base, { waitUntil: "networkidle" });
  await page.waitForTimeout(2500);
  const skip = [/skip/i, /confirm/i, /close/i];
  for (let i = 0; i < 6; i++) {
    let clicked = false;
    for (const name of skip) {
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
  await page.waitForTimeout(1500);
  return page;
}

for (const { label, base } of targets) {
  const context = await browser.newContext();
  const page = await prepare(context, base);

  const newChat = page.getByRole("link", { name: /new chat/i }).first();
  const href = await newChat.getAttribute("href");

  // 1. What the link advertises.
  const resolved = new URL(href, base).toString();

  // 2. ctrl/cmd+click — opens a new tab, exactly the reported action. Headless
  // popup handling is timing-sensitive, so retry once before reporting.
  async function cmdClick() {
    const popupPromise = context.waitForEvent("page", { timeout: 15000 });
    await newChat
      .click({ modifiers: ["ControlOrMeta"], timeout: 5000 })
      .catch(() => {});
    return popupPromise.catch(() => null);
  }

  let newTab = await cmdClick();
  if (!newTab) newTab = await cmdClick();
  await page.waitForTimeout(1200);

  const newTabUrl = newTab ? newTab.url() : "(no new tab opened)";
  let landedOn = "unknown";
  if (newTab) {
    await newTab.waitForLoadState("domcontentloaded").catch(() => {});
    const text = await newTab
      .locator("body")
      .innerText()
      .catch(() => "");
    landedOn = text.includes("Enterprise app root") ? "ENTERPRISE APP" : "Canvas";
    await newTab.close().catch(() => {});
  }

  console.log(`\n=== ${label} ===`);
  console.log(`New Chat href           : ${href}`);
  console.log(`resolves to             : ${resolved}`);
  console.log(`cmd/ctrl+click new tab  : ${newTabUrl}`);
  console.log(`landed on               : ${landedOn}`);

  await context.close();
}

await browser.close();