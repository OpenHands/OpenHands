/**
 * Captures side-by-side screenshots of the cmd/ctrl+click outcome for the PR.
 * Writes into .pr/artifacts/.
 *
 *   node .pr/capture-screenshots.mjs
 */
import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";

const targets = [
  { label: "pre-fix", port: 13002 },
  { label: "post-fix", port: 13001 },
];

await mkdir(".pr/artifacts", { recursive: true });
const browser = await chromium.launch();

for (const { label, port } of targets) {
  const base = `http://localhost:${port}/canvas/`;
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
  });
  await context.addInitScript((origin) => {
    const backend = {
      id: "default-local",
      name: "Local",
      host: origin,
      apiKey: "mock-session-key",
      kind: "local",
    };
    window.localStorage.setItem("openhands-backends", JSON.stringify([backend]));
    window.localStorage.setItem(
      "openhands-active-backend",
      JSON.stringify({ backendId: backend.id }),
    );
    window.localStorage.setItem("openhands-onboarded", "true");
  }, `http://localhost:${port}`);

  const page = await context.newPage();
  await page.goto(base, { waitUntil: "networkidle" });
  await page.waitForTimeout(2500);
  for (const name of [/skip/i, /confirm/i, /close/i]) {
    const btn = page.getByRole("button", { name }).first();
    if (await btn.isVisible({ timeout: 400 }).catch(() => false)) {
      await btn.click({ timeout: 1500 }).catch(() => {});
      await page.waitForTimeout(900);
    }
  }
  await page.waitForTimeout(1500);
  await page.screenshot({ path: `.pr/artifacts/${label}-1-canvas.png` });

  const newChat = page.getByRole("link", { name: /new chat/i }).first();
  const [newTab] = await Promise.all([
    context.waitForEvent("page", { timeout: 8000 }).catch(() => null),
    newChat.click({ modifiers: ["ControlOrMeta"], timeout: 4000 }).catch(() => {}),
  ]);
  await page.waitForTimeout(1500);

  if (newTab) {
    await newTab.waitForLoadState("domcontentloaded").catch(() => {});
    await newTab.waitForTimeout(1200);
    await newTab.screenshot({
      path: `.pr/artifacts/${label}-2-new-tab.png`,
    });
    await newTab.close().catch(() => {});
  }
  await context.close();
  console.log(`captured ${label}`);
}

await browser.close();