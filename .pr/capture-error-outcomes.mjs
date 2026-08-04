/**
 * PR #16231 screenshot capture.
 *
 * Spins up the real mock-mode frontend (evidence for PR #16231) (built with VITE_MOCK_API=true, served
 * on :4317) in a completely isolated Playwright browser context (fresh
 * localStorage, dedicated context — no shared browser profile). Classified
 * error events are injected over the real app WebSocket so the
 * ErrorMessageBanner renders both outcome states:
 *
 *   - auth/recoverable classification  -> warning banner (Incorrect API key)
 *   - internal classification          -> error banner   (Internal failure)
 */
import { chromium } from "playwright-core";
import { mkdirSync } from "node:fs";

const CHROMIUM_PATH =
  process.env.PLAYWRIGHT_CHROMIUM ||
  "/home/gneubig/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome";
const APP_URL = "http://localhost:4317";
const OUT_DIR = "/tmp/OpenHands/.pr";

mkdirSync(OUT_DIR, { recursive: true });

const NOW = Date.now();
const makeErrorEvent = (id, detail, classification) => ({
  id,
  timestamp: new Date(NOW).toISOString(),
  source: "environment",
  kind: "ConversationErrorEvent",
  detail,
  code: classification?.kind === "auth" ? "ACPAuthRequired" : "InternalError",
  ...(classification ? { classification } : {}),
});

const WARNING_EVENT = makeErrorEvent("evt-auth-screenshot", "Incorrect API key", {
  kind: "auth",
  retryable: false,
  user_action: "settings",
});

const ERROR_EVENT = makeErrorEvent("evt-internal-screenshot", "Internal failure", {
  kind: "internal",
  retryable: false,
  user_action: "none",
});

const browser = await chromium.launch({
  executablePath: CHROMIUM_PATH,
  headless: true,
});

const context = await browser.newContext({
  viewport: { width: 1440, height: 900 },
  colorScheme: "dark",
});

// Seed backend + skip onboarding in an isolated, throwaway context.
await context.addInitScript(() => {
  const backends = [
    {
      id: "default-local",
      name: "Local",
      host: "http://localhost:4317",
      apiKey: "mock-session-key",
      kind: "local",
    },
  ];
  try {
    window.localStorage.setItem("openhands-backends", JSON.stringify(backends));
    window.localStorage.setItem(
      "openhands-active-backend",
      JSON.stringify({ backendId: "default-local", orgId: null }),
    );
    window.localStorage.setItem("openhands-onboarded", "1");
  } catch {
    /* ignore */
  }
});

let wsReadyResolve;
const wsReady = new Promise((resolve) => {
  wsReadyResolve = resolve;
});

let capturedSend = null;

await context.routeWebSocket(/\/sockets\/events\//, (ws) => {
  ws.onMessage((message) => {
    try {
      const parsed = JSON.parse(String(message));
      if (parsed.type === "auth") {
        ws.send(JSON.stringify({ type: "auth_success" }));
        return;
      }
    } catch {
      /* not JSON */
    }
  });
  capturedSend = ws.send.bind(ws);
  wsReadyResolve();
});

const page = await context.newPage();

await page.goto(APP_URL, { waitUntil: "domcontentloaded" });
await page.waitForTimeout(4000);

// Dismiss any overlaying dialog (posthog consent) that blocks clicks.
const dialog = page.locator("[role='dialog']").first();
if (await dialog.count()) {
  const closeBtn = dialog.locator("button").first();
  await closeBtn.click({ timeout: 3000 }).catch(() => {});
  await page.waitForTimeout(500);
}

// Open the first mock conversation.
const firstConv = page.locator("a[href^='/conversations/']").first();
if (await firstConv.count()) {
  await firstConv.click({ timeout: 15000 });
  await page.waitForTimeout(5000);
}

await wsReady;
await new Promise((r) => setTimeout(r, 500));

const banner = page.locator("[data-testid='error-message-banner']");
const warnIcon = page.locator("[data-testid='warning-message-banner-icon']");
const errIcon = page.locator("[data-testid='error-message-banner-icon']");

// ---- State 1: recoverable/auth classification -> warning banner ----
capturedSend(JSON.stringify(WARNING_EVENT));
await page.waitForTimeout(2000);
console.log(
  "warning state | banner:",
  (await banner.textContent())?.trim().slice(0, 120),
  "| warnIcon:",
  await warnIcon.count(),
  "| errIcon:",
  await errIcon.count(),
);
const warnBox = await banner.boundingBox();
console.log("warning banner box:", JSON.stringify(warnBox));
await page.screenshot({ path: `${OUT_DIR}/error-outcome-warning.png` });
await banner.screenshot({ path: `${OUT_DIR}/error-outcome-warning-banner.png` });

// ---- State 2: internal classification -> error banner ----
capturedSend(JSON.stringify(ERROR_EVENT));
await page.waitForTimeout(2000);
console.log(
  "error state   | banner:",
  (await banner.textContent())?.trim().slice(0, 120),
  "| warnIcon:",
  await warnIcon.count(),
  "| errIcon:",
  await errIcon.count(),
);
const errBox = await banner.boundingBox();
console.log("error banner box:", JSON.stringify(errBox));
await page.screenshot({ path: `${OUT_DIR}/error-outcome-internal.png` });
await banner.screenshot({ path: `${OUT_DIR}/error-outcome-internal-banner.png` });

console.log("Screenshots written to", OUT_DIR);
await browser.close();
