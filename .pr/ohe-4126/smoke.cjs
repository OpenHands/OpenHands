/* OHE-4126: the Connect Git tour on the MCP page, driven on the Canvas mock
 * API (`npm run dev:mock`) with a cloud backend on the same origin. The two
 * cloud calls the setup guide makes are answered from the page, and the setup
 * state can be delayed: that delay is the window a quick click falls into.
 *
 *   node .pr/ohe-4126/smoke.cjs <label> [output dir]
 *
 * BASE defaults to http://localhost:3001. Uses the installed Chrome. */
const path = require("node:path");
const { chromium } = require("@playwright/test");

const BASE = process.env.BASE || "http://localhost:3001";
const LABEL = process.argv[2] || "run";
const OUT = path.resolve(process.argv[3] || __dirname);
const VIEWPORT = { width: 1600, height: 1000 };

function seed(setupStateDelayMs) {
  localStorage.setItem("analytics-consent", "false");
  localStorage.setItem("openhands-telemetry-consent", "denied");
  localStorage.setItem("openhands-telemetry-first-use", "true");
  localStorage.setItem("openhands-onboarded", "1");
  localStorage.setItem(
    "openhands-backends",
    JSON.stringify([
      {
        id: "cloud-1",
        name: "OHE",
        host: window.location.origin,
        apiKey: "",
        kind: "cloud",
      },
    ]),
  );
  localStorage.setItem(
    "openhands-active-backend",
    JSON.stringify({ backendId: "cloud-1", orgId: "2" }),
  );
  const state = {
    wizard_pending: false,
    guide_org_id: "2",
    guide_dismissed: false,
    guide_steps: {
      org_llm: true,
      git_provider: false,
      mcp_server: false,
      automation: false,
      invite: false,
    },
  };
  const me = {
    org_id: "2",
    user_id: "user-1",
    role: "owner",
    permissions: ["manage_super_admins", "view_automations"],
  };
  const realFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    const url = typeof input === "string" ? input : input.url;
    const method = (
      (init && init.method) ||
      (input && input.method) ||
      "GET"
    ).toUpperCase();
    const json = (body) =>
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    if (method === "GET" && url.includes("/api/admin/setup-state")) {
      await new Promise((resolve) => {
        setTimeout(resolve, setupStateDelayMs);
      });
      // Set by the "step done" scenario once the Git step has succeeded.
      if (sessionStorage.getItem("git-done")) {
        state.guide_steps.git_provider = true;
      }
      return json(state);
    }
    if (method === "GET" && /\/api\/organizations\/[^/]+\/me(\?|$)/.test(url)) {
      return json(me);
    }
    return realFetch(input, init);
  };
}

async function capture(page, name) {
  // The mock API does not serve every cloud endpoint, which raises a toast.
  await page
    .getByText("An error occurred")
    .waitFor({ state: "hidden", timeout: 8000 })
    .catch(() => undefined);
  const seen = await page.evaluate(() => {
    const rect = (el) => {
      if (!el) return null;
      const box = el.getBoundingClientRect();
      return { x: Math.round(box.x), y: Math.round(box.y) };
    };
    const popover = document.querySelector(".driver-popover");
    const overlay = document.querySelector(".driver-overlay");
    const dialog = document.querySelector('[data-testid="mcp-install-modal"]');
    return {
      tourStop:
        popover && getComputedStyle(popover).display !== "none"
          ? {
              title: popover.querySelector(".driver-popover-title").textContent,
              progress: popover.querySelector(".driver-popover-progress-text")
                .textContent,
              at: rect(popover),
            }
          : null,
      overlay: overlay ? getComputedStyle(overlay).display !== "none" : false,
      dialog: rect(dialog),
      dialogTakesClicks: dialog
        ? getComputedStyle(dialog).pointerEvents !== "none"
        : null,
      setupGuide: Boolean(
        document.querySelector('[data-testid="super-admin-setup-guide"]'),
      ),
      path: window.location.pathname,
    };
  });
  console.log(`[${LABEL}] ${name}: ${JSON.stringify(seen)}`);
  await page.screenshot({ path: path.join(OUT, `${LABEL}-${name}.png`) });
}

async function scenario(browser, setupStateDelayMs, run) {
  const context = await browser.newContext({ viewport: VIEWPORT });
  await context.addInitScript(seed, setupStateDelayMs);
  const page = await context.newPage();
  try {
    await run(page);
  } finally {
    await context.close();
  }
}

const openTour = (page) =>
  page.goto(`${BASE}/mcp?setup_tour=connect-git`, {
    waitUntil: "domcontentloaded",
  });

async function openTokenStop(page) {
  await openTour(page);
  await page.locator(".driver-popover").waitFor({ timeout: 20000 });
  await page.waitForTimeout(600);
  await page.getByTestId("mcp-marketplace-card-github").click();
  await page.getByTestId("mcp-install-modal").waitFor({ timeout: 10000 });
  await page.waitForTimeout(1200);
}

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });

  // Vite optimizes dependencies on the first request and reloads the page.
  await scenario(browser, 0, async (page) => {
    await page.goto(`${BASE}/mcp`, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(6000);
    await page.goto(`${BASE}/mcp`, { waitUntil: "domcontentloaded" });
    await page
      .getByTestId("mcp-marketplace-card-github")
      .waitFor({ timeout: 30000 });
  });

  // The admin clicks GitHub before the guide's state has arrived.
  await scenario(browser, 2500, async (page) => {
    await openTour(page);
    await page.getByTestId("mcp-marketplace-card-github").click();
    await page.getByTestId("mcp-install-modal").waitFor({ timeout: 10000 });
    await page.waitForTimeout(4500);
    await capture(page, "quick-click");
  });

  // The usual order, then the dialog is closed on the token stop.
  await scenario(browser, 0, async (page) => {
    await openTokenStop(page);
    await capture(page, "token-stop");
    await page.getByTestId("mcp-install-modal-close").click();
    await page.waitForTimeout(600);
    await capture(page, "dialog-closed");
  });

  // A second click on the card's "+" lands on the dialog's backdrop.
  await scenario(browser, 0, async (page) => {
    await openTour(page);
    await page.locator(".driver-popover").waitFor({ timeout: 20000 });
    await page.waitForTimeout(600);
    const box = await page
      .getByTestId("mcp-marketplace-toggle-github")
      .boundingBox();
    const x = box.x + box.width / 2;
    const y = box.y + box.height / 2;
    await page.mouse.click(x, y);
    await page.waitForTimeout(380);
    await page.mouse.click(x, y);
    await page.waitForTimeout(1200);
    await capture(page, "second-click");
  });

  // The Git step succeeds on the token stop, then its dialog closes.
  await scenario(browser, 0, async (page) => {
    await openTokenStop(page);
    await page.evaluate(() => {
      sessionStorage.setItem("git-done", "1");
      window.dispatchEvent(
        new CustomEvent("oh-super-admin-setup-step", {
          detail: { id: "connect-git" },
        }),
      );
    });
    await page.getByTestId("mcp-install-modal-close").click();
    await page.waitForTimeout(3000);
    await capture(page, "step-done");
  });

  await browser.close();
})().catch((error) => {
  console.error("SMOKE FAILED:", error);
  process.exit(1);
});
