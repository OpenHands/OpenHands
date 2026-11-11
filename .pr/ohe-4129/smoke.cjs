/* OHE-4129: a "Hide the Setup guide?" question left open when a tour starts
 * came back when the tour ended. Driven on the Canvas mock API with a cloud
 * backend on the same origin; the guide's cloud calls are answered from the
 * page.
 *
 *   node .pr/ohe-4129/smoke.cjs <label> [output dir]
 *
 * BASE defaults to http://localhost:3001. Uses the installed Chrome. */
const path = require("node:path");
const { chromium } = require("@playwright/test");

const BASE = process.env.BASE || "http://localhost:3001";
const LABEL = process.argv[2] || "run";
const OUT = path.resolve(process.argv[3] || __dirname);
const VIEWPORT = { width: 1600, height: 1000 };

function seed() {
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
  // The Git step is done, so the automation step is next.
  const state = {
    wizard_pending: false,
    guide_org_id: "2",
    guide_dismissed: false,
    guide_steps: {
      org_llm: true,
      git_provider: true,
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
    if (url.includes("/api/admin/setup-state")) {
      if (method === "PATCH") {
        state.guide_dismissed = true;
        sessionStorage.setItem("dismissed", "1");
        return json(state);
      }
      // Set once the automation step has succeeded. The slow answer is the
      // window in which the admin closes the panel.
      if (sessionStorage.getItem("automation-done")) {
        state.guide_steps.automation = true;
        await new Promise((resolve) => {
          setTimeout(resolve, 2500);
        });
      }
      return json(state);
    }
    if (method === "GET" && /\/api\/organizations\/[^/]+\/me(\?|$)/.test(url)) {
      return json(me);
    }
    return realFetch(input, init);
  };
}

async function look(page, name, screenshot = false) {
  const seen = await page.evaluate(() => {
    const popover = document.querySelector(".driver-popover");
    const toggle = document.querySelector(
      '[data-testid="super-admin-setup-guide-toggle"]',
    );
    return {
      path: window.location.pathname,
      question: Boolean(
        document.querySelector('[data-testid="confirmation-modal"]'),
      ),
      tourStop:
        popover && getComputedStyle(popover).display !== "none"
          ? popover.querySelector(".driver-popover-title").textContent
          : null,
      setupGuide: Boolean(
        document.querySelector('[data-testid="super-admin-setup-guide"]'),
      ),
      panelOpen: toggle ? toggle.getAttribute("aria-expanded") : null,
    };
  });
  console.log(`[${LABEL}] ${name}: ${JSON.stringify(seen)}`);
  if (screenshot) {
    // The mock API does not serve every cloud endpoint, which raises a toast.
    await page
      .getByText("An error occurred")
      .waitFor({ state: "hidden", timeout: 8000 })
      .catch(() => undefined);
    await page.screenshot({ path: path.join(OUT, `${LABEL}-${name}.png`) });
  }
  return seen;
}

async function scenario(browser, run) {
  const context = await browser.newContext({ viewport: VIEWPORT });
  await context.addInitScript(seed);
  const page = await context.newPage();
  try {
    await run(page);
  } finally {
    await context.close();
  }
}

async function openGuide(page) {
  await page.goto(`${BASE}/automations`, { waitUntil: "domcontentloaded" });
  await page
    .getByTestId("super-admin-setup-guide-panel")
    .waitFor({ timeout: 30000 });
}

const closePanel = (page) =>
  page
    .getByTestId("super-admin-setup-guide-panel")
    .getByRole("button", { name: "Close" })
    .click();

(async () => {
  const browser = await chromium.launch({ channel: "chrome", headless: true });

  // Vite optimizes dependencies on the first request and reloads the page.
  await scenario(browser, async (page) => {
    await page.goto(`${BASE}/automations`, { waitUntil: "domcontentloaded" });
    await page.waitForTimeout(8000);
    await openGuide(page);
  });

  // The automation step succeeds. While the guide re-reads its progress the
  // admin closes the panel; the next step's tour then starts over the
  // question, and the admin closes the tour.
  await scenario(browser, async (page) => {
    await openGuide(page);
    await page.evaluate(() => {
      sessionStorage.setItem("automation-done", "1");
      window.dispatchEvent(
        new CustomEvent("oh-super-admin-setup-step", {
          detail: { id: "first-automation" },
        }),
      );
    });
    await closePanel(page);
    await page.getByTestId("confirmation-modal").waitFor({ timeout: 2000 });
    await look(page, "question-asked", true);

    await page.locator(".driver-popover").waitFor({ timeout: 20000 });
    await page.waitForTimeout(600);
    await look(page, "tour-started", true);

    await page.locator(".driver-popover-close-btn").click();
    await page.waitForTimeout(1000);
    await look(page, "tour-closed", true);
  });

  // Cancel leaves the guide on its pill.
  await scenario(browser, async (page) => {
    await openGuide(page);
    await closePanel(page);
    await page.getByTestId("cancel-button").click();
    await page.waitForTimeout(300);
    await look(page, "cancelled");
  });

  // Confirm turns the guide off.
  await scenario(browser, async (page) => {
    await openGuide(page);
    await closePanel(page);
    await page.getByTestId("confirm-button").click();
    await page.waitForTimeout(800);
    const seen = await look(page, "confirmed");
    const sent = await page.evaluate(() => sessionStorage.getItem("dismissed"));
    console.log(
      `[${LABEL}] dismissal sent: ${sent === "1"}, guide shown: ${seen.setupGuide}`,
    );
  });

  await browser.close();
})().catch((error) => {
  console.error("SMOKE FAILED:", error);
  process.exit(1);
});
