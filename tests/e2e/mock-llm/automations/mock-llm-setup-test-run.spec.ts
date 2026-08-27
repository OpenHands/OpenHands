import { test, expect } from "@playwright/test";
import {
  seedLocalStorage,
  routeSessionApiKey,
  dismissAnalyticsModal,
  waitForTestId,
} from "../utils/mock-llm-helpers";

const DISPATCH_ROUTE = "**/api/automation/v1/*/dispatch";
const RUNS_ROUTE = "**/api/automation/v1/*/runs*";

function automationRun(
  id: string,
  status: "FAILED" | "COMPLETED",
  overrides: Record<string, unknown> = {},
) {
  const completedAt = new Date().toISOString();
  return {
    id,
    status,
    conversation_id: null,
    bash_command_id: null,
    error_detail: null,
    phase_code: status.toLowerCase(),
    phase_label:
      status === "COMPLETED" ? "Test run completed" : "Test run failed",
    phase_updated_at: completedAt,
    started_at: completedAt,
    completed_at: completedAt,
    ...overrides,
  };
}

/**
 * Exercise the setup-only test gate in a real browser against the real local
 * automation create/update/delete API. Dispatch/run responses are deterministic
 * at the browser boundary so both terminal UI states can be captured without
 * depending on public RSS availability or a live LLM during evidence capture.
 */
test("automation setup shows actionable failure and successful enable gate", async ({
  page,
}, testInfo) => {
  await seedLocalStorage(page);
  await routeSessionApiKey(page);

  await page.goto("/automations/templates", { waitUntil: "domcontentloaded" });
  await dismissAnalyticsModal(page);
  await waitForTestId(page, "recommended-automations-section", 15_000);

  await page.getByTestId("recommended-automation-card-news-digest").click();
  await expect(page.getByTestId("setup-dialog")).toBeVisible({
    timeout: 15_000,
  });

  // Daily news digest has safe defaults and no integration prerequisites. The
  // first continue validates the defaults; the second confirms creation.
  await page.getByTestId("setup-continue-button").click();
  await expect(page.getByTestId("setup-review")).toBeVisible({
    timeout: 15_000,
  });
  await page.getByTestId("setup-continue-button").click();
  await expect(page.getByTestId("setup-test-run")).toBeVisible({
    timeout: 30_000,
  });
  await expect(page.getByTestId("setup-test-run-ready")).toBeVisible();
  await expect(page.getByTestId("setup-finalize-button")).toHaveCount(0);

  const failed = automationRun("evidence-failed", "FAILED", {
    error_detail: "Simulated repository access failure for setup evidence",
  });

  await page.route(DISPATCH_ROUTE, async (route) => {
    await route.fulfill({
      status: 201,
      contentType: "application/json",
      body: JSON.stringify(failed),
    });
  });
  await page.route(RUNS_ROUTE, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ runs: [failed], total: 1 }),
    });
  });

  await page.getByTestId("setup-test-run-button").click();
  await expect(page.getByTestId("setup-test-run-error")).toContainText(
    "Simulated repository access failure for setup evidence",
  );
  await expect(page.getByTestId("setup-finalize-button")).toHaveCount(0);
  await page.screenshot({
    path: testInfo.outputPath("automation-test-run-failure.png"),
    fullPage: true,
  });

  await page.unroute(DISPATCH_ROUTE);
  await page.unroute(RUNS_ROUTE);

  const completed = automationRun("evidence-completed", "COMPLETED", {
    conversation_id: "evidence-conversation",
  });

  await page.route(DISPATCH_ROUTE, async (route) => {
    await route.fulfill({
      status: 201,
      contentType: "application/json",
      body: JSON.stringify(completed),
    });
  });
  await page.route(RUNS_ROUTE, async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ runs: [completed], total: 1 }),
    });
  });

  await page.getByTestId("setup-test-run-button").click();
  await expect(page.getByTestId("setup-test-run-success")).toBeVisible();
  await expect(page.getByTestId("setup-finalize-button")).toBeVisible();
  await expect(page.getByTestId("setup-test-run-conversation")).toHaveAttribute(
    "href",
    "/conversations/evidence-conversation",
  );
  await page.screenshot({
    path: testInfo.outputPath("automation-test-run-success.png"),
    fullPage: true,
  });

  // Do not leave an enabled evidence automation behind. Back from the test step
  // deletes only the setup-owned disabled draft and returns to editable setup.
  await page.getByTestId("setup-back-button").click();
  await expect(page.getByTestId("setup-test-run")).toHaveCount(0);
});
