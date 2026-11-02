import { expect, test } from "@playwright/test";
import {
  BACKEND_URL,
  SESSION_API_KEY,
  seedLocalStorage,
  routeSessionApiKey,
} from "../utils/mock-llm-helpers";

// Uses the real automation backend. No LLM invocation or scheduled job is needed.
for (const language of ["English", "Arabic"]) {
  test(`detail header actions fit phone and desktop in ${language}`, async ({
    page,
    request,
  }) => {
    await seedLocalStorage(page);
    await routeSessionApiKey(page);
    await page.setViewportSize({ width: 1440, height: 1000 });
    const headers = { "X-Session-API-Key": SESSION_API_KEY };
    const created = await request.post(
      `${BACKEND_URL}/api/automation/v1/preset/prompt`,
      {
        headers,
        data: {
          name: "QA_Inert_Event_Detail",
          prompt: "No execution needed. Do not run tools.",
          trigger: {
            type: "event",
            source: "qa-never-delivered",
            on: "qa.never",
          },
          enabled: false,
        },
      },
    );
    expect(created.ok()).toBe(true);
    const automation = await created.json();
    try {
      const settings = await request.patch(`${BACKEND_URL}/api/settings`, {
        headers,
        data: {
          misc_settings_diff: {
            app_preferences: { language: language === "Arabic" ? "ar" : "en" },
          },
        },
      });
      expect(settings.ok()).toBe(true);
      await page.goto(`/automations/${automation.id}`);
      const heading = page.getByRole("heading", {
        name: "QA_Inert_Event_Detail",
        exact: true,
      });
      await expect(heading).toBeVisible();
      await expect(page.locator("html")).toHaveAttribute(
        "dir",
        language === "Arabic" ? "rtl" : "ltr",
      );
      for (const width of [390, 1440]) {
        await page.setViewportSize({ width, height: 1000 });
        const toggle = page.getByRole("switch");
        const header = heading.locator("../..");
        const boxes = await header
          .getByRole("button")
          .or(toggle)
          .evaluateAll((controls) =>
            controls.map((control) => {
              const r = control.getBoundingClientRect();
              return { left: r.left, right: r.right, width: r.width };
            }),
          );
        expect(boxes.length).toBeGreaterThanOrEqual(3);
        expect(
          boxes.every(
            (box) => box.left >= 0 && box.right <= width && box.width > 0,
          ),
        ).toBe(true);
        await toggle.click();
        await expect(toggle).toHaveAttribute("aria-checked", "true");
        await toggle.click();
        await expect(toggle).toHaveAttribute("aria-checked", "false");
      }
    } finally {
      await request.delete(
        `${BACKEND_URL}/api/automation/v1/${automation.id}`,
        { headers },
      );
      await request.patch(`${BACKEND_URL}/api/settings`, {
        headers,
        data: { misc_settings_diff: { app_preferences: { language: "en" } } },
      });
    }
  });
}
