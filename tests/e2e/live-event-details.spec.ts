import { expect, test } from "@playwright/test";

// Import the live helper without configuring or contacting a live backend.
test("expands collapsed event groups and their nested details without closing open groups", async ({
  page,
}) => {
  process.env.LIVE_E2E_SESSION_API_KEY ??= "event-details-regression";
  const { expandVisibleEventDetails } =
    await import("./live/utils/agent-server-conversation");

  await page.setContent(`
    <button data-testid="event-group-toggle" aria-expanded="false"
      aria-label="Expand actions" onclick="this.setAttribute('aria-expanded', 'true'); document.querySelector('#details').hidden = false">
      Run the specified bash command
    </button>
    <div id="details" hidden>
      <button aria-label="Expand" onclick="this.setAttribute('aria-label', 'Collapse'); document.querySelector('#output').hidden = false">Terminal</button>
      <pre id="output" hidden>command output</pre>
    </div>
    <button data-testid="event-group-toggle" aria-expanded="true"
      onclick="this.setAttribute('aria-expanded', 'false')">Already open</button>
    <button data-testid="event-group-toggle" aria-expanded="false" hidden
      onclick="throw new Error('Hidden group must not be clicked')">Hidden group</button>
  `);

  await expandVisibleEventDetails(page);

  await expect(page.locator("#output")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Already open" }),
  ).toHaveAttribute("aria-expanded", "true");
  await expandVisibleEventDetails(page);
  await expect(page.locator("#output")).toBeVisible();
});
