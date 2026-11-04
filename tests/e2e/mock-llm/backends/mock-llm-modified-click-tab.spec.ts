import { expect, test } from "@playwright/test";
import { openModifiedClickTab } from "../utils/modified-click-tab";

test("a modified link click opens one tab and records the delivered event", async ({
  page,
}, testInfo) => {
  // Arrange: use the browser's native link behavior, without an app/backend.
  await page.setContent(
    '<a href="https://example.test/conversations/123?backend=A"><span id="title">Conversation</span></a>',
  );
  await page
    .context()
    .route("https://example.test/**", (route) =>
      route.fulfill({ body: "Owning backend A" }),
    );

  // Act: the same helper used by the full cross-connect regression.
  const newTab = await openModifiedClickTab(page.locator("a"), testInfo);

  // Assert: browser navigation and its diagnostic attachment stay intact.
  await expect(newTab).toHaveURL(
    "https://example.test/conversations/123?backend=A",
  );
  const attachment = testInfo.attachments.find(
    (item) => item.name === "modified-click-diagnostics.json",
  );
  const diagnostics = JSON.parse(attachment!.body!.toString());
  expect(diagnostics.observed.events).toEqual([
    expect.objectContaining({
      anchor: {
        href: "https://example.test/conversations/123?backend=A",
        target: null,
      },
      isTrusted: true,
      defaultPrevented: false,
    }),
  ]);
});

test("a consumed click fails quickly and records its actual target and final prevention", async ({
  page,
}, testInfo) => {
  test.setTimeout(15_000);
  // Arrange: deliberately prevent a new tab, including bubble propagation.
  await page.setContent(
    '<a href="https://example.test/conversations/123?backend=A"><span id="title">Conversation</span></a>',
  );
  await page.locator("#title").evaluate((title) => {
    title.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
    });
  });

  // Act / Assert: bound only this new-tab step, preserving the suite budget.
  const startedAt = Date.now();
  await expect(
    openModifiedClickTab(page.locator("a"), testInfo),
  ).rejects.toThrow("cmd-click opened no new tab within 5000 ms");
  expect(Date.now() - startedAt).toBeLessThan(10_000);
  const attachment = testInfo.attachments.find(
    (item) => item.name === "modified-click-diagnostics.json",
  );
  const diagnostics = JSON.parse(attachment!.body!.toString());
  expect(diagnostics.observed.events).toEqual([
    expect.objectContaining({
      target: { tag: "SPAN", id: "title", testId: null },
      anchor: {
        href: "https://example.test/conversations/123?backend=A",
        target: null,
      },
      defaultPrevented: true,
    }),
  ]);
  expect(diagnostics.after.openedTab).toBe(false);
});

test("an obscured link reports click failure and handles its pending page waiter", async ({
  page,
}, testInfo) => {
  test.setTimeout(5_000);
  // Arrange: a real overlay prevents Playwright from delivering the click.
  await page.setContent(`
    <a href="https://example.test/conversations/123?backend=A">Conversation</a>
    <div style="position:fixed;inset:0;background:white">Blocking overlay</div>
  `);

  // Act: both the actionability failure and page wait remain bounded.
  await expect(
    openModifiedClickTab(page.locator("a"), testInfo, 300),
  ).rejects.toThrow("cmd-click failed before opening a new tab");

  // Assert: diagnostics distinguish non-delivery from prevented dispatch.
  const attachment = testInfo.attachments.find(
    (item) => item.name === "modified-click-diagnostics.json",
  );
  const diagnostics = JSON.parse(attachment!.body!.toString());
  expect(diagnostics.observed.events).toEqual([]);
  expect(diagnostics.after).toMatchObject({
    openedTab: false,
    clickFailed: true,
  });
});
