import { expect, test, type Locator, type Page } from "@playwright/test";

const DISPOSED_MODEL_ERROR =
  "TextModel got disposed before DiffEditorWidget model got reset";

async function openTab(page: Page, tab: "commits" | "planner") {
  const button = page.getByTestId(`conversation-tab-${tab}`);
  if (await button.isVisible()) {
    await button.click();
  } else {
    await page
      .getByTestId("tabs-pane-header")
      .getByTestId("ellipsis-button")
      .click();
    await page.getByTestId(`conversation-tabs-menu-open-${tab}`).click();
  }
  await expect(button).toHaveAttribute("aria-selected", "true");
  if (tab === "planner") {
    await expect(page.getByRole("tabpanel", { name: "Planner" })).toBeVisible();
    await expect(
      page.getByRole("tabpanel", { name: "Planner" }),
    ).not.toBeEmpty();
  }
}

async function openChanges(page: Page) {
  await openTab(page, "commits");
  const changes = page.getByTestId("uncommitted-changes-row-toggle");
  await expect(changes).toBeVisible();
  await changes.click();
  await expect(changes).toHaveAttribute("aria-expanded", "true");
}

function fileRow(page: Page, path: string) {
  return page.getByTestId("file-diff-viewer-outer").filter({ hasText: path });
}

async function expectModifiedDiff(row: Locator) {
  const editor = row.locator(".monaco-diff-editor");
  await expect(editor).toBeVisible();
  await expect(
    editor.locator(".original .view-lines[role=presentation]"),
  ).toContainText("def greet(name):");
  await expect(
    editor.locator(".modified .view-lines[role=presentation]"),
  ).toContainText("Welcome.");
}

async function inMemoryModelCount(page: Page) {
  return page.evaluate(() => {
    const { monaco } = window as unknown as {
      monaco: typeof import("monaco-editor");
    };
    // Monaco's language service can retain library models such as lib.dom.d.ts.
    // FileDiffViewer owns in-memory models; those must be released.
    return monaco.editor
      .getModels()
      .filter((model) => model.uri.scheme === "inmemory").length;
  });
}

for (const viewport of [
  { width: 1440, height: 1000 },
  { width: 390, height: 844 },
]) {
  test(`expanded diff teardown and remount at ${viewport.width}×${viewport.height}`, async ({
    page,
  }) => {
    test.setTimeout(90_000);
    await page.setViewportSize(viewport);

    // Arrange: the app's MSW fixtures supply Git data, but React, Monaco,
    // accordion animations and tab transitions all execute their real code.
    const browserErrors: string[] = [];
    page.on("pageerror", (error) => {
      browserErrors.push(error.message);
    });
    page.on("console", (message) => {
      if (
        message.type() === "error" &&
        message.text().includes(DISPOSED_MODEL_ERROR)
      ) {
        browserErrors.push(message.text());
      }
    });
    await page.addInitScript(() => {
      localStorage.setItem("analytics-consent", "false");
      localStorage.setItem("openhands-telemetry-consent", "denied");
      localStorage.setItem("openhands-telemetry-first-use", "true");
      localStorage.setItem("openhands-onboarded", "1");
      localStorage.setItem(
        "openhands-backends",
        JSON.stringify([
          {
            id: "default-local",
            name: "Local QA",
            host: location.origin,
            apiKey: "",
            kind: "local",
          },
        ]),
      );
      localStorage.setItem(
        "openhands-active-backend",
        JSON.stringify({ backendId: "default-local", orgId: null }),
      );
    });
    await page.goto("/conversations/table-demo");
    const consent = page.getByRole("dialog", {
      name: "Help improve OpenHands",
    });
    await expect(consent).toBeVisible({ timeout: 30_000 });
    await consent.getByRole("checkbox").uncheck();
    await consent.getByRole("button", { name: "Confirm preferences" }).click();
    await expect(consent).toBeHidden();
    const panelToggle = page.getByTestId("right-panel-toggle");
    await expect(panelToggle).toBeVisible();
    if ((await panelToggle.getAttribute("aria-pressed")) !== "true") {
      await panelToggle.click();
    }

    const modified = fileRow(page, "src/components/hello.tsx");
    const added = fileRow(page, "src/utils/new-helper.ts");
    const expectReleased = async () => {
      await expect(page.locator(".monaco-diff-editor")).toHaveCount(0);
      await expect.poll(() => inMemoryModelCount(page)).toBe(0);
      expect(browserErrors).toEqual([]);
    };

    // Act / assert: every round-trip mounts a fresh real DiffEditor, checks
    // both rendered sides and lets Planner unmount the whole viewer subtree.
    for (let round = 0; round < 10; round += 1) {
      await openChanges(page);
      await modified.getByTestId("collapse").click();
      await expectModifiedDiff(modified);
      await openTab(page, "planner");
      await expectReleased();
    }

    // The nested teardown paths also dispose the same editor: collapsing,
    // opening another file, and replacing the diff with a single-file view.
    await openChanges(page);
    await modified.getByTestId("collapse").click();
    await expectModifiedDiff(modified);
    await modified.getByTestId("collapse").click();
    await expectReleased();
    await modified.getByTestId("collapse").click();
    await expectModifiedDiff(modified);
    await added.getByTestId("collapse").click();
    await expect(modified.locator(".monaco-diff-editor")).toHaveCount(0);
    await expect(added.locator(".monaco-diff-editor")).toBeVisible();
    await expect(
      added.locator(".modified .view-lines[role=presentation]"),
    ).toContainText("Welcome.");
    await openTab(page, "planner");
    await expectReleased();

    await openChanges(page);
    await modified.getByTestId("collapse").click();
    await expectModifiedDiff(modified);
    await modified.getByTestId("view-mode-new").click();
    await expect(modified.locator(".monaco-diff-editor")).toHaveCount(0);
    await expect(
      modified.locator(".monaco-editor .view-lines[role=presentation]"),
    ).toContainText("Welcome.");
    await modified.getByTestId("view-mode-diff").click();
    await expectModifiedDiff(modified);
    await openTab(page, "planner");
    await expectReleased();

    if (viewport.width === 1440) {
      // Leave during initialization, before the wrapper's onMount effect.
      // Observe real editor creation and schedule the actual Planner control;
      // waiting for rendered diff contents would miss this teardown window.
      await openChanges(page);
      await expect(page.getByTestId("conversation-tab-planner")).toBeVisible();
      await page.evaluate(() => {
        const { monaco } = window as unknown as {
          monaco: typeof import("monaco-editor");
        };
        const subscription = monaco.editor.onDidCreateDiffEditor(() => {
          subscription.dispose();
          requestAnimationFrame(() => {
            document
              .querySelector<HTMLButtonElement>(
                '[data-testid="conversation-tab-planner"]',
              )!
              .click();
          });
        });
      });
      await modified.getByTestId("collapse").click();
      await expect(
        page.getByRole("tabpanel", { name: "Planner" }),
      ).toBeVisible();
      await expect(page.locator(".monaco-diff-editor")).toHaveCount(0);
      await expect.poll(() => inMemoryModelCount(page)).toBe(0);
      // This deliberately timed pending-computation exit also raises Monaco's
      // existing "no diff result available" before and after the fix. Permit
      // only that message here; ordinary transitions and recovery reject all errors.
      expect(
        browserErrors
          .splice(0)
          .filter((message) => message !== "no diff result available"),
      ).toEqual([]);
      await openChanges(page);
      await modified.getByTestId("collapse").click();
      await expectModifiedDiff(modified);
      await openTab(page, "planner");
      await expectReleased();
    }
  });
}
