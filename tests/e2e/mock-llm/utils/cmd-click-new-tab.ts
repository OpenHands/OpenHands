import type { BrowserContext, Locator, Page } from "@playwright/test";

export interface ClickDiagnostics {
  targetTag?: string;
  targetTestId?: string | null;
  anchorHref?: string | null;
  ctrlKey?: boolean;
  metaKey?: boolean;
  defaultPrevented?: boolean;
}

/**
 * Click an element expected to open a new tab with modifier keys (Cmd/Ctrl),
 * bounding the tab-wait by a short timeout rather than the global test timeout.
 * Attaches a capture click listener to record diagnostics (target, modifiers,
 * defaultPrevented) so that if no tab opens, the test fails fast with recorded evidence.
 */
export async function cmdClickAndExpectNewTab(
  context: BrowserContext,
  cardLocator: Locator,
  options?: { timeout?: number },
): Promise<Page> {
  const timeout = options?.timeout ?? 15_000;
  const page = cardLocator.page();

  await page.evaluate(() => {
    (
      window as unknown as {
        __lastCmdClickDiagnostics?: ClickDiagnostics | null;
      }
    ).__lastCmdClickDiagnostics = null;

    window.addEventListener(
      "click",
      (event) => {
        const target = event.target as HTMLElement | null;
        const anchor = target?.closest?.("a");
        (
          window as unknown as {
            __lastCmdClickDiagnostics?: ClickDiagnostics | null;
          }
        ).__lastCmdClickDiagnostics = {
          targetTag: target?.tagName,
          targetTestId: target?.getAttribute?.("data-testid") ?? null,
          anchorHref: anchor?.getAttribute?.("href") ?? null,
          ctrlKey: event.ctrlKey,
          metaKey: event.metaKey,
          defaultPrevented: event.defaultPrevented,
        };
      },
      { capture: true, once: true },
    );
  });

  const newTabPromise = context.waitForEvent("page", { timeout });
  await cardLocator.click({ modifiers: ["ControlOrMeta"] });

  try {
    return await newTabPromise;
  } catch (error) {
    const diagnostics = await page
      .evaluate(
        () =>
          (
            window as unknown as {
              __lastCmdClickDiagnostics?: ClickDiagnostics | null;
            }
          ).__lastCmdClickDiagnostics,
      )
      .catch(() => null);

    throw new Error(
      `cmd-click opened no new tab within ${timeout}ms. Recorded click diagnostics: ${JSON.stringify(diagnostics)}. ${(error as Error)?.message ?? ""}`,
    );
  }
}
