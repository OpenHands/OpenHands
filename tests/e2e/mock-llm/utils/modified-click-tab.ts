import type { Locator, Page, TestInfo } from "@playwright/test";

/** Observe one real modified click without changing its browser behavior. */
export async function openModifiedClickTab(
  link: Locator,
  testInfo: Pick<TestInfo, "attach">,
  timeoutMs = 5_000,
): Promise<Page> {
  const page = link.page();
  const context = page.context();
  const before = {
    sourceUrl: page.url(),
    expectedHref: await link.getAttribute("href"),
    modifiers: ["ControlOrMeta"],
    pages: context.pages().map((tab) => tab.url()),
    timeoutMs,
  };

  await page.evaluate(() => {
    const state = {
      events: [] as {
        event: MouseEvent;
        target: { tag: string; id: string; testId: string | null } | null;
        anchor: { href: string | null; target: string | null } | null;
      }[],
      listener: (event: MouseEvent) => {
        const target = event.target instanceof Element ? event.target : null;
        const anchor = target?.closest("a");
        state.events.push({
          event,
          target: target
            ? {
                tag: target.tagName,
                id: target.id,
                testId: target.getAttribute("data-testid"),
              }
            : null,
          anchor: anchor
            ? {
                href: anchor.getAttribute("href"),
                target: anchor.getAttribute("target"),
              }
            : null,
        });
      },
    };
    // Keep the event itself until dispatch finishes, so defaultPrevented is
    // accurate even when a descendant stops propagation before window bubbles.
    const observedWindow = window as typeof window & {
      __openhandsModifiedClick?: typeof state;
    };
    observedWindow.__openhandsModifiedClick = state;
    window.addEventListener("click", state.listener, true);
  });

  // Handle rejection immediately: a failed click must not leave an unhandled
  // page-event timeout behind it. There is one click and no retry.
  const newTabPromise = context
    .waitForEvent("page", { timeout: timeoutMs })
    .then(
      (newTab) => ({ newTab }),
      (error: unknown) => ({ error }),
    );
  let clickError: unknown;
  try {
    await link.click({ modifiers: ["ControlOrMeta"], timeout: timeoutMs });
  } catch (error) {
    clickError = error;
  }
  const outcome = await newTabPromise;

  let observed: unknown;
  try {
    observed = await page.evaluate(() => {
      const observedWindow = window as typeof window & {
        __openhandsModifiedClick?: {
          events: {
            event: MouseEvent;
            target: unknown;
            anchor: unknown;
          }[];
          listener: (event: MouseEvent) => void;
        };
      };
      const state = observedWindow.__openhandsModifiedClick;
      if (!state)
        return { error: "click observer unavailable after navigation" };
      window.removeEventListener("click", state.listener, true);
      delete observedWindow.__openhandsModifiedClick;
      return {
        userAgent: navigator.userAgent,
        events: state.events.map(({ event, target, anchor }) => ({
          target,
          anchor,
          type: event.type,
          button: event.button,
          ctrlKey: event.ctrlKey,
          metaKey: event.metaKey,
          shiftKey: event.shiftKey,
          altKey: event.altKey,
          isTrusted: event.isTrusted,
          defaultPrevented: event.defaultPrevented,
        })),
      };
    });
  } catch {
    observed = { error: "click observer unavailable because the page closed" };
  }
  await testInfo.attach("modified-click-diagnostics.json", {
    body: Buffer.from(
      JSON.stringify(
        {
          before,
          observed,
          after: {
            sourceUrl: page.url(),
            pages: context.pages().map((tab) => tab.url()),
            openedTab: "newTab" in outcome,
            clickFailed: clickError !== undefined,
          },
        },
        null,
        2,
      ),
    ),
    contentType: "application/json",
  });

  if (clickError !== undefined) {
    throw new Error("cmd-click failed before opening a new tab", {
      cause: clickError,
    });
  }
  if ("error" in outcome) {
    throw new Error(`cmd-click opened no new tab within ${timeoutMs} ms`, {
      cause: outcome.error,
    });
  }
  return outcome.newTab;
}
