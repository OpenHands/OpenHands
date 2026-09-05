import { expect, test, type Page } from "@playwright/test";

type ProbeMeasurement = {
  storeCallMs: number;
  composerLatencyMs: number;
};

type FixtureMeasurement = {
  append: ProbeMeasurement;
  streaming: ProbeMeasurement[];
  tokenSpans: number;
  elements: number;
  longTasks: number[];
  codePreserved: boolean;
};

type ContextMeasurement = {
  context: number;
  control: FixtureMeasurement;
  longHistory: FixtureMeasurement;
};

const FIXTURE_CONVERSATION_ID = "pagination-local";
const BASE_URL = "http://localhost:3001/";
const RUN_PERFORMANCE_ACCEPTANCE =
  process.env.RUN_CONVERSATION_MARKDOWN_PERF === "1";

const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
};

async function runFixture(
  page: Page,
  historyCount: number,
  lineCount: number,
): Promise<FixtureMeasurement> {
  return page.evaluate(
    async ({ conversationId, fixtureHistoryCount, fixtureLineCount }) => {
      type FixtureStore = {
        getState: () => {
          clearEventsForConversation: (id: string) => void;
          addEvents: (events: Array<Record<string, unknown>>) => void;
          addEvent: (event: Record<string, unknown>) => void;
        };
      };

      const store = (window as unknown as { __OH_EVENT_STORE__?: FixtureStore })
        .__OH_EVENT_STORE__;
      if (!store) {
        throw new Error(
          "The performance fixture requires the mock-build event store seam.",
        );
      }

      const raf = () =>
        new Promise<void>((resolve) =>
          window.requestAnimationFrame(() => resolve()),
        );
      const code = Array.from(
        { length: fixtureLineCount },
        (_, line) =>
          "const item_" +
          String(line).padStart(4, "0") +
          ' = { alpha: "value-' +
          line +
          '", count: ' +
          line +
          ", active: true }; // deterministic fixture",
      ).join("\n");
      const base = Date.UTC(2026, 7, 29, 12, 0, 0);
      const events: Array<Record<string, unknown>> = [];

      for (let index = 0; index < fixtureHistoryCount; index += 1) {
        events.push({
          id: "fixture-user-" + index,
          timestamp: new Date(base + index * 2000).toISOString(),
          source: "user",
          llm_message: {
            role: "user",
            content: [
              {
                type: "text",
                text: "Run deterministic tool step " + index,
              },
            ],
          },
          activated_skills: [],
          extended_content: [],
        });
        events.push({
          id: "fixture-agent-" + index,
          timestamp: new Date(base + index * 2000 + 1000).toISOString(),
          source: "agent",
          llm_message: {
            role: "assistant",
            content: [
              {
                type: "text",
                text: [
                  "Tool output " + index,
                  "",
                  "```javascript",
                  code,
                  "```",
                ].join("\n"),
              },
            ],
          },
          activated_skills: [],
          extended_content: [],
        });
      }

      store.getState().clearEventsForConversation(conversationId);
      store.getState().addEvents(events);

      const expectedTokenSpans = fixtureHistoryCount * fixtureLineCount * 17;
      const renderDeadline = performance.now() + 30_000;
      while (
        document.querySelectorAll("span.token").length < expectedTokenSpans
      ) {
        if (performance.now() > renderDeadline) {
          throw new Error(
            "Timed out mounting the highlighted conversation fixture.",
          );
        }
        await raf();
      }
      await raf();

      const longTasks: number[] = [];
      const observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          longTasks.push(Number(entry.duration.toFixed(1)));
        }
      });
      observer.observe({ type: "longtask" });

      const probe = (event: Record<string, unknown>) => {
        const started = performance.now();
        const composerProbe = new Promise<number>((resolve) => {
          window.setTimeout(() => {
            const input = document.querySelector<HTMLElement>(
              '[data-testid="chat-input"]',
            );
            input?.focus();
            resolve(Number((performance.now() - started).toFixed(1)));
          }, 0);
        });
        const storeStarted = performance.now();
        store.getState().addEvent(event);
        const storeCallMs = Number(
          (performance.now() - storeStarted).toFixed(1),
        );
        return composerProbe.then((composerLatencyMs) => ({
          storeCallMs,
          composerLatencyMs,
        }));
      };

      const append = await probe({
        id: "fixture-plain-tail",
        timestamp: new Date(Date.UTC(2026, 7, 29, 14, 0, 0)).toISOString(),
        source: "agent",
        llm_message: {
          role: "assistant",
          content: [{ type: "text", text: "Final plain-text response." }],
        },
        activated_skills: [],
        extended_content: [],
      });
      await raf();

      const streaming: ProbeMeasurement[] = [];
      for (let update = 0; update < 5; update += 1) {
        streaming.push(
          await probe({
            id: "fixture-stream-" + update,
            timestamp: new Date(
              Date.UTC(2026, 7, 29, 15, 0, update),
            ).toISOString(),
            source: "agent",
            kind: "StreamingDeltaEvent",
            content: " streaming-" + update,
            reasoning_content: null,
          }),
        );
        await raf();
      }

      await raf();
      observer.disconnect();
      const lastLine =
        "const item_" + String(fixtureLineCount - 1).padStart(4, "0");

      return {
        append,
        streaming,
        tokenSpans: document.querySelectorAll("span.token").length,
        elements: document.getElementsByTagName("*").length,
        longTasks,
        codePreserved: document.body.textContent?.includes(lastLine) ?? false,
      };
    },
    {
      conversationId: FIXTURE_CONVERSATION_ID,
      fixtureHistoryCount: historyCount,
      fixtureLineCount: lineCount,
    },
  );
}

test("keeps code-heavy history responsive across tail updates", async ({
  browser,
  browserName,
}, testInfo) => {
  test.skip(
    !RUN_PERFORMANCE_ACCEPTANCE,
    "Set RUN_CONVERSATION_MARKDOWN_PERF=1 to run the isolated wall-clock benchmark.",
  );
  test.skip(
    browserName !== "chromium",
    "Performance acceptance is defined against Chromium.",
  );
  test.setTimeout(240_000);

  const measurements: ContextMeasurement[] = [];
  for (let contextIndex = 0; contextIndex < 3; contextIndex += 1) {
    const context = await browser.newContext({ baseURL: BASE_URL });
    await context.addInitScript(() => {
      window.localStorage.setItem("analytics-consent", "false");
      window.localStorage.setItem("openhands-telemetry-consent", "denied");
      window.localStorage.setItem("openhands-telemetry-first-use", "true");
      window.localStorage.setItem("openhands-onboarded", "1");
      window.localStorage.setItem(
        "openhands-backends",
        JSON.stringify([
          {
            id: "default-local",
            name: "Local",
            host: window.location.origin,
            apiKey: "",
            kind: "local",
          },
        ]),
      );
      window.localStorage.setItem(
        "openhands-active-backend",
        JSON.stringify({ backendId: "default-local", orgId: null }),
      );
    });

    const page = await context.newPage();
    await page.goto("/conversations/" + FIXTURE_CONVERSATION_ID, {
      waitUntil: "domcontentloaded",
    });
    await expect(page.getByTestId("chat-input")).toBeVisible({
      timeout: 30_000,
    });

    // The local telemetry prompt can cover the conversation even when the
    // stored preference is seeded. Close it before measuring or capturing UI
    // evidence so the modal is neither part of the timing nor the screenshot.
    try {
      const consentForm = page.getByTestId("telemetry-consent-form");
      await consentForm.waitFor({ state: "visible", timeout: 5_000 });
      await consentForm
        .getByRole("button", { name: "Confirm preferences" })
        .click();
      await consentForm.waitFor({ state: "hidden", timeout: 5_000 });
    } catch {
      // The prompt is absent when the seeded consent has already synchronized.
    }

    const control = await runFixture(page, 5, 10);
    const longHistory = await runFixture(page, 20, 50);
    expect(control.codePreserved).toBe(true);
    expect(longHistory.codePreserved).toBe(true);
    expect(control.tokenSpans).toBeGreaterThanOrEqual(850);
    expect(longHistory.tokenSpans).toBeGreaterThanOrEqual(17_000);
    await expect(
      page.getByText("Final plain-text response.", { exact: true }),
    ).toBeVisible();
    await expect(page.getByText(/streaming-4/)).toBeVisible();

    measurements.push({
      context: contextIndex + 1,
      control,
      longHistory,
    });

    if (contextIndex === 0) {
      const conversationEvidence = page.getByTestId("chat-scroll-container");
      if (process.env.PERFORMANCE_SCREENSHOT_PATH) {
        await conversationEvidence.screenshot({
          path: process.env.PERFORMANCE_SCREENSHOT_PATH,
        });
      }
      await testInfo.attach("long-history-after.png", {
        body: await conversationEvidence.screenshot(),
        contentType: "image/png",
      });
    }
    await context.close();
  }

  const controlAppends = measurements.map(
    ({ control }) => control.append.composerLatencyMs,
  );
  const longAppends = measurements.map(
    ({ longHistory }) => longHistory.append.composerLatencyMs,
  );
  const controlStreaming = measurements.flatMap(({ control }) =>
    control.streaming.map(({ composerLatencyMs }) => composerLatencyMs),
  );
  const longStreaming = measurements.flatMap(({ longHistory }) =>
    longHistory.streaming.map(({ composerLatencyMs }) => composerLatencyMs),
  );
  const appendRatio = median(longAppends) / Math.max(median(controlAppends), 1);
  const streamingRatio =
    median(longStreaming) / Math.max(median(controlStreaming), 1);
  const longUpdateMaximum = Math.max(...longAppends, ...longStreaming);
  const longTaskMaximum = Math.max(
    0,
    ...measurements.flatMap(({ longHistory }) => longHistory.longTasks),
  );

  const result = {
    measurements,
    summary: {
      controlAppendMedianMs: median(controlAppends),
      longAppendMedianMs: median(longAppends),
      appendRatio,
      controlStreamingMedianMs: median(controlStreaming),
      longStreamingMedianMs: median(longStreaming),
      streamingRatio,
      longUpdateMaximumMs: longUpdateMaximum,
      longTaskMaximumMs: longTaskMaximum,
    },
  };
  await testInfo.attach("performance-results.json", {
    body: Buffer.from(JSON.stringify(result, null, 2)),
    contentType: "application/json",
  });
  console.log("Issue #16910 performance results", JSON.stringify(result));

  expect(appendRatio).toBeLessThanOrEqual(2);
  expect(streamingRatio).toBeLessThanOrEqual(2);
  expect(longUpdateMaximum).toBeLessThanOrEqual(500);
  expect(longTaskMaximum).toBeLessThanOrEqual(500);
});
