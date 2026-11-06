// Start VITE_MOCK_API=true react-router dev on port 3195, then run:
// PROOF_OUTPUT=/tmp/resume-proof node .pr/reproduce-resume-position.mjs before|after
// Synthetic history and tab-local session state; no backend or LLM runs.
import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";

const mode = process.argv[2] ?? "after";
const output = process.env.PROOF_OUTPUT ?? "/tmp/resume-proof";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, channel: "chrome" });
const context = await browser.newContext({
  timezoneId: "America/New_York",
  viewport: { width: 1280, height: 960 },
  recordVideo: { dir: output, size: { width: 1280, height: 960 } },
});
const page = await context.newPage();
page.setDefaultTimeout(15000);
await page.addInitScript(() => {
  localStorage.setItem("openhands-onboarded", "1");
  localStorage.setItem("openhands-telemetry-consent", "denied");
  localStorage.setItem("openhands-telemetry-first-use", "true");
  localStorage.setItem("analytics-consent", "false");
  localStorage.setItem(
    "openhands-backends",
    JSON.stringify([
      {
        id: "default-local",
        name: "Resume positioning fixture (mock)",
        host: location.origin,
        apiKey: "mock-session-key",
        kind: "local",
      },
    ]),
  );
  localStorage.setItem(
    "openhands-active-backend",
    JSON.stringify({ backendId: "default-local", orgId: null }),
  );
});
try {
  await page.goto("http://127.0.0.1:3195/conversations/2", {
    waitUntil: "domcontentloaded",
  });
  await page.getByTestId("chat-scroll-container").waitFor();
  await page.waitForTimeout(2000);
  const consent = page.getByRole("button", { name: "Confirm preferences" });
  if (await consent.isVisible()) {
    await page.getByRole("checkbox").uncheck();
    await consent.click();
  }
  await page.evaluate(async () => {
    const { useEventStore } = await import("/src/stores/use-event-store.ts");
    const { useSessionActivityStore, sessionActivityKey } =
      await import("/src/stores/session-activity-store.ts");
    const key = sessionActivityKey("default-local", null, "2");
    useEventStore.getState().clearEvents();
    useSessionActivityStore.setState({
      entries: {
        [key]: [
          {
            id: "initialize",
            kind: "initialize",
            startedAt: Date.parse("2026-11-05T18:32:35Z"),
            phase: "complete",
            steps: [],
            open: false,
          },
          {
            id: "resume",
            kind: "resume",
            startedAt: Date.parse("2026-11-05T23:20:00Z"),
            phase: "pending",
            steps: [],
            open: false,
          },
        ],
      },
    });
  });
  await page.waitForTimeout(1000);
  await page.evaluate(async () => {
    const { useEventStore } = await import("/src/stores/use-event-store.ts");
    const rows = [
      [
        "first-user",
        "user",
        "2026-11-05T18:32:35.376752",
        "Hello, this is the original conversation.",
      ],
      [
        "first-reply",
        "agent",
        "2026-11-05T18:32:44.416381",
        "Earlier assistant reply, from before the sandbox paused.",
      ],
      [
        "last-user",
        "user",
        "2026-11-05T18:38:51.913504",
        "This is my latest message in the saved history.",
      ],
      [
        "last-reply",
        "agent",
        "2026-11-05T18:39:10.417648",
        "Last saved reply. The sandbox is being resumed several hours later.",
      ],
    ];
    useEventStore.getState().addEvents(
      rows.map(([id, source, timestamp, text]) => ({
        id,
        source,
        timestamp,
        kind: "MessageEvent",
        llm_message: {
          role: source === "user" ? "user" : "assistant",
          content: [{ type: "text", text }],
        },
        activated_skills: [],
        extended_content: [],
      })),
    );
  });
  await page.getByText("Last saved reply.", { exact: false }).waitFor();
  await page.waitForTimeout(1500);
  await page.evaluate(async () => {
    const { useSessionActivityStore, sessionActivityKey } =
      await import("/src/stores/session-activity-store.ts");
    const key = sessionActivityKey("default-local", null, "2");
    useSessionActivityStore.getState().observe(key, "READY");
    useSessionActivityStore.getState().complete(key, "resume");
  });
  const resume = page.getByText("Resumed session", { exact: true });
  await resume.waitFor();
  const resumeBox = await resume.boundingBox();
  const firstReply = await page
    .getByText("Earlier assistant reply,", { exact: false })
    .boundingBox();
  const lastReply = await page
    .getByText("Last saved reply.", { exact: false })
    .boundingBox();
  const correctlyPlaced = resumeBox.y > lastReply.y;
  if (mode === "before" ? resumeBox.y >= firstReply.y : !correctlyPlaced) {
    throw new Error(
      `Unexpected placement: ${JSON.stringify({ resumeBox, firstReply, lastReply })}`,
    );
  }
  await page.screenshot({ path: `${output}/${mode}.png` });
  await page.waitForTimeout(1500);
  console.log(
    JSON.stringify({
      mode,
      correctlyPlaced,
      resumeY: resumeBox.y,
      lastReplyY: lastReply.y,
    }),
  );
} catch (error) {
  await page.screenshot({ path: `${output}/${mode}-error.png` });
  console.log(await page.locator("body").innerText());
  throw error;
} finally {
  await context.close();
  console.log(JSON.stringify({ video: await page.video().path() }));
  await browser.close();
}
