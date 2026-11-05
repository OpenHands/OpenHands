// Run with VITE_MOCK_API=true react-router dev on port 3193.
// Synthetic history is inserted into the real Canvas event store; no LLM runs.
import { chromium } from "@playwright/test";
import { mkdir } from "node:fs/promises";
const mode = process.argv[2] ?? "after";
const output = process.env.PROOF_OUTPUT ?? "/tmp/oh-thinking-investigation";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, channel: "chrome" });
const context = await browser.newContext({
  viewport: { width: 1280, height: 960 },
  recordVideo: { dir: output, size: { width: 1280, height: 960 } },
});
const page = await context.newPage();
page.setDefaultTimeout(15000);
page.setDefaultNavigationTimeout(30000);
page.on("pageerror", (e) => console.log("page error", e.message));
const errors = [];
page.on("console", (m) => {
  if (m.type() === "error" && m.text().includes("same key"))
    errors.push(m.text());
});
await page.addInitScript(() => {
  localStorage.setItem("openhands-onboarded", "1");
  localStorage.setItem(
    "openhands-backends",
    JSON.stringify([
      {
        id: "default-local",
        name: "Synthetic history fixture",
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
  localStorage.setItem("analytics-consent", "false");
  localStorage.setItem("openhands-telemetry-consent", "denied");
  localStorage.setItem("openhands-telemetry-first-use", "true");
});
try {
  await page.goto("http://127.0.0.1:3193/conversations/2", {
    waitUntil: "domcontentloaded",
  });
  console.log("loaded page");
  await page.getByTestId("chat-scroll-container").waitFor({ timeout: 15000 });
  await page.waitForTimeout(2000);
  const consent = page.getByRole("button", { name: "Confirm preferences" });
  if (await consent.isVisible()) {
    await page.getByRole("checkbox").uncheck();
    await consent.click();
  }
  await page.evaluate(async () => {
    const { useEventStore } = await import("/src/stores/use-event-store.ts");
    const events = [];
    let seq = 0;
    const stamp = () =>
      new Date(Date.UTC(2026, 10, 5, 12, 0, seq++)).toISOString();
    for (let i = 0; i < 8; i++) {
      const id = `old-${i}`;
      events.push({
        id,
        timestamp: stamp(),
        source: "agent",
        thought: [{ type: "text", text: `Checking step ${i + 1}` }],
        reasoning_content: `Reasoning for step ${i + 1}`,
        thinking_blocks: [],
        action: {
          kind: "ExecuteBashAction",
          command: "pwd",
          is_input: false,
          timeout: null,
          reset: false,
        },
        tool_name: "execute_bash",
        tool_call_id: `call_${id}`,
        tool_call: {
          id: `call_${id}`,
          type: "function",
          function: { name: "execute_bash", arguments: '{"command":"pwd"}' },
        },
        llm_response_id: `response_${id}`,
        security_risk: "LOW",
      });
      events.push({
        id: `obs-${i}`,
        timestamp: stamp(),
        source: "environment",
        tool_name: "execute_bash",
        tool_call_id: `call_${id}`,
        action_id: id,
        observation: {
          kind: "ExecuteBashObservation",
          content: [{ type: "text", text: "/workspace/project" }],
          command: "pwd",
          exit_code: 0,
          error: false,
          timeout: false,
          metadata: {},
        },
      });
      events.push({
        id: `prompt-${i}`,
        timestamp: stamp(),
        source: "user",
        llm_message: {
          role: "user",
          content: [{ type: "text", text: "Continue to the next step." }],
        },
        activated_skills: [],
        extended_content: [],
      });
    }
    useEventStore.getState().clearEvents();
    useEventStore.getState().addEvents(events);
  });
  await page.getByTestId("event-group-toggle").first().waitFor();
  await page.waitForTimeout(600);
  await page.evaluate(async () => {
    const { useEventStore } = await import("/src/stores/use-event-store.ts");
    useEventStore.getState().clearEvents();
    useEventStore
      .getState()
      .addEvents([
        {
          id: "new-prompt",
          timestamp: "2026-11-05T13:00:00Z",
          source: "user",
          llm_message: {
            role: "user",
            content: [
              { type: "text", text: "This is the newly loaded conversation." },
            ],
          },
          activated_skills: [],
          extended_content: [],
        },
      ]);
  });
  await page.getByText("This is the newly loaded conversation.").waitFor();
  await page.getByTestId("chat-scroll-container").evaluate((el) => {
    el.scrollTop = 0;
  });
  await page.waitForTimeout(600);
  const ghosts = page.getByTestId("event-group-toggle");
  const ghostCount = await ghosts.count();
  let expandedAfterClick = null;
  if (ghostCount) {
    await ghosts.first().click();
    expandedAfterClick = await ghosts.first().getAttribute("aria-expanded");
  }
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${output}/${mode}.png` });
  console.log(
    JSON.stringify({
      mode,
      ghostCount,
      expandedAfterClick,
      duplicateKeyWarnings: errors.length,
    }),
  );
  if (
    mode === "before"
      ? ghostCount !== 8 || expandedAfterClick !== "false"
      : ghostCount !== 0
  )
    throw new Error("Unexpected reproduction result");
} catch (error) {
  console.log("capture failure");
  await page
    .screenshot({ path: `${output}/${mode}-error.png`, timeout: 5000 })
    .catch(() => {});
  console.log((await page.locator("body").innerText()).slice(0, 2000));
  throw error;
} finally {
  await context.close();
  const videoPath = await page.video().path();
  console.log(JSON.stringify({ videoPath }));
  await browser.close();
}
