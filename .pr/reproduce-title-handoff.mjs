// VITE_MOCK_API=true react-router dev --port 3196
// PROOF_OUTPUT=/tmp/title-proof node .pr/reproduce-title-handoff.mjs before|after
import { chromium } from "@playwright/test";
import { mkdir, writeFile } from "node:fs/promises";
const mode = process.argv[2] ?? "after";
const output = process.env.PROOF_OUTPUT ?? "/tmp/title-proof";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ channel: "chrome", headless: true });
const context = await browser.newContext({
  viewport: { width: 1280, height: 900 },
  recordVideo: { dir: output, size: { width: 1280, height: 900 } },
});
const page = await context.newPage();
await page.addInitScript(() => {
  localStorage.setItem("openhands-onboarded", "1");
  localStorage.setItem(
    "openhands-backends",
    JSON.stringify([
      {
        id: "default-local",
        name: "Mock",
        host: location.origin,
        apiKey: "mock",
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
  await page.goto("http://127.0.0.1:3196/conversations/2", {
    waitUntil: "domcontentloaded",
  });
  await page.getByTestId("chat-scroll-container").waitFor();
  await page.waitForTimeout(1500);
  const consent = page.getByRole("button", { name: "Confirm preferences" });
  if (await consent.isVisible()) {
    await page.getByRole("checkbox").uncheck();
    await consent.click();
  }
  await page.evaluate(async () => {
    const { queryClient } = await import("/src/query-client-config.ts");
    const { CONVERSATION_QUERY_KEYS: keys } =
      await import("/src/hooks/query/query-keys.ts");
    const { useConversationLaunchStore: store } =
      await import("/src/stores/conversation-launch-store.ts");
    const task = {
      id: "title-proof-task",
      status: "WAITING_FOR_SANDBOX",
      app_conversation_id: "title-proof-chat",
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      request: { title: "Check the startup handoff" },
      title_generation: {
        status: "generated",
        title: "Review the startup handoff",
        deadline_at: new Date(Date.now() + 8000).toISOString(),
      },
    };
    queryClient.setQueryData(keys.startTask(task.id), task);
    const launch = store
      .getState()
      .begin({
        backendId: "default-local",
        orgId: null,
        files: [],
        message: "Check the startup handoff",
      });
    store
      .getState()
      .update(launch, {
        targetId: `task-${task.id}`,
        conversationId: task.app_conversation_id,
      });
  });
  const title = page
    .getByTestId("conversation-card-title")
    .filter({ hasText: "Review the startup handoff" });
  await title.waitFor();
  await page.waitForTimeout(1200);
  await page.screenshot({ path: `${output}/${mode}-settled.png` });
  await page.evaluate(() => {
    window.titleSamples = [];
    window.titleSampling = setInterval(() => {
      const el = [
        ...document.querySelectorAll(
          '[data-testid="conversation-card-title"] span',
        ),
      ].find((e) => e.textContent === "Review the startup handoff");
      if (el)
        window.titleSamples.push({
          at: performance.now(),
          opacity: Number(getComputedStyle(el).opacity),
          text: el.textContent,
        });
    }, 8);
  });
  await page.evaluate(async () => {
    const { queryClient } = await import("/src/query-client-config.ts");
    const { CONVERSATION_QUERY_KEYS: keys } =
      await import("/src/hooks/query/query-keys.ts");
    const key = keys.paginated(20, "default-local", null);
    const existing = queryClient.getQueryData(key);
    const row = {
      ...existing.pages[0].items[0],
      id: "title-proof-chat",
      title: "Review the startup handoff",
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      title_generation: {
        status: "generated",
        title: "Review the startup handoff",
        deadline_at: new Date().toISOString(),
      },
      selected_repository: null,
      selected_workspace: null,
      workspace: null,
      tags: {},
    };
    queryClient.setQueryData(key, {
      ...existing,
      pages: [
        { ...existing.pages[0], items: [row, ...existing.pages[0].items] },
        ...existing.pages.slice(1),
      ],
    });
  });
  await page.waitForTimeout(1200);
  const samples = await page.evaluate(() => {
    clearInterval(window.titleSampling);
    return window.titleSamples;
  });
  const minimumOpacity = Math.min(...samples.map((s) => s.opacity));
  console.log(
    JSON.stringify({ mode, minimumOpacity, samples: samples.length }),
  );
  await writeFile(
    `${output}/${mode}-samples.json`,
    JSON.stringify(samples, null, 2),
  );
  if (mode === "after" ? minimumOpacity !== 1 : minimumOpacity >= 0.9)
    throw new Error(`Unexpected minimum opacity ${minimumOpacity}`);
  await page.screenshot({ path: `${output}/${mode}.png` });
  await page.waitForTimeout(800);
} catch (error) {
  await page.screenshot({ path: `${output}/${mode}-error.png` });
  throw error;
} finally {
  await context.close();
  console.log(JSON.stringify({ video: await page.video().path() }));
  await browser.close();
}
