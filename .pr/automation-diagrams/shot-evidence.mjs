// Screenshot any Canvas path in a dark and a light theme; report the diagram box.
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";

const require = createRequire(
  "/Users/VascoSchiavo/git/OpenHands/.claude/worktrees/canvas-mermaid/package.json",
);
const { chromium } = require("playwright");
const [path, outPrefix, height = "1100"] = process.argv.slice(2);
const apiKey = readFileSync("/Users/VascoSchiavo/.openhands/agent-canvas/api-key.txt", "utf8").trim();
const origin = process.env.ORIGIN ?? "http://localhost:8000";
const clip = process.env.CLIP ? (([x, y, width, height]) => ({ x, y, width, height }))(process.env.CLIP.split(",").map(Number)) : undefined;

const browser = await chromium.launch();
for (const theme of ["openhands-deepsea", "light-plus"]) {
  const context = await browser.newContext({ viewport: { width: 1280, height: Number(height) } });
  await context.addInitScript(
    ({ apiKey, origin, theme }) => {
      const active = JSON.stringify({ backendId: "local", orgId: null });
      localStorage.setItem("openhands-onboarded", "1");
      localStorage.setItem("openhands-backends", JSON.stringify([{ id: "local", name: "Local", host: origin, apiKey, kind: "local" }]));
      localStorage.setItem("openhands-active-backend", active);
      sessionStorage.setItem("openhands-active-backend", active);
      localStorage.setItem("openhands-color-theme", theme);
      localStorage.setItem("openhands-sidebar-onboarding-checklist-dismissed", "true");
    },
    { apiKey, origin, theme },
  );
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(`${origin}${path}`);
  await page.getByRole("button", { name: "Confirm preferences" }).click({ timeout: 8000 }).catch(() => {});
  const diagram = page.getByTestId("mermaid-diagram");
  try {
    await diagram.waitFor({ timeout: Number(process.env.DIAGRAM_WAIT ?? 30000) });
    console.log(theme, "diagram", JSON.stringify(await diagram.boundingBox()));
    await diagram.screenshot({ path: `${outPrefix}-diagram-${theme}.png` });
  } catch {
    console.log(theme, "NO DIAGRAM; dialog text:", (await page.locator("body").innerText()).slice(0, 300).replace(/\n/g, " | "));
  }
  await page.screenshot({ path: `${outPrefix}-${theme}.png`, clip });
  console.log(theme, "page errors", JSON.stringify(errors));
  await context.close();
}
await browser.close();
