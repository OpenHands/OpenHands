/**
 * Captures the Maritime provider screens from Canvas running in mock mode
 * (`npm run dev:mock`, MSW stands in for the agent-server) with a fake
 * `window.maritimeBridge` injected before the app loads. Nothing here talks to
 * Maritime: these are UI captures, not live evidence.
 *
 *   npm run dev:mock            # serves http://localhost:3001 by default
 *   node .pr/maritime/capture-screenshots.mjs http://localhost:3001
 */
import { chromium } from "playwright";
import path from "node:path";
import { fileURLToPath } from "node:url";

const baseUrl = process.argv[2] ?? "http://localhost:3001";
const outDir = path.dirname(fileURLToPath(import.meta.url));

/** Runs in the page before any app script: a fake bridge with two agents. */
function installFakeBridge() {
  let signedIn = false;
  const agents = [
    {
      id: "6f1c0d3e-agent-a",
      name: "openhands-billing-fixes",
      status: "active",
      webUrl: "https://6f1c0d3e-agent-a.mtm.sh/",
      memMb: 4096,
      vcpus: 2,
      createdAt: null,
      lastActiveAt: null,
      hasAccessKey: true,
    },
    {
      id: "9a2b7c4d-agent-b",
      name: "openhands-docs-sweep",
      status: "sleeping",
      webUrl: "https://9a2b7c4d-agent-b.mtm.sh/",
      memMb: 2048,
      vcpus: 1,
      createdAt: null,
      lastActiveAt: null,
      hasAccessKey: true,
    },
    {
      id: "c3d4e5f6-agent-c",
      name: "openhands-nightly",
      status: "deploying",
      webUrl: "https://c3d4e5f6-agent-c.mtm.sh/",
      memMb: 4096,
      vcpus: 2,
      createdAt: null,
      lastActiveAt: null,
      hasAccessKey: true,
    },
  ];
  const auth = () => ({
    isSignedIn: signedIn,
    isSaved: true,
    storage: { kind: "keychain", label: "macOS Keychain", host: "maria-mbp" },
  });
  window.maritimeBridge = {
    getAuthState: async () => auth(),
    saveApiKey: async () => {
      await new Promise((r) => setTimeout(r, 300));
      signedIn = true;
      return auth();
    },
    signOut: async () => {
      signedIn = false;
      return auth();
    },
    getAccount: async () => ({
      plan: "starter",
      machines: 3,
      includedMachines: 20,
      awakeMachines: 1,
      maxAwakeMachines: 5,
      minMemMb: 512,
      maxMemMb: 8192,
    }),
    listAgents: async () => agents,
    createAgent: () => new Promise(() => {}),
    openAgent: () => new Promise(() => {}),
    resetAccessKey: () => new Promise(() => {}),
    sleepAgent: async () => agents[0],
    deleteAgent: async () => {},
  };
}

const browser = await chromium.launch({ channel: "chrome", headless: true });
const context = await browser.newContext({
  viewport: { width: 1280, height: 860 },
  deviceScaleFactor: 2,
  colorScheme: "dark",
});
const page = await context.newPage();
await page.addInitScript(installFakeBridge);
await page.addInitScript(() => {
  localStorage.setItem("openhands-onboarded", "true");
});
await page.goto(baseUrl, { waitUntil: "networkidle" });

async function shot(name) {
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(outDir, `${name}.png`) });
  console.log(`saved ${name}.png`);
}

// Open Manage Backends -> Add Backend, however the shell exposes it.
async function openAddBackend() {
  const chooser = page.getByTestId("add-backend-chooser");
  if (await chooser.isVisible().catch(() => false)) return;
  for (const testId of ["backend-selector"]) {
    const el = page.getByTestId(testId).first();
    if (await el.isVisible().catch(() => false)) {
      await el.click();
      break;
    }
  }
  for (const testId of ["add-backend-menu-item", "manage-backends-add"]) {
    const el = page.getByTestId(testId).first();
    if (await el.isVisible().catch(() => false)) {
      await el.click();
      await page.waitForTimeout(300);
    }
  }
  await chooser.waitFor({ timeout: 15_000 });
}

await openAddBackend();
await page.getByTestId("add-backend-option-providers").click();
await shot("01-providers");

await page.getByTestId("add-backend-provider-maritime").click();
await page.getByTestId("maritime-api-key").waitFor();
await shot("02-sign-in");

await page.getByTestId("maritime-api-key").fill(`mk_${"x".repeat(43)}`);
await page.getByTestId("maritime-api-key-submit").click();
await page.getByTestId("maritime-agents").waitFor();
await shot("03-agents");

await page.getByTestId("maritime-new-agent").click();
await page.getByTestId("maritime-new-agent-form").waitFor();
await shot("04-new-agent");

await page.getByTestId("maritime-new-agent-cancel").click();
await page.getByTestId("maritime-agent-delete-9a2b7c4d-agent-b").click();
await page.getByTestId("confirmation-modal").waitFor();
await shot("05-delete-confirmation");

await browser.close();
