#!/usr/bin/env node
// Evidence driver for PR #18913 (#18883).
//
// Drives the REAL production Canvas build against the mock cloud app-server
// (.pr/mock-cloud-server.mjs) with a cloud-kind backend registered directly
// in localStorage — the same registration shape the device flow produces.
//
// Run (after `npm run build:app`, sirv on FRONTEND_URL, mock cloud up):
//   node .pr/evidence-driver.mjs --label baseline --out .pr/evidence/baseline
//
// Emits: screenshots + evidence.json (extracted state per step). The mock
// server's JSONL log records every settings write and the launch-time kind.

import { chromium } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, arg, i, arr) => {
    if (arg.startsWith("--")) acc.push([arg.slice(2), arr[i + 1]]);
    return acc;
  }, []),
);
const LABEL = args.label ?? "run";
const OUT = args.out ?? join(".pr", "evidence", LABEL);
const FRONTEND = args.frontend ?? "http://127.0.0.1:18300";
const CLOUD = args.cloud ?? "http://127.0.0.1:18901";
const TOKEN = "mock-cloud-key";

mkdirSync(OUT, { recursive: true });
const evidence = { label: LABEL, frontend: FRONTEND, cloud: CLOUD, steps: {} };
const shot = (page, name) =>
  page.screenshot({ path: join(OUT, `${name}.png`), fullPage: false });
const step = (name, data) => {
  evidence.steps[name] = data;
  console.log(`[${LABEL}] ${name}: ${JSON.stringify(data)}`);
};

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });

// Register the mock cloud backend + select it before any app script runs.
await context.addInitScript(
  ({ backends, selection }) => {
    window.localStorage.setItem("openhands-onboarded", "true");
    window.localStorage.setItem("openhands-backends", JSON.stringify(backends));
    window.localStorage.setItem(
      "openhands-active-backend",
      JSON.stringify(selection),
    );
  },
  {
    backends: [
      {
        id: "cloud-mock",
        name: "Mock Cloud",
        host: CLOUD,
        apiKey: TOKEN,
        kind: "cloud",
        connectionRevision: 1,
      },
    ],
    selection: { backendId: "cloud-mock", orgId: "org-personal-1" },
  },
);

const page = await context.newPage();
const readSettingsKind = async () =>
  page.evaluate(async (cloud) => {
    const res = await fetch(`${cloud}/api/v1/settings`, {
      headers: { Authorization: "Bearer mock-cloud-key" },
    });
    const body = await res.json();
    return {
      agent_kind: body?.agent_settings?.agent_kind ?? null,
      acp_server: body?.agent_settings?.acp_server ?? null,
    };
  }, CLOUD);

// Show a fetched JSON blob on the page so the screenshot carries the state.
const showJson = async (title, value) => {
  await page.evaluate(
    ({ title, text }) => {
      let host = document.getElementById("evidence-overlay");
      if (!host) {
        host = document.createElement("div");
        host.id = "evidence-overlay";
        host.style.cssText =
          "position:fixed;inset:auto 24px 24px auto;z-index:99999;background:#111;color:#0f0;font:14px/1.5 monospace;padding:16px;border-radius:8px;max-width:520px;white-space:pre-wrap;";
        document.body.appendChild(host);
      }
      host.innerText = `${title}\n${text}`;
    },
    { title, text: JSON.stringify(value, null, 2) },
  );
};

// 1. Home with the ACP profile ("codex") active — identical on both builds.
await page.goto(`${FRONTEND}/`, { waitUntil: "domcontentloaded" });
await page.getByTestId("chat-input").waitFor({ timeout: 30000 });
await page.waitForTimeout(1500);
await shot(page, "home-1-acp-profile-active");
step("home-initial", {
  profilePill: await homePill(page),
});

// 2. Settings → Agents: "codex" is active, "default" available.
await page.goto(`${FRONTEND}/settings/agents`, { waitUntil: "domcontentloaded" });
await page.getByTestId("agent-profile-row").first().waitFor({ timeout: 30000 });
await page.waitForTimeout(1200);
await shot(page, "agents-1-before");

// 3. Activate the OpenHands profile "default" — the sync trigger under test.
const defaultRow = page.locator("[data-testid=agent-profile-row]", { hasText: "default" }).first();
await defaultRow.hover();
await defaultRow.getByTestId("agent-profile-menu-trigger").click({ timeout: 15000 });
await page.getByTestId("agent-profile-actions-menu").waitFor({ timeout: 10000 });
await shot(page, "agents-2-menu");
await page.getByTestId("agent-profile-set-active").click();
await page.waitForTimeout(2500); // activation + (PR build) settings sync + invalidations
await shot(page, "agents-3-after-activate");
const kindAfterActivate = await readSettingsKind();
await showJson("GET /api/v1/settings after activating the OpenHands profile", kindAfterActivate);
await shot(page, "settings-after-activate");
step("after-activate", kindAfterActivate);

// 4. Delete the active OpenHands profile → no active profile remains.
await defaultRow.hover();
await defaultRow.getByTestId("agent-profile-menu-trigger").click({ timeout: 15000 });
await page.getByTestId("agent-profile-actions-menu").waitFor({ timeout: 10000 });
await page.getByTestId("agent-profile-delete").click();
await page.getByTestId("delete-agent-profile-confirm").click();
await page.waitForTimeout(2500);
await shot(page, "agents-4-after-delete");
const kindAfterDelete = await readSettingsKind();
step("after-delete", kindAfterDelete);

// 5. Home with no active profile — the stale-kind fallback drives the UI.
await page.goto(`${FRONTEND}/`, { waitUntil: "domcontentloaded" });
await page.getByTestId("chat-input").waitFor({ timeout: 30000 });
await page.waitForTimeout(2000);
await shot(page, "home-2-no-active-profile");
step("home-after-delete", await homePill(page));

// 6. Launch a conversation — the server composes from the recorded kind.
await page.getByTestId("chat-input").fill("Reply with exactly: QA_18883. Do not run any tools.");
await page.getByTestId("submit-button").click().catch(() => {});
await page.keyboard.press("Enter");
await page.waitForURL(/\/conversations\//, { timeout: 30000 }).catch(() => {});
await page.waitForTimeout(3000);
await shot(page, "home-3-after-launch");
step("launched-url", { url: page.url() });

writeFileSync(join(OUT, "evidence.json"), JSON.stringify(evidence, null, 2));
console.log(`[${LABEL}] evidence written to ${OUT}`);
await browser.close();

async function homePill(page) {
  return page.evaluate(() => {
    const pill = (name) =>
      document.querySelector(`[data-testid=${name}]`)?.textContent?.trim() ?? null;
    return {
      agentProfilePill: pill("chat-input-agent-profile"),
      llmProfilePill: pill("chat-input-llm-profile"),
      acpModelPicker: Boolean(
        document.querySelector("[data-testid*=acp-model], [data-testid=chat-input-acp-model]"),
      ),
    };
  });
}
