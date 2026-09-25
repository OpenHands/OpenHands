#!/usr/bin/env node
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { lstat, mkdir, readFile, writeFile } from "node:fs/promises";
import { isAbsolute, join } from "node:path";
import { chromium, expect } from "@playwright/test";

const run = process.env.QA_RUN;
assert.ok(run && isAbsolute(run), "Set QA_RUN from the runtime recipe");
for (const path of [run, join(run, "private"), join(run, "evidence")]) {
  const stat = await lstat(path);
  assert.ok(
    stat.isDirectory() && !stat.isSymbolicLink(),
    `Unsafe run path: ${path}`,
  );
}
const base = new URL(process.env.QA_BASE_URL);
assert.ok(["127.0.0.1", "localhost", "[::1]"].includes(base.hostname));
assert.equal(base.protocol, "http:");
assert.equal(base.pathname, "/");
assert.ok(!base.username && !base.password && !base.search && !base.hash);
const key = (await readFile(join(run, "private/session-key"), "utf8")).trim();
assert.ok(key.length >= 32, "Missing run-owned session key");
const revision = execFileSync("git", ["rev-parse", "HEAD"], {
  encoding: "utf8",
}).trim();
const marker = await fetch(new URL("verify-revision.txt", base), {
  redirect: "error",
  signal: AbortSignal.timeout(10000),
});
assert.equal(marker.status, 200);
assert.equal(
  (await marker.text()).trim(),
  revision,
  "Wrong or stale frontend build",
);
const settings = await fetch(new URL("api/settings", base), {
  headers: { "X-Session-API-Key": key },
  redirect: "error",
  signal: AbortSignal.timeout(10000),
});
assert.equal(settings.status, 200, "Run doctor before driving");
await settings.json();

const attempt = `smoke-${randomUUID()}`;
const evidence = join(run, "evidence", attempt);
const diagnostics = join(run, "private", attempt);
await mkdir(evidence, { mode: 0o700 });
await mkdir(diagnostics, { mode: 0o700 });
let outcome = "failed";
const browser = await chromium.launch({
  executablePath: process.env.QA_BROWSER_EXECUTABLE,
});
const context = await browser.newContext({
  viewport: { width: 1440, height: 1000 },
  locale: "en-US",
});
const page = await context.newPage();
page.setDefaultTimeout(20000);
const errors = [];
const checks = [];
const fixture = `QA_VERIFY_${randomUUID().replaceAll("-", "")}`;
let fixtureMayExist = false;
page.on("pageerror", (error) => errors.push(error.message));

async function capture(name) {
  await page.evaluate(() => document.fonts.ready);
  await page.screenshot({
    path: join(evidence, `${name}.png`),
    animations: "disabled",
  });
}

try {
  await page.goto(base.href);
  await page.getByTestId("onboarding-skip").click();
  await expect(page.getByTestId("api-key-entry-screen")).toBeVisible();
  await page.getByTestId("api-key-entry-name").fill("Verification backend");
  await page.getByTestId("api-key-entry-api-key").fill(key);
  await page.getByTestId("api-key-entry-submit").click();
  await expect(page.getByTestId("api-key-entry-screen")).toBeHidden();
  await page.goto(new URL("settings/secrets", base).href, {
    waitUntil: "networkidle",
  });
  const consent = page.getByTestId("telemetry-consent-form");
  if (await consent.isVisible()) {
    await consent.getByRole("checkbox").uncheck();
    await page.getByTestId("confirm-telemetry-preferences").click();
    await expect(consent).toBeHidden();
  }
  checks.push({
    id: "F01.auth",
    action: "Authenticate through public-mode UI",
    result: "pass",
  });

  await page.getByTestId("add-secret-button").click();
  const form = page.getByTestId("add-secret-form");
  await form.getByTestId("name-input").fill(fixture);
  await form.getByTestId("value-input").fill("visual-qa-not-a-real-credential");
  await form
    .getByTestId("description-input")
    .fill("Disposable verification fixture");
  await capture("F05-secret-before-save");
  fixtureMayExist = true;
  await form.getByTestId("submit-button").click();
  const row = page.getByTestId("secret-item").filter({ hasText: fixture });
  await expect(row).toHaveCount(1);
  await page.reload({ waitUntil: "networkidle" });
  await expect(row).toHaveCount(1);
  await expect(row).toContainText("Disposable verification fixture");
  await capture("F05-secret-after-reload");
  checks.push({
    id: "F05.secret-create",
    action: "Add dummy secret and reload",
    result: "pass",
    evidence: ["F05-secret-before-save.png", "F05-secret-after-reload.png"],
  });
  await row.getByTestId("delete-secret-button").click();
  await page
    .getByTestId("confirmation-modal")
    .getByTestId("confirm-button")
    .click();
  await expect(row).toHaveCount(0);
  await page.reload({ waitUntil: "networkidle" });
  await expect(page.getByTestId("add-secret-button")).toBeVisible();
  await expect(row).toHaveCount(0);
  fixtureMayExist = false;
  await capture("F05-secret-after-delete");
  checks.push({
    id: "F05.secret-delete",
    action: "Confirm deletion and reload",
    result: "pass",
    evidence: ["F05-secret-after-delete.png"],
  });

  await page.goto(new URL("mcp", base).href);
  await page.getByTestId("mcp-marketplace-card-github").click();
  const password = page.getByTestId("mcp-install-field-api_key");
  await expect(password).toHaveAttribute("type", "password");
  await password.fill("visual-qa-not-a-real-credential");
  await password.blur();
  for (const viewport of [
    { width: 1440, height: 1000 },
    { width: 390, height: 844 },
  ]) {
    await page.setViewportSize(viewport);
    await password.scrollIntoViewIfNeeded();
    await expect(password).toHaveAttribute("type", "password");
    await expect(password).toHaveValue("visual-qa-not-a-real-credential");
    const box = await password.boundingBox();
    assert.ok(
      box &&
        box.x >= 0 &&
        box.y >= 0 &&
        box.x + box.width <= viewport.width &&
        box.y + box.height <= viewport.height,
    );
    const name = `F06-github-credential-${viewport.width}`;
    await capture(name);
    checks.push({
      id: "F06.credential-layout",
      action: "Open GitHub install form with dummy password; do not install",
      result: "pass",
      viewport,
      evidence: [`${name}.png`],
    });
  }
  assert.equal(
    errors.length,
    0,
    `Browser exceptions occurred; inspect ${diagnostics}/browser-errors.json`,
  );
  outcome = "passed";
  console.log(
    `PASS: ${checks.length} scoped checks; inspect screenshots in ${evidence}. Not a full feature audit.`,
  );
} catch (error) {
  await page
    .screenshot({ path: join(diagnostics, "failure.png") })
    .catch(() => {});
  throw error;
} finally {
  await writeFile(
    join(diagnostics, "browser-errors.json"),
    JSON.stringify({ errors, fixture, fixtureMayExist }, null, 2),
  );
  await writeFile(
    join(evidence, "smoke-results.json"),
    JSON.stringify(
      {
        outcome,
        revision,
        browser: browser.version(),
        checks,
        browserExceptionCount: errors.length,
        fixtureCleanupComplete: !fixtureMayExist,
      },
      null,
      2,
    ),
  );
  await browser.close();
}
