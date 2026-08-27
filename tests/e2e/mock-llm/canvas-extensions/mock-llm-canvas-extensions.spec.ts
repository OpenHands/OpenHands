/**
 * Mock-LLM E2E test: Canvas Extensions lifecycle through the production build.
 *
 * Drives the real agent-canvas stack (pre-built static frontend, ingress,
 * backend registry, session auth) through the full extension lifecycle:
 *
 *   1. Install from a source path → lands disabled, no sidebar item
 *   2. Enable behind the trust confirmation → the sidebar item appears and the
 *      contributed page renders at /extensions/{name}/{path}, including a
 *      nested remainder path
 *   3. Disable → the sidebar item and the page are torn down
 *   4. Uninstall → the inventory is empty
 *
 * This is the only test that imports an extension bundle in a real browser
 * (Blob-URL ESM import in `canvas-extension-module-loader.ts`); the unit
 * tests inject a module loader stub instead.
 *
 * The pinned agent-server predates the `/api/canvas-extensions` endpoints
 * (software-agent-sdk#4395), so `serveCanvasExtensionApi` answers them at the
 * browser level from the checked-in demo fixture. Once the pin includes the
 * endpoints, delete that helper and install the fixture by absolute path —
 * the UI steps below stay unchanged.
 */

import { readFileSync } from "fs";
import { join, resolve } from "path";
import { test, expect, type Page } from "@playwright/test";
import {
  seedLocalStorage,
  routeSessionApiKey,
  dismissAnalyticsModal,
  waitForPath,
  waitForTestId,
} from "../utils/mock-llm-helpers";

const DEMO_SOURCE = "src/fixtures/canvas-extensions/demo-page";
const FIXTURE_DIR = resolve(DEMO_SOURCE);
const manifest = JSON.parse(
  readFileSync(join(FIXTURE_DIR, "canvas-extension.json"), "utf8"),
) as { name: string; version: string; description: string };
const bundle = readFileSync(join(FIXTURE_DIR, "extension.js"), "utf8");

const CARD = `canvas-extension-card-${manifest.name}`;
const SIDEBAR_ITEM = `sidebar-canvas-extension-${manifest.name}-hello`;
const PAGE_PATH = `/extensions/${manifest.name}/hello`;
const EMPTY_INVENTORY_TEXT =
  "No Canvas Extensions are installed on this backend.";

/** In-memory installation state shared by the serial steps below. */
let installed: Record<string, unknown> | null = null;

/**
 * Serve the Canvas Extension API contract from the demo fixture. Registered
 * after `routeSessionApiKey` so it takes precedence for these routes only;
 * every other request still reaches the real stack.
 */
async function serveCanvasExtensionApi(page: Page) {
  await page.route("**/api/canvas-extensions/**", async (route) => {
    const request = route.request();
    const method = request.method();
    const path = new URL(request.url()).pathname.replace(
      /^.*\/api\/canvas-extensions/,
      "",
    );
    const json = (status: number, body: unknown) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    if (method === "GET" && path === "/installed") {
      return json(200, { canvas_extensions: installed ? [installed] : [] });
    }
    if (method === "POST" && path === "/install") {
      const body = request.postDataJSON() as {
        source: string;
        ref?: string | null;
        repo_path?: string | null;
      };
      installed = {
        name: manifest.name,
        version: manifest.version,
        description: manifest.description,
        enabled: false,
        source: body.source,
        requested_ref: body.ref ?? null,
        resolved_ref: body.ref || "e2e-working-tree",
        repo_path: body.repo_path ?? null,
        installed_at: new Date().toISOString(),
        install_path: `/e2e/canvas-extensions/${manifest.name}`,
        manifest,
      };
      return json(200, installed);
    }
    if (!installed) {
      return json(404, {
        detail: `Canvas extension '${manifest.name}' is not installed.`,
      });
    }
    if (method === "GET" && path === `/installed/${manifest.name}/bundle`) {
      return route.fulfill({
        status: 200,
        contentType: "application/javascript; charset=utf-8",
        body: bundle,
      });
    }
    if (path === `/installed/${manifest.name}`) {
      if (method === "GET") return json(200, installed);
      if (method === "PATCH") {
        const { enabled } = request.postDataJSON() as { enabled: boolean };
        installed = { ...installed, enabled };
        return json(200, { name: manifest.name, enabled });
      }
      if (method === "DELETE") {
        installed = null;
        return json(200, { message: "Canvas extension uninstalled." });
      }
    }
    return json(404, { detail: `Unhandled ${method} ${path}` });
  });
}

async function openExtensionsPage(page: Page) {
  await page.goto("/extensions", { waitUntil: "domcontentloaded" });
  await dismissAnalyticsModal(page);
  await waitForTestId(page, "canvas-extensions-screen");
}

function cardToggle(page: Page) {
  return page.getByTestId(CARD).getByRole("switch");
}

function extensionPage(page: Page) {
  return page.getByRole("main", { name: "Hello from an extension" });
}

test.describe.configure({ mode: "serial" });

test.describe("Canvas Extensions lifecycle", () => {
  test.beforeEach(async ({ page }) => {
    await seedLocalStorage(page);
    await routeSessionApiKey(page);
    await serveCanvasExtensionApi(page);
  });

  test.afterAll(() => {
    installed = null;
  });

  test("step 1: install from a source path lands disabled with no sidebar item", async ({
    page,
  }) => {
    await openExtensionsPage(page);
    await expect(page.getByText(EMPTY_INVENTORY_TEXT)).toBeVisible();

    await page.getByTestId("canvas-extensions-add-button").click();
    const modal = page.getByTestId("add-canvas-extension-modal");
    await expect(modal).toBeVisible();
    await page
      .getByTestId("add-canvas-extension-source-input")
      .fill(DEMO_SOURCE);
    await page.getByTestId("add-canvas-extension-submit").click();
    await expect(modal).toBeHidden();

    const card = page.getByTestId(CARD);
    await expect(card).toBeVisible();
    await expect(card).toContainText("Demo page");
    await expect(cardToggle(page)).toHaveAttribute("aria-checked", "false");
    await expect(page.getByTestId(SIDEBAR_ITEM)).toHaveCount(0);
  });

  test("step 2: enabling behind the trust confirmation registers the sidebar item and renders the page", async ({
    page,
  }) => {
    await openExtensionsPage(page);
    await cardToggle(page).click();

    const confirmation = page.getByTestId("confirmation-modal");
    await expect(confirmation).toBeVisible();
    await expect(confirmation).toContainText("Enable Demo page?");
    await confirmation.getByTestId("confirm-button").click();

    await expect(cardToggle(page)).toHaveAttribute("aria-checked", "true");
    const sidebarItem = page.getByTestId(SIDEBAR_ITEM);
    await expect(sidebarItem).toBeVisible();

    await sidebarItem.click();
    await waitForPath(page, new RegExp(`${PAGE_PATH}$`));
    await expect(extensionPage(page)).toContainText(
      "Hello from a Canvas Extension",
    );
    await expect(extensionPage(page)).toContainText(
      "Host API 1 on backend default-local",
    );

    await page.goto(`${PAGE_PATH}/nested`, { waitUntil: "domcontentloaded" });
    await expect(extensionPage(page)).toContainText(
      "Nested extension path: nested",
    );
  });

  test("step 3: disabling tears down the sidebar item and the page", async ({
    page,
  }) => {
    await openExtensionsPage(page);
    await expect(page.getByTestId(SIDEBAR_ITEM)).toBeVisible();

    // Disabling needs no trust confirmation.
    await cardToggle(page).click();
    await expect(cardToggle(page)).toHaveAttribute("aria-checked", "false");
    await expect(page.getByTestId(SIDEBAR_ITEM)).toHaveCount(0);

    await page.goto(PAGE_PATH, { waitUntil: "domcontentloaded" });
    await expect(
      page.getByText(
        "This extension is disabled, missing, or does not provide this page.",
      ),
    ).toBeVisible();
  });

  test("step 4: uninstalling empties the inventory", async ({ page }) => {
    await openExtensionsPage(page);
    await page
      .getByTestId(`canvas-extension-uninstall-${manifest.name}`)
      .click();
    const confirmation = page.getByTestId("confirmation-modal");
    await expect(confirmation).toBeVisible();
    await confirmation.getByTestId("confirm-button").click();

    await expect(page.getByTestId(CARD)).toHaveCount(0);
    await expect(page.getByText(EMPTY_INVENTORY_TEXT)).toBeVisible();
  });
});
