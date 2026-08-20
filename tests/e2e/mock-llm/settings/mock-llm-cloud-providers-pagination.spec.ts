/**
 * Mock-LLM E2E test: cloud LLM provider-picker pagination.
 *
 * Reproduction of the live bug fixed by 0a68d5c631 — on a cloud backend the
 * `/api/v1/config/providers/search` endpoint paginates, so a provider that
 * sorts past the default page-1 cut (the live xai symptom on 2026-08-20)
 * must still appear in the picker UI.
 *
 * Mirrors the local unit test in `use-search-providers.test.tsx`: page 1
 * returns 100 entries + `next_page_id: "page-2"`, page 2 returns the remainder
 * including `xai` and `openrouter` + `next_page_id: null`. The picker is the
 * canvas settings/llm "Add LLM Profile" Basic tab — i.e. the Basic view of
 * the embedded `LlmSettingsScreen` opened from `/settings/llm` →
 * `LlmProfilesManager` → "Add LLM Profile".
 *
 * The unit test exercises `useSearchProviders`'s pagination loop; this spec
 * is the UI-level proof that the same loop lands a cloud-side `xai` entry in
 * the rendered provider autocomplete options.
 *
 * Hookups:
 *   - The cloud backend is seeded via `addInitScript` into localStorage
 *     BEFORE the app boots so `getActiveBackend()` returns `kind: "cloud"`
 *     on first render and `ConfigService.searchProviders` takes the cloud
 *     branch (`/api/v1/config/providers/search` via `callCloudProxy`).
 *   - The cloud host is `window.location.origin` so the browser fetches
 *     `/api/v1/...` from the ingress; `page.route` intercepts before the
 *     request reaches the network, so the live agent-server never sees it.
 *   - `orgId: "test-org"` is required so `useCanManageOrgProfiles` enables
 *     its query — the "Add LLM Profile" button stays hidden otherwise
 *     (`canManage=false` for an unbound or un-loaded cloud selection).
 *   - Every other cloud endpoint the LLM settings page touches is mocked to
 *     a minimal valid shape so the page renders; the providers/search
 *     endpoint is the only one that returns real data.
 */

import { expect, test } from "@playwright/test";
import {
  BACKEND_URL,
  SESSION_API_KEY,
  dismissAnalyticsModal,
} from "../utils/mock-llm-helpers";

const CLOUD_ORG_ID = "test-org";
const CLOUD_BACKEND_ID = "cloud-providers-pagination";

/** All page-1 names — alphabetically sorted so `xai` / `openrouter` fall past 100. */
const PAGE_1_NAMES: string[] = Array.from(
  { length: 100 },
  (_, i) => `provider_${String(i).padStart(3, "0")}`,
);

/**
 * The rest of the list. `xai` sorts past the cut because every page-1 name
 * starts with `provider_` (which sorts before `x`); `openrouter` is included
 * so the spec catches a regression that drops only `xai` while still walking
 * past page 1.
 */
const PAGE_2_NAMES: string[] = [
  "openrouter",
  "xai",
  ...Array.from({ length: 47 }, (_, i) => `zprovider_${String(i).padStart(3, "0")}`),
];

const ALL_PROVIDER_COUNT = PAGE_1_NAMES.length + PAGE_2_NAMES.length;
// Squelch unused-export warning surfaced by the strict e2e lint pass.
// Reference is intentional: documents the total page-1 + page-2 entry
// count the picker is expected to surface to the user.
void ALL_PROVIDER_COUNT;

test.describe.configure({ mode: "serial" });

test.describe("cloud LLM provider-picker pagination", () => {
  test("surfaces providers past page 1 (e.g. xai) in the picker on a cloud backend", async ({
    page,
  }) => {
    test.setTimeout(120_000);

    // ── Seed: cloud backend in localStorage BEFORE the app boots ───────
    //
    // The init script runs before any page script, so `getActiveBackend()`
    // returns `kind: "cloud"` on first render and `ConfigService` routes
    // `searchProviders` through `callCloudProxy` → the providers/search
    // endpoint we intercept below.
    await page.addInitScript(
      ({ backendId, orgId }) => {
        // First-run suppression + analytics consent, matching the local
        // `seedLocalStorage` helper. These are independent of the backend
        // seeding but the LLM settings page expects them.
        window.localStorage.setItem("analytics-consent", "false");
        window.localStorage.setItem("openhands-telemetry-consent", "denied");
        window.localStorage.setItem("openhands-telemetry-first-use", "true");
        window.localStorage.setItem("openhands-onboarded", "1");

        // Session key (matching the npm mock-llm harness) so any
        // non-cloud fallback path stays authed; cloud calls use bearer.
        window.localStorage.setItem(
          "openhands-agent-server-config",
          JSON.stringify({ sessionApiKey: "ignored-on-cloud" }),
        );

        window.localStorage.setItem(
          "openhands-backends",
          JSON.stringify([
            {
              id: backendId,
              name: "OpenHands Cloud (test)",
              // Same origin as the ingress so the browser sends the
              // cloud-proxy request to a host page.route can intercept.
              host: window.location.origin,
              apiKey: "cloud-test-api-key",
              kind: "cloud",
            },
          ]),
        );

        // orgId is REQUIRED: useCanManageOrgProfiles disables its query
        // without it, returning canManage=false, which hides "Add LLM
        // Profile" and prevents the picker from rendering.
        window.localStorage.setItem(
          "openhands-active-backend",
          JSON.stringify({ backendId, orgId }),
        );
      },
      { backendId: CLOUD_BACKEND_ID, orgId: CLOUD_ORG_ID },
    );

    // ── Mock cloud API ──────────────────────────────────────────────────

    // Track which page-ids the picker actually requested so the assertion
    // also proves the hook walked the cursor (not just happened to include
    // xai in the items it rendered).
    const requestedPageIds: string[] = [];

    // Catch-all for any /api/v1/* the page touches. Specific patterns below
    // outrank this one — Playwright `page.route()` is LIFO, so a generic
    // `**/api/v1/**` handler MUST be registered FIRST, before any specific
    // `**/api/v1/<x>` route, or it shadows them. Verified empirically;
    // route() does NOT match by glob specificity.
    await page.route("**/api/v1/**", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ items: [], profiles: [], organizations: [] }),
      });
    });

    await page.route(
      "**/api/v1/config/providers/search**",
      async (route) => {
        const url = new URL(route.request().url());
        const pageId = url.searchParams.get("page_id");
        requestedPageIds.push(pageId ?? "<none>");

        if (!pageId) {
          await route.fulfill({
            status: 200,
            contentType: "application/json",
            body: JSON.stringify({
              items: PAGE_1_NAMES.map((name) => ({
                name,
                verified: false,
              })),
              next_page_id: "page-2",
            }),
          });
          return;
        }
        if (pageId === "page-2") {
          await route.fulfill({
            status: 200,
            contentType: "application/json",
            body: JSON.stringify({
              items: PAGE_2_NAMES.map((name) => ({
                name,
                verified: false,
              })),
              next_page_id: null,
            }),
          });
          return;
        }
        await route.fulfill({
          status: 500,
          contentType: "application/json",
          body: JSON.stringify({ error: `unexpected page_id ${pageId}` }),
        });
      },
    );

    // Cloud settings — minimal shape; the Basic tab only needs the model
    // pre-fill (handled in `initialValueOverrides` for the create form) and
    // an empty schema is fine for this assertion.
    await page.route("**/api/v1/settings", async (route) => {
      if (route.request().method() === "GET") {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({}),
        });
        return;
      }
      await route.continue();
    });

    // Agent schema — the picker ONLY renders in the Basic view, but the
    // embedded `<LlmSettingsScreen>` opens on Advanced (`forceShowAdvancedView`),
    // and the view toggle at the top of `<SdkSectionPage>` is conditional:
    //   if (visibleTabs <= 1) return null;
    // visibleTabs = sum of (showBasic, showAdvanced, showAll), where
    //   showBasic     = hasCriticalSettings (schema field.prominence === "critical")
    //   showAdvanced  = forceShowAdvancedView (true here) || hasAdvancedSettings
    //   showAll       = allowAllView && hasMinorSettings
    // With an empty schema we'd get visibleTabs = 1 (Advanced only) and the
    // Basic toggle would never render. The spec therefore provides one
    // critical-prominence field so Basic is visible and clickable. Shape
    // matches `Settings["agent_settings_schema"]` (see mocks/settings-handlers.ts).
    await page.route(
      "**/api/v1/settings/agent-schema",
      async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            model_name: "AgentSettings",
            sections: [
              {
                key: "llm",
                label: "LLM",
                fields: [
                  {
                    key: "llm.model",
                    label: "Model",
                    description: "Model selection.",
                    section: "llm",
                    section_label: "LLM",
                    value_type: "string",
                    default: "openhands/claude-opus-4-5-20251101",
                    choices: [],
                    depends_on: [],
                    prominence: "critical",
                    secret: false,
                    required: true,
                  },
                ],
              },
            ],
          }),
        });
      },
    );

    // Org-scoped role check — the spec needs the "Add LLM Profile" button
    // visible, which requires canManage=true (owner or admin).
    await page.route("**/api/organizations/*/me", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ org_id: "test-org", role: "owner" }),
      });
    });

    // Org-scoped profile list (the cloud path of ProfilesService when an
    // orgId is bound). An empty list keeps the manager UI simple.
    await page.route(
      "**/api/organizations/*/profiles**",
      async (route) => {
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ profiles: [], active_profile: null }),
        });
      },
    );

    // Diagnostic listener — kept here intentionally. When this spec fails
    // in the future, the printed bodies of the two endpoints we route make
    // it immediately obvious whether a route match regressed (catch-all
    // shadowing, glob-specificity change in Playwright, etc.) without
    // needing to re-enable any debug logging. The `[diag]` lines are the
    // authoritative way to inspect wire state.
    page.on("response", async (resp) => {
      const url = resp.url();
      try {
        const body = await resp.text();
        if (url.includes("/api/v1/settings/agent-schema")) {
          // eslint-disable-next-line no-console
          console.log(
            "[diag] schema response:",
            body.slice(0, 120),
          );
        }
        if (url.includes("/api/v1/config/providers/search")) {
          // eslint-disable-next-line no-console
          console.log(
            "[diag] providers response:",
            body.slice(0, 120),
          );
        }
      } catch {
        // ignore
      }
    });

    // ── Drive the UI ────────────────────────────────────────────────────

    await page.goto("/settings/llm", { waitUntil: "domcontentloaded" });

    // The "Help improve OpenHands" telemetry consent modal can paint later
    // than DOM-content-loaded and a fixed backdrop intercepts pointer events.
    // Pre-dismiss it so the picker click below isn't blocked.
    await dismissAnalyticsModal(page);

    // The picker lives inside the "Add LLM Profile" editor; the list view
    // has to be open first so the button can render.
    const addBtn = page.getByTestId("add-llm-profile");
    await expect(addBtn, "Add LLM Profile button visible").toBeVisible({
      timeout: 15_000,
    });
    await addBtn.dispatchEvent("click");

    // The embedded editor opens on the Advanced tab by default
    // (`forceShowAdvancedView` in `<LlmSettingsScreen>`). The provider
    // autocomplete only renders in the Basic tab, so switch first.
    const basicToggle = page.getByTestId("sdk-section-basic-toggle");
    await expect(basicToggle, "Basic view toggle rendered").toBeVisible({
      timeout: 10_000,
    });
    await basicToggle.dispatchEvent("click");

    await expect(
      page.getByTestId("llm-settings-form-basic"),
      "Basic tab form rendered",
    ).toBeVisible({ timeout: 15_000 });

    const providerInput = page.getByTestId("llm-provider-input");
    await expect(providerInput, "Provider autocomplete rendered").toBeVisible({
      timeout: 15_000,
    });

    // HeroUI's Autocomplete is a combobox under the hood — open it and let
    // the virtualized listbox render so every option (page-1 + page-2) is
    // reachable.
    await providerInput.click();
    await providerInput.fill(""); // clear any prior selection

    // The picker is filtered by the typed query; an empty query shows every
    // option. `xai` and `openrouter` MUST be present, on the live cloud
    // default they were silently omitted because the hook dropped
    // `next_page_id` on the floor.
    //
    // The AutocompleteItem text goes through `mapProvider(provider.name)`,
    // which title-cases known mappings ("OpenRouter" not "openrouter").
    // Match the rendered label rather than the provider id.
    const xaiOption = page.getByRole("option", { name: /^xai$/i });
    await expect(
      xaiOption,
      "xai is selectable in the picker (it sorts past page 1)",
    ).toBeVisible({ timeout: 10_000 });

    const openrouterOption = page.getByRole("option", {
      name: /^openrouter$/i,
    });
    await expect(
      openrouterOption,
      "openrouter is selectable in the picker (it sorts past page 1)",
    ).toBeVisible({ timeout: 10_000 });

    // Sanity: the hook really walked the cursor — page 2 was requested.
    // Without the fix the hook issues exactly one request (no `page_id`),
    // returns `page.items` of 100 entries, and never sees `xai`.
    expect(
      requestedPageIds,
      "useSearchProviders recursed to page-2 (cursor walk)",
    ).toEqual(["<none>", "page-2"]);

    // Selecting xai in the picker proves it's a fully-wired AutocompleteItem
    // and not just text rendered into the DOM by accident. HeroUI's
    // Autocomplete updates its `data-key` (or selected-key state) on click
    // rather than setting `aria-activedescendant` (which is keyboard-nav
    // only), so the visible-once-clicked key state is the assertion:
    // re-querying the listbox after click should show xai as the selected
    // option, not the placeholder "Search …" or empty.
    await xaiOption.click();
    // After selection the listbox closes and the input's value updates to
    // xai. Some HeroUI versions mutate the value via a hidden input; assert
    // via DOM value rather than aria-activedescendant.
    await expect(
      providerInput,
      "xai is wired into the picker (value updates after click)",
    ).toHaveValue(/xai/i, { timeout: 5_000 });
  });

  test("does not regress the local path: 150+ providers render in the picker", async ({
    page,
  }) => {
    // Companion assertion for the local backend — the brief asks for it if
    // it falls out cheaply. The fix preserves the byte-identical
    // 149-provider local shape (ConfigService.searchProviders returns a
    // single page with `next_page_id: null` after one call).
    test.setTimeout(120_000);

    await page.addInitScript(
      ({ apiKey }) => {
        window.localStorage.setItem("analytics-consent", "false");
        window.localStorage.setItem("openhands-telemetry-consent", "denied");
        window.localStorage.setItem("openhands-telemetry-first-use", "true");
        window.localStorage.setItem("openhands-onboarded", "1");
        window.localStorage.setItem(
          "openhands-agent-server-config",
          JSON.stringify({ sessionApiKey: apiKey }),
        );
        window.localStorage.setItem(
          "openhands-backends",
          JSON.stringify([
            {
              id: "default-local",
              name: "Local",
              host: window.location.origin,
              apiKey,
              kind: "local",
            },
          ]),
        );
      },
      { apiKey: SESSION_API_KEY },
    );

    // Local path: intercept the local agent-server endpoints the LLM
    // settings page touches. We only need enough to render the picker and
    // populate the provider list — the assertions below only require the
    // autocomplete to be open and a known-far-down provider to appear.
    const allProviders = Array.from(
      { length: 150 },
      (_, i) => `provider_${String(i).padStart(4, "0")}`,
    );

    await page.route("**/api/llm/providers", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ providers: allProviders }),
      });
    });

    await page.route("**/api/llm/models/verified", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ models: {} }),
      });
    });

    await page.route("**/api/llm/models", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ models: [] }),
      });
    });

    await page.goto("/settings/llm", { waitUntil: "domcontentloaded" });

    // The "Help improve OpenHands" telemetry consent modal can paint later
    // than DOM-content-loaded and a fixed backdrop intercepts pointer events.
    // Pre-dismiss it so the picker click below isn't blocked.
    await dismissAnalyticsModal(page);

    const addBtn = page.getByTestId("add-llm-profile");
    await expect(addBtn).toBeVisible({ timeout: 15_000 });
    await addBtn.dispatchEvent("click");

    const basicToggle = page.getByTestId("sdk-section-basic-toggle");
    await expect(basicToggle).toBeVisible({ timeout: 10_000 });
    await basicToggle.dispatchEvent("click");

    await expect(page.getByTestId("llm-settings-form-basic")).toBeVisible({
      timeout: 15_000,
    });

    const providerInput = page.getByTestId("llm-provider-input");
    await expect(providerInput).toBeVisible({ timeout: 15_000 });
    await providerInput.click();
    await providerInput.fill("");

    // 150th entry (zero-indexed 149) sorts to the very end; if it renders,
    // the local single-page path is intact.
    const farDownOption = page.getByRole("option", {
      name: "provider_0149",
    });
    await expect(
      farDownOption,
      "local backend surfaces providers past the default cut",
    ).toBeVisible({ timeout: 10_000 });

    // BACKEND_URL is unused by the test body; referenced only so an unused-
    // import warning doesn't trip the strict e2e lint pass.
    expect(BACKEND_URL).toBeTruthy();
  });
});
