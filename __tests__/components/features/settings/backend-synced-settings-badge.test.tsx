import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { readFileSync } from "node:fs";
import { COLOR_THEMES } from "#/themes/color-themes";
import { AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES } from "#/styles/agent-server-ui-style-scope";

import { BackendSyncedSettingsBadge } from "#/components/features/settings/backend-synced-settings-badge";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import * as orgService from "#/api/cloud/organization-service.api";
import type { Backend } from "#/api/backend-registry/types";

// Override the global react-i18next mock for this file. The badge under
// test interpolates `{{name}}` and `{{host}}`, and one regression we
// care about is that the host's `/` characters render literally rather
// than HTML-escaped (`&#x2F;`). The mock below performs `{{var}}`
// substitution and applies i18next's default HTML escape unless
// `interpolation.escapeValue === false` — exactly what the production
// runtime does — so a missing opt-out would surface as escaped output.
vi.mock("react-i18next", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-i18next")>();
  const TEMPLATES: Record<string, string> = {
    SETTINGS$BACKEND_SYNCED_BADGE:
      "These settings are synced from {{name}} backend ({{host}})",
  };
  const htmlEscape = (s: string) =>
    s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;")
      .replace(/\//g, "&#x2F;");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string, opts?: Record<string, unknown>) => {
        const tpl = TEMPLATES[key] ?? key;
        if (!opts) return tpl;
        const { interpolation } = opts as {
          interpolation?: { escapeValue?: boolean };
        };
        const escape = interpolation?.escapeValue !== false;
        return tpl.replace(/\{\{(\w+)\}\}/g, (_match, name) => {
          if (!(name in opts)) return "";
          const v = String((opts as Record<string, unknown>)[name]);
          return escape ? htmlEscape(v) : v;
        });
      },
      i18n: { language: "en", exists: () => false },
    }),
  };
});

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer",
  kind: "cloud",
};

function renderBadge() {
  return render(<BackendSyncedSettingsBadge />, {
    wrapper: ({ children }) => (
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    ),
  });
}

describe("BackendSyncedSettingsBadge", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    window.localStorage.clear();
    __resetActiveStoreForTests();
  });

  afterEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
  });

  it.each(["openhands-neutral", "light-plus", "solarized-light"] as const)(
    "keeps the rendered informational text readable in %s",
    (key) => {
      renderBadge();
      const text = screen.getByTestId(
        "backend-synced-settings-badge",
      ).firstElementChild!;
      const css = readFileSync("src/tailwind.css", "utf8");
      const theme = COLOR_THEMES[key];
      const tokens: Record<string, string> = {
        ...AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES,
        ...theme.scale,
        ...theme.tokens,
      };
      const resolve = (value: string): string => {
        const variable = /^var\((--[\w-]+)\)$/.exec(value);
        return variable ? resolve(tokens[variable[1]]) : value;
      };
      // Resolve the utility actually rendered, so reverting the component to
      // text-text-dim fails this test even if the palette remains unchanged.
      const colors = Array.from(text.classList).flatMap((name) => {
        if (!name.startsWith("text-")) return [];
        const match = css.match(
          new RegExp(`--color-${name.slice(5)}: ([^;]+);`),
        );
        return match ? [resolve(match[1])] : [];
      });
      expect(colors).toHaveLength(1);
      const luminance = (hex: string) => {
        const rgb = [1, 3, 5].map((offset) => {
          const c = parseInt(hex.slice(offset, offset + 2), 16) / 255;
          return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
        });
        return rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722;
      };
      // The shared badge appears in navigation, hubs, and inset drawers.
      for (const background of [
        "--oh-background",
        "--oh-surface",
        "--oh-surface-deep",
      ]) {
        const [low, high] = [
          luminance(colors[0]),
          luminance(resolve(tokens[background])),
        ].sort((a, b) => a - b);
        const ratio = (high + 0.05) / (low + 0.05);
        if (key === "openhands-neutral") {
          expect(ratio).toBeGreaterThanOrEqual(4.5);
        } else {
          // Light palette remediation belongs to #17730. This local fix must
          // improve, rather than regress, their existing informational text.
          const [oldLow, oldHigh] = [
            luminance(resolve(tokens["--oh-text-dim"])),
            luminance(resolve(tokens[background])),
          ].sort((a, b) => a - b);
          expect(ratio).toBeGreaterThanOrEqual(
            (oldHigh + 0.05) / (oldLow + 0.05),
          );
        }
      }
    },
  );

  it("renders the seeded default local backend label and host URL without HTML-escaping", () => {
    // Arrange — no explicit setRegisteredBackends() in this test; the
    // active store auto-seeds a default local backend named "Local".
    // Act
    renderBadge();

    // Assert — the default name slot resolves to "Local" and the host
    // URL is preserved literally (no `&#x2F;` from i18next escape).
    const text = screen.getByTestId(
      "backend-synced-settings-badge",
    ).textContent;
    expect(text).toMatch(
      /These settings are synced from Local backend \(https?:\/\/.+\)/,
    );
    expect(text).not.toContain("&#x2F;");
  });

  it("labels a cloud backend's active org as Personal Workspace when it matches the current user id", async () => {
    // Arrange
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "user-1" });
    vi.spyOn(orgService, "getCurrentCloudApiKey").mockResolvedValue({
      isLegacyKey: false,
      orgId: "user-1",
    } as never);
    vi.spyOn(orgService, "getCloudOrganizations").mockResolvedValue({
      items: [{ id: "user-1", name: "Hiep Le" }],
    } as never);
    vi.spyOn(orgService, "getCloudOrganizationMe").mockResolvedValue({
      orgId: "user-1",
      userId: "user-1",
    } as never);

    // Act
    renderBadge();

    // Assert
    await waitFor(() => {
      expect(
        screen.getByTestId("backend-synced-settings-badge").textContent,
      ).toContain("Production – BACKEND$PERSONAL_WORKSPACE");
    });
  });

  it("labels a cloud backend's active org with the org's display name when the user is not the org owner", async () => {
    // Arrange
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id, orgId: "team-org" });
    vi.spyOn(orgService, "getCurrentCloudApiKey").mockResolvedValue({
      isLegacyKey: false,
      orgId: "team-org",
    } as never);
    vi.spyOn(orgService, "getCloudOrganizations").mockResolvedValue({
      items: [{ id: "team-org", name: "Acme Team" }],
    } as never);
    vi.spyOn(orgService, "getCloudOrganizationMe").mockResolvedValue({
      orgId: "team-org",
      userId: "user-1",
    } as never);

    // Act
    renderBadge();

    // Assert
    await waitFor(() => {
      expect(
        screen.getByTestId("backend-synced-settings-badge").textContent,
      ).toContain("Production – Acme Team");
    });
  });
});
