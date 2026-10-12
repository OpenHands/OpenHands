import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { BackendRow } from "#/components/features/backends/backend-row";
import ApiKeyEntryScreen from "#/components/features/backends/api-key-entry-screen";
import { BackendFormModal } from "#/components/features/backends/backend-form-modal";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { __resetActiveStoreForTests } from "#/api/backend-registry/active-store";
import { type Backend } from "#/api/backend-registry/types";
import { type BackendHealth } from "#/hooks/query/use-backends-health";
import { COLOR_THEMES } from "#/themes/color-themes";
import { AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES } from "#/styles/agent-server-ui-style-scope";

const getSettingsMock = vi.fn();
const getServerInfoMock = vi.fn();
const useBackendsHealthMock = vi.fn();

vi.mock("#/hooks/query/use-backends-health", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/hooks/query/use-backends-health")>();
  return {
    ...actual,
    useBackendsHealth: (...args: unknown[]) => useBackendsHealthMock(...args),
  };
});

vi.mock("@openhands/typescript-client/clients", () => ({
  SettingsClient: vi.fn(function SettingsClientMock() {
    return { getSettings: getSettingsMock };
  }),
  ServerClient: vi.fn(function ServerClientMock() {
    return { getServerInfo: getServerInfoMock };
  }),
}));

vi.mock("#/api/cloud/organization-service.api", () => ({
  getCurrentCloudApiKey: vi.fn().mockResolvedValue({
    orgId: null,
    isLegacyKey: true,
  }),
}));

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>{ui}</ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

const mockBackend: Backend = {
  id: "test-backend",
  name: "Test Backend",
  host: "http://localhost:8000",
  apiKey: "test-key",
  kind: "local",
};

describe("Backend connection feedback theme roles (Issue #18319)", () => {
  beforeEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
    getSettingsMock.mockReset();
    getServerInfoMock.mockReset();
    useBackendsHealthMock.mockReset();
    getServerInfoMock.mockResolvedValue({ version: "1.52.0" });
    useBackendsHealthMock.mockReturnValue({
      [mockBackend.id]: { isConnected: true, lastError: null },
    });
  });

  describe("BackendRow status and error detail classes", () => {
    it("uses text-feedback-success and no raw green classes when connected", () => {
      const connectedHealth: BackendHealth = {
        isConnected: true,
        lastError: null,
        consecutiveFailures: 0,
        disabled: false,
      };

      renderWithProviders(
        <BackendRow
          backend={mockBackend}
          health={connectedHealth}
          onSelect={vi.fn()}
          onEdit={vi.fn()}
          onRemove={vi.fn()}
        />,
      );

      const statusEl = screen.getByTestId("manage-backends-status-Test Backend");
      expect(statusEl).toHaveClass("text-feedback-success");
      expect(statusEl.className).not.toMatch(/text-green-\d+/);
    });

    it("uses text-feedback-error and no raw red classes when disconnected", () => {
      const disconnectedHealth: BackendHealth = {
        isConnected: false,
        lastError: null,
        consecutiveFailures: 1,
        disabled: false,
      };

      renderWithProviders(
        <BackendRow
          backend={mockBackend}
          health={disconnectedHealth}
          onSelect={vi.fn()}
          onEdit={vi.fn()}
          onRemove={vi.fn()}
        />,
      );

      const statusEl = screen.getByTestId("manage-backends-status-Test Backend");
      expect(statusEl).toHaveClass("text-feedback-error");
      expect(statusEl.className).not.toMatch(/text-red-\d+/);
    });

    it("uses text-feedback-error and no raw red classes for status detail line", () => {
      const errorDetailHealth: BackendHealth = {
        isConnected: false,
        lastError: "Failed to connect to agent-server: 502",
        consecutiveFailures: 1,
        disabled: false,
      };

      renderWithProviders(
        <BackendRow
          backend={mockBackend}
          health={errorDetailHealth}
          onSelect={vi.fn()}
          onEdit={vi.fn()}
          onRemove={vi.fn()}
        />,
      );

      const detailEl = screen.getByTestId(
        "manage-backends-status-detail-Test Backend",
      );
      expect(detailEl).toHaveClass("text-feedback-error");
      expect(detailEl.className).not.toMatch(/text-red-\d+/);
    });
  });

  describe("ApiKeyEntryScreen status classes", () => {
    it("uses text-feedback-error and no raw red classes on validation failure", async () => {
      getSettingsMock.mockRejectedValueOnce(new Error("Unauthorized: 401"));
      renderWithProviders(<ApiKeyEntryScreen />);

      const user = userEvent.setup();
      await user.type(screen.getByTestId("api-key-entry-name"), "Test Server");
      await user.type(screen.getByTestId("api-key-entry-api-key"), "bad-key");
      await user.click(screen.getByTestId("api-key-entry-submit"));

      const statusEl = await screen.findByTestId("api-key-entry-status");
      expect(statusEl).toHaveClass("text-feedback-error");
      expect(statusEl.className).not.toMatch(/text-red-\d+/);
    });

    it("uses text-feedback-success and no raw green classes on validation success", async () => {
      getSettingsMock.mockResolvedValueOnce({ llm_model: "test-model" });
      renderWithProviders(<ApiKeyEntryScreen />);

      const user = userEvent.setup();
      await user.type(screen.getByTestId("api-key-entry-name"), "Test Server");
      await user.type(screen.getByTestId("api-key-entry-api-key"), "good-key");
      await user.click(screen.getByTestId("api-key-entry-submit"));

      const statusEl = await screen.findByTestId("api-key-entry-status");
      expect(statusEl).toHaveClass("text-feedback-success");
      expect(statusEl.className).not.toMatch(/text-green-\d+/);
    });
  });

  describe("BackendFormModal connection error and diagnostics panels", () => {
    it("uses semantic danger border/bg and text-feedback-error for connection error panel", async () => {
      getServerInfoMock.mockRejectedValueOnce(new Error("Connection refused"));

      renderWithProviders(
        <BackendFormModal
          mode="add"
          onClose={vi.fn()}
        />,
      );

      const user = userEvent.setup();
      // Click Agent Server tab to display local configuration fields
      await user.click(screen.getByTestId("add-backend-option-agent-server"));
      await user.type(screen.getByTestId("add-backend-name"), "Fail Server");
      await user.type(screen.getByTestId("add-backend-host"), "http://localhost:9999");
      await user.click(screen.getByTestId("add-backend-submit"));

      const errorPanel = await screen.findByTestId("add-backend-error");
      expect(errorPanel).toHaveClass("border-danger/40");
      expect(errorPanel).toHaveClass("bg-danger/5");
      expect(errorPanel).toHaveClass("text-feedback-error");
      expect(errorPanel.className).not.toMatch(/(text|border|bg)-red-\d+/);
    });

    it("uses semantic danger border/bg and text-feedback-error for disabled diagnostics panel", async () => {
      useBackendsHealthMock.mockReturnValue({
        [mockBackend.id]: {
          isConnected: false,
          disabled: true,
          consecutiveFailures: 3,
          lastError: "Failed 3 times in a row",
        },
      });

      renderWithProviders(
        <BackendFormModal
          mode="edit"
          backend={mockBackend}
          onClose={vi.fn()}
        />,
      );

      const errorPanel = await screen.findByTestId("edit-backend-status-error");
      expect(errorPanel).toHaveClass("border-danger/40");
      expect(errorPanel).toHaveClass("bg-danger/5");
      expect(errorPanel.className).not.toMatch(/(border|bg)-red-\d+/);

      const errorMessage = screen.getByTestId(
        "edit-backend-status-error-message",
      );
      expect(errorMessage).toHaveClass("text-feedback-error");
      expect(errorMessage.className).not.toMatch(/text-red-\d+/);
    });
  });

  describe("Theme contrast on composited error panel background", () => {
    function parseRgb(hex: string): [number, number, number] {
      return [1, 3, 5].map((offset) =>
        parseInt(hex.slice(offset, offset + 2), 16),
      ) as [number, number, number];
    }

    function luminance(rgb: [number, number, number]): number {
      const channels = rgb.map((val) => {
        const channel = val / 255;
        return channel <= 0.04045
          ? channel / 12.92
          : ((channel + 0.055) / 1.055) ** 2.4;
      });
      return (
        channels[0] * 0.2126 + channels[1] * 0.7152 + channels[2] * 0.0722
      );
    }

    function composite(
      bgRgb: [number, number, number],
      fgRgb: [number, number, number],
      fgAlpha: number,
    ): [number, number, number] {
      return bgRgb.map((bgVal, i) =>
        Math.round(fgRgb[i] * fgAlpha + bgVal * (1 - fgAlpha)),
      ) as [number, number, number];
    }

    function contrast(
      rgb1: [number, number, number],
      rgb2: [number, number, number],
    ): number {
      const l1 = luminance(rgb1);
      const l2 = luminance(rgb2);
      return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
    }

    it.each(Object.entries(COLOR_THEMES))(
      "preserves >= 4.5:1 contrast between feedback-error and composited panel in %s",
      (_key, theme) => {
        const variables: Record<string, string> = {
          ...AGENT_SERVER_UI_DEFAULT_CSS_VARIABLES,
          ...theme.scale,
          ...theme.tokens,
        };
        const resolve = (name: string): string => {
          const value = variables[name];
          expect(value, `Missing theme token ${name}`).toBeDefined();
          const reference = /^var\((--[\w-]+)\)$/.exec(value);
          return reference ? resolve(reference[1]) : value;
        };

        const baseSecondaryRgb = parseRgb(resolve("--oh-color-base-secondary"));
        const dangerRgb = parseRgb(resolve("--oh-color-danger"));
        const feedbackErrorRgb = parseRgb(resolve("--oh-feedback-error"));

        // Composited panel background: base-secondary with 5% danger tint
        const compositedPanelRgb = composite(baseSecondaryRgb, dangerRgb, 0.05);
        const ratio = contrast(feedbackErrorRgb, compositedPanelRgb);

        expect(
          ratio,
          `feedback-error on composited panel in ${_key}`,
        ).toBeGreaterThanOrEqual(4.5);
      },
    );
  });
});
