import { afterEach, describe, expect, it, vi } from "vitest";
import {
  renderWithProviders,
  createAxiosNotFoundErrorObject,
} from "test-utils";
import { createRoutesStub } from "react-router";
import { screen, waitFor } from "@testing-library/react";
import { Sidebar } from "#/components/features/sidebar/sidebar";
import SettingsService from "#/api/settings-service/settings-service.api";
import OptionService from "#/api/option-service/option-service.api";
import { MOCK_DEFAULT_USER_SETTINGS } from "#/mocks/handlers";
import { GetConfigResponse } from "#/api/option-service/option.types";

// Helper to create mock config with sensible defaults
const createMockConfig = (
  overrides: Omit<Partial<GetConfigResponse>, "FEATURE_FLAGS"> & {
    FEATURE_FLAGS?: Partial<GetConfigResponse["FEATURE_FLAGS"]>;
  } = {},
): GetConfigResponse => {
  const { FEATURE_FLAGS: featureFlagOverrides, ...restOverrides } = overrides;
  return {
    APP_MODE: "oss",
    GITHUB_CLIENT_ID: "test-client-id",
    POSTHOG_CLIENT_KEY: "test-posthog-key",
    FEATURE_FLAGS: {
      ENABLE_BILLING: false,
      HIDE_LLM_SETTINGS: false,
      ENABLE_JIRA: false,
      ENABLE_JIRA_DC: false,
      ENABLE_LINEAR: false,
      ...featureFlagOverrides,
    },
    ...restOverrides,
  };
};

// These tests will now fail because the conversation panel is rendered through a portal
// and technically not a child of the Sidebar component.

const ConversationRouterStub = createRoutesStub([
  {
    path: "/conversation/:conversationId",
    Component: () => <Sidebar />,
  },
]);

const SettingsRouterStub = createRoutesStub([
  {
    path: "/settings",
    Component: () => <Sidebar />,
  },
]);

const renderSidebar = (path: "conversation" | "settings" = "conversation") => {
  if (path === "settings") {
    return renderWithProviders(
      <SettingsRouterStub initialEntries={["/settings"]} />,
    );
  }
  return renderWithProviders(
    <ConversationRouterStub initialEntries={["/conversation/123"]} />,
  );
};

describe("Sidebar", () => {
  const getSettingsSpy = vi.spyOn(SettingsService, "getSettings");
  const getConfigSpy = vi.spyOn(OptionService, "getConfig");

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should fetch settings data on mount", async () => {
    renderSidebar();
    await waitFor(() => expect(getSettingsSpy).toHaveBeenCalled());
  });

  describe("Settings modal auto-open behavior", () => {
    it("should NOT open settings modal when HIDE_LLM_SETTINGS is true even with 404 error", async () => {
      getConfigSpy.mockResolvedValue(
        createMockConfig({ FEATURE_FLAGS: { HIDE_LLM_SETTINGS: true } }),
      );
      getSettingsSpy.mockRejectedValue(createAxiosNotFoundErrorObject());

      renderSidebar();

      await waitFor(() => {
        expect(getConfigSpy).toHaveBeenCalled();
        expect(getSettingsSpy).toHaveBeenCalled();
      });

      // Settings modal should NOT appear when HIDE_LLM_SETTINGS is true
      await waitFor(() => {
        expect(screen.queryByTestId("ai-config-modal")).not.toBeInTheDocument();
      });
    });

    it("should open settings modal when HIDE_LLM_SETTINGS is false and 404 error in OSS mode", async () => {
      getConfigSpy.mockResolvedValue(
        createMockConfig({ FEATURE_FLAGS: { HIDE_LLM_SETTINGS: false } }),
      );
      getSettingsSpy.mockRejectedValue(createAxiosNotFoundErrorObject());

      renderSidebar();

      // Settings modal should appear when HIDE_LLM_SETTINGS is false
      await waitFor(() => {
        expect(screen.getByTestId("ai-config-modal")).toBeInTheDocument();
      });
    });

    it("should NOT open settings modal in SaaS mode even with 404 error", async () => {
      getConfigSpy.mockResolvedValue(
        createMockConfig({
          APP_MODE: "saas",
          FEATURE_FLAGS: { HIDE_LLM_SETTINGS: false },
        }),
      );
      getSettingsSpy.mockRejectedValue(createAxiosNotFoundErrorObject());

      renderSidebar();

      await waitFor(() => {
        expect(getConfigSpy).toHaveBeenCalled();
        expect(getSettingsSpy).toHaveBeenCalled();
      });

      // Settings modal should NOT appear in SaaS mode (only opens in OSS mode)
      await waitFor(() => {
        expect(screen.queryByTestId("ai-config-modal")).not.toBeInTheDocument();
      });
    });

    it("should NOT open settings modal when settings exist (no 404 error)", async () => {
      getConfigSpy.mockResolvedValue(
        createMockConfig({ FEATURE_FLAGS: { HIDE_LLM_SETTINGS: false } }),
      );
      getSettingsSpy.mockResolvedValue(MOCK_DEFAULT_USER_SETTINGS);

      renderSidebar();

      await waitFor(() => {
        expect(getConfigSpy).toHaveBeenCalled();
        expect(getSettingsSpy).toHaveBeenCalled();
      });

      // Settings modal should NOT appear when settings exist
      await waitFor(() => {
        expect(screen.queryByTestId("ai-config-modal")).not.toBeInTheDocument();
      });
    });

    it("should NOT open settings modal when on /settings path", async () => {
      getConfigSpy.mockResolvedValue(
        createMockConfig({ FEATURE_FLAGS: { HIDE_LLM_SETTINGS: false } }),
      );
      getSettingsSpy.mockRejectedValue(createAxiosNotFoundErrorObject());

      renderSidebar("settings");

      await waitFor(() => {
        expect(getConfigSpy).toHaveBeenCalled();
        expect(getSettingsSpy).toHaveBeenCalled();
      });

      // Settings modal should NOT appear when on /settings path
      // (prevents modal from showing when user is already viewing settings)
      await waitFor(() => {
        expect(screen.queryByTestId("ai-config-modal")).not.toBeInTheDocument();
      });
    });
  });
});
