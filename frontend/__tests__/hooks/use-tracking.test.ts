import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useTracking } from "#/hooks/use-tracking";

// Mock PostHog
const mockCapture = vi.fn();
vi.mock("posthog-js/react", () => ({
  usePostHog: () => ({
    capture: mockCapture,
  }),
}));

// Mock useConfig
vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({
    data: { APP_MODE: "saas" },
  }),
}));

// Mock useSettings
vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({
    data: { email: "test@example.com", git_user_email: "git@example.com" },
  }),
}));

describe("useTracking", () => {
  beforeEach(() => {
    mockCapture.mockClear();
  });

  const expectedCommonProperties = {
    app_surface: "saas",
    plan_tier: null,
    current_url: expect.any(String),
    user_email: "test@example.com",
  };

  describe("trackDownloadViaVSCodeButtonClick", () => {
    it("should capture download_via_vscode_button_clicked with common properties", () => {
      const { result } = renderHook(() => useTracking());
      result.current.trackDownloadViaVSCodeButtonClick();

      expect(mockCapture).toHaveBeenCalledWith(
        "download_via_vscode_button_clicked",
        expectedCommonProperties,
      );
    });
  });

  describe("trackDownloadTrajectoryButtonClick", () => {
    it("should capture download_trajectory_button_clicked with common properties", () => {
      const { result } = renderHook(() => useTracking());
      result.current.trackDownloadTrajectoryButtonClick();

      expect(mockCapture).toHaveBeenCalledWith(
        "download_trajectory_button_clicked",
        expectedCommonProperties,
      );
    });
  });

  describe("trackMcpConfigUpdated", () => {
    it("should capture mcp_config_updated with custom and common properties", () => {
      const { result } = renderHook(() => useTracking());
      result.current.trackMcpConfigUpdated({
        hasMcpConfig: true,
        sseServersCount: 2,
        stdioServersCount: 3,
      });

      expect(mockCapture).toHaveBeenCalledWith("mcp_config_updated", {
        has_mcp_config: true,
        sse_servers_count: 2,
        stdio_servers_count: 3,
        ...expectedCommonProperties,
      });
    });
  });

  describe("trackSettingsSaved", () => {
    it("should capture settings_saved with custom and common properties", () => {
      const { result } = renderHook(() => useTracking());
      result.current.trackSettingsSaved({
        llmModel: "gpt-4",
        llmApiKeySet: "SET",
        searchApiKeySet: "UNSET",
        remoteRuntimeResourceFactor: 2,
      });

      expect(mockCapture).toHaveBeenCalledWith("settings_saved", {
        LLM_MODEL: "gpt-4",
        LLM_API_KEY_SET: "SET",
        SEARCH_API_KEY_SET: "UNSET",
        REMOTE_RUNTIME_RESOURCE_FACTOR: 2,
        ...expectedCommonProperties,
      });
    });
  });

  describe("trackInitialQuerySubmitted", () => {
    it("should capture initial_query_submitted with custom and common properties", () => {
      const { result } = renderHook(() => useTracking());
      result.current.trackInitialQuerySubmitted({
        entryPoint: "github",
        queryCharacterLength: 150,
        replayJsonSize: 1024,
      });

      expect(mockCapture).toHaveBeenCalledWith("initial_query_submitted", {
        entry_point: "github",
        query_character_length: 150,
        replay_json_size: 1024,
        ...expectedCommonProperties,
      });
    });
  });

  describe("trackUserMessageSent", () => {
    it("should capture user_message_sent with custom and common properties", () => {
      const { result } = renderHook(() => useTracking());
      result.current.trackUserMessageSent({
        sessionMessageCount: 5,
        currentMessageLength: 200,
      });

      expect(mockCapture).toHaveBeenCalledWith("user_message_sent", {
        session_message_count: 5,
        current_message_length: 200,
        ...expectedCommonProperties,
      });
    });
  });
});
