import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as telemetry from "#/services/telemetry";

const useSettingsMock = vi.fn();
vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => useSettingsMock(),
}));

const compatibilityMocks = vi.hoisted(() => ({
  getCachedAgentServerVersion: vi.fn(),
}));
vi.mock("#/api/agent-server-compatibility", () => ({
  getCachedAgentServerVersion: compatibilityMocks.getCachedAgentServerVersion,
}));

const automationSdkVersionHookMock = vi.hoisted(() => ({
  useAutomationSdkVersion: vi.fn(),
}));
vi.mock("#/hooks/query/use-automation-sdk-version", () => ({
  useAutomationSdkVersion: automationSdkVersionHookMock.useAutomationSdkVersion,
}));

import { useTracking } from "#/hooks/use-tracking";

// Resolved at test-run time so it matches whatever URL jsdom is configured
// with in the current environment (varies between local and CI).
let COMMON: Record<string, unknown>;

describe("useTracking", () => {
  let captureMock: ReturnType<typeof vi.spyOn>;
  let setBackendContextMock: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    captureMock = vi
      .spyOn(telemetry, "trackEvent")
      .mockResolvedValue(undefined);
    setBackendContextMock = vi.spyOn(telemetry, "setTelemetryBackendContext");
    useSettingsMock.mockReset();
    automationSdkVersionHookMock.useAutomationSdkVersion.mockReset();
    automationSdkVersionHookMock.useAutomationSdkVersion.mockReturnValue(null);
    compatibilityMocks.getCachedAgentServerVersion.mockReset();
    compatibilityMocks.getCachedAgentServerVersion.mockReturnValue(null);

    useSettingsMock.mockReturnValue({
      data: { email: "user@example.com", user_consents_to_analytics: true },
    });
    COMMON = {
      current_url: window.location.href,
    };
  });

  afterEach(() => {
    captureMock.mockRestore();
    setBackendContextMock.mockRestore();
  });

  const getTracking = () => renderHook(() => useTracking()).result.current;

  describe("trackLoginButtonClick", () => {
    it("captures login_button_clicked with provider and commonProperties", () => {
      getTracking().trackLoginButtonClick({ provider: "github" });

      expect(captureMock).toHaveBeenCalledWith("login_button_clicked", {
        provider: "github",
        ...COMMON,
      });
    });
  });

  describe("trackConversationCreated", () => {
    it("captures conversation_start_requested with full metadata for a repository, task-start conversation", () => {
      getTracking().trackConversationCreated({
        conversationId: "task-abc123",
        taskId: "abc123",
        hasRepository: true,
        gitProvider: "github",
        hasWorkspace: false,
        hasInitialQuery: true,
        hasParentConversation: false,
        entryPoint: "sidebar_cloud_menu",
      });

      expect(captureMock).toHaveBeenCalledWith("conversation_start_requested", {
        conversation_id: "task-abc123",
        task_id: "abc123",
        // task-prefixed conversation_id => cloud sandbox still provisioning
        is_start_task: true,
        has_repository: true,
        git_provider: "github",
        has_workspace: false,
        workspace_mode: undefined,
        has_initial_query: true,
        agent_type: undefined,
        has_parent_conversation: false,
        entry_point: "sidebar_cloud_menu",
        ...COMMON,
      });
    });

    it("captures conversation_start_requested for a local workspace start with no repository", () => {
      getTracking().trackConversationCreated({
        conversationId: "conv-1",
        taskId: "conv-1",
        hasRepository: false,
        hasWorkspace: true,
        workspaceMode: "local_repo",
        hasInitialQuery: false,
        hasParentConversation: false,
        entryPoint: "sidebar_local_menu",
      });

      expect(captureMock).toHaveBeenCalledWith("conversation_start_requested", {
        conversation_id: "conv-1",
        task_id: "conv-1",
        // real conversation_id => ready immediately (local)
        is_start_task: false,
        has_repository: false,
        git_provider: undefined,
        has_workspace: true,
        workspace_mode: "local_repo",
        has_initial_query: false,
        agent_type: undefined,
        has_parent_conversation: false,
        entry_point: "sidebar_local_menu",
        ...COMMON,
      });
    });

    it("captures agent_type and has_parent_conversation for a plan sub-conversation", () => {
      getTracking().trackConversationCreated({
        conversationId: "task-plan-1",
        taskId: "plan-1",
        hasRepository: false,
        hasWorkspace: false,
        hasInitialQuery: false,
        agentType: "plan",
        hasParentConversation: true,
        entryPoint: "plan_sub_conversation",
      });

      expect(captureMock).toHaveBeenCalledWith(
        "conversation_start_requested",
        expect.objectContaining({
          agent_type: "plan",
          has_parent_conversation: true,
        }),
      );
    });

    it("declares backend versions to the telemetry service when known", async () => {
      automationSdkVersionHookMock.useAutomationSdkVersion.mockReturnValue(
        "1.36.3",
      );
      compatibilityMocks.getCachedAgentServerVersion.mockReturnValue("1.36.2");

      const { result } = renderHook(() => useTracking());
      await waitFor(() =>
        expect(setBackendContextMock).toHaveBeenCalledWith(
          expect.objectContaining({ automationSdkVersion: "1.36.3" }),
        ),
      );
      setBackendContextMock.mockClear();

      result.current.trackConversationCreated({
        conversationId: "conv-versioned",
        taskId: "conv-versioned",
        hasRepository: false,
        hasWorkspace: false,
        hasInitialQuery: true,
        hasParentConversation: false,
        entryPoint: "home",
      });

      expect(setBackendContextMock).toHaveBeenCalledWith(
        expect.objectContaining({
          backendKind: "local",
          agentServerVersion: "1.36.2",
          automationSdkVersion: "1.36.3",
        }),
      );
      expect(captureMock).toHaveBeenCalledWith(
        "conversation_start_requested",
        expect.not.objectContaining({
          agent_server_version: "1.36.2",
          automation_sdk_version: "1.36.3",
          backend_version: "1.36.2",
        }),
      );
      expect(
        compatibilityMocks.getCachedAgentServerVersion,
      ).toHaveBeenCalledWith(expect.any(String));
    });

    it("uses the latest cached agent-server version when an event fires", () => {
      compatibilityMocks.getCachedAgentServerVersion.mockReturnValue(null);
      const tracking = getTracking();
      setBackendContextMock.mockClear();
      compatibilityMocks.getCachedAgentServerVersion.mockReturnValue("1.36.1");

      tracking.trackConversationCreated({
        conversationId: "conv-cache-late",
        taskId: "conv-cache-late",
        hasRepository: false,
        hasWorkspace: false,
        hasInitialQuery: true,
        hasParentConversation: false,
        entryPoint: "home",
      });

      expect(setBackendContextMock).toHaveBeenCalledWith(
        expect.objectContaining({
          backendKind: "local",
          agentServerVersion: "1.36.1",
        }),
      );
    });
  });

  describe("trackPushButtonClick", () => {
    it("captures push_button_clicked with commonProperties", () => {
      getTracking().trackPushButtonClick();

      expect(captureMock).toHaveBeenCalledWith(
        "push_button_clicked",
        expect.objectContaining(COMMON),
      );
    });
  });

  describe("trackPullButtonClick", () => {
    it("captures pull_button_clicked with commonProperties", () => {
      getTracking().trackPullButtonClick();

      expect(captureMock).toHaveBeenCalledWith(
        "pull_button_clicked",
        expect.objectContaining(COMMON),
      );
    });
  });

  describe("trackCreatePrButtonClick", () => {
    it("captures create_pr_button_clicked with commonProperties", () => {
      getTracking().trackCreatePrButtonClick();

      expect(captureMock).toHaveBeenCalledWith(
        "create_pr_button_clicked",
        expect.objectContaining(COMMON),
      );
    });
  });

  describe("trackUserSignupCompleted", () => {
    it("captures user_signup_completed with signup_timestamp and commonProperties", () => {
      getTracking().trackUserSignupCompleted();

      expect(captureMock).toHaveBeenCalledWith(
        "user_signup_completed",
        expect.objectContaining({
          signup_timestamp: expect.any(String),
          ...COMMON,
        }),
      );
    });
  });

  describe("trackPrebuiltAutomationEnabled", () => {
    it("captures prebuilt_automation_enabled with all fields", () => {
      getTracking().trackPrebuiltAutomationEnabled({
        automationId: "auto-1",
        automationName: "My Automation",
        automationCategory: "CI",
      });

      expect(captureMock).toHaveBeenCalledWith("prebuilt_automation_enabled", {
        automation_id: "auto-1",
        automation_name: "My Automation",
        automation_category: "CI",
        ...COMMON,
      });
    });

    it("sends undefined fields when optional args are omitted", () => {
      getTracking().trackPrebuiltAutomationEnabled({
        automationName: "Minimal",
      });

      expect(captureMock).toHaveBeenCalledWith(
        "prebuilt_automation_enabled",
        expect.objectContaining({
          automation_name: "Minimal",
          automation_id: undefined,
          automation_category: undefined,
        }),
      );
    });
  });

  describe("trackInitialQuerySubmitted", () => {
    it("captures initial_query_submitted with all properties", () => {
      getTracking().trackInitialQuerySubmitted({
        entryPoint: "github",
        queryCharacterLength: 42,
        replayJsonSize: 100,
      });

      expect(captureMock).toHaveBeenCalledWith("initial_query_submitted", {
        entry_point: "github",
        query_character_length: 42,
        replay_json_size: 100,
        ...COMMON,
      });
    });

    it("sends undefined replayJsonSize when omitted", () => {
      getTracking().trackInitialQuerySubmitted({
        entryPoint: "direct",
        queryCharacterLength: 10,
      });

      expect(captureMock).toHaveBeenCalledWith(
        "initial_query_submitted",
        expect.objectContaining({ replay_json_size: undefined }),
      );
    });
  });

  describe("trackUserMessageSent", () => {
    it("captures user_message_sent with message counts", () => {
      getTracking().trackUserMessageSent({
        sessionMessageCount: 5,
        currentMessageLength: 120,
      });

      expect(captureMock).toHaveBeenCalledWith("user_message_sent", {
        session_message_count: 5,
        current_message_length: 120,
        ...COMMON,
      });
    });
  });

  describe("trackDownloadVsCodeButtonClicked", () => {
    it("captures download_via_vscode_button_clicked with commonProperties", () => {
      getTracking().trackDownloadVsCodeButtonClicked();

      expect(captureMock).toHaveBeenCalledWith(
        "download_via_vscode_button_clicked",
        expect.objectContaining(COMMON),
      );
    });
  });

  describe("trackSettingsSaved", () => {
    it("captures settings_saved with all properties using SCREAMING_SNAKE_CASE keys", () => {
      getTracking().trackSettingsSaved({
        llmModel: "gpt-4",
        llmApiKeySet: "SET",
        searchApiKeySet: "UNSET",
        remoteRuntimeResourceFactor: 2,
      });

      expect(captureMock).toHaveBeenCalledWith("settings_saved", {
        LLM_MODEL: "gpt-4",
        LLM_API_KEY_SET: "SET",
        SEARCH_API_KEY_SET: "UNSET",
        REMOTE_RUNTIME_RESOURCE_FACTOR: 2,
        ...COMMON,
      });
    });
  });

  describe("trackMcpConfigUpdated", () => {
    it("captures mcp_config_updated with server counts and has_mcp_config: true", () => {
      getTracking().trackMcpConfigUpdated({
        sseServersCount: 2,
        stdioServersCount: 1,
      });

      expect(captureMock).toHaveBeenCalledWith("mcp_config_updated", {
        has_mcp_config: true,
        sse_servers_count: 2,
        stdio_servers_count: 1,
        ...COMMON,
      });
    });
  });

  describe("trackDownloadTrajectoryButtonClicked", () => {
    it("captures download_trajectory_button_clicked with commonProperties", () => {
      getTracking().trackDownloadTrajectoryButtonClicked();

      expect(captureMock).toHaveBeenCalledWith(
        "download_trajectory_button_clicked",
        expect.objectContaining(COMMON),
      );
    });
  });

  describe("trackBackendAdded", () => {
    it("captures backend_added with coarse, non-sensitive backend metadata", () => {
      getTracking().trackBackendAdded({
        backendKind: "cloud",
        connectionMethod: "cloud_login",
        hasApiKey: true,
        source: "add_backend_modal",
        agentServerVersion: "1.36.1",
      });

      expect(captureMock).toHaveBeenCalledWith("backend_added", {
        ...COMMON,
        backend_kind: "cloud",
        connection_method: "cloud_login",
        has_api_key: true,
        source: "add_backend_modal",
        agent_server_version: "1.36.1",
        automation_sdk_version: "unknown",
        backend_version: "1.36.1",
      });
    });
  });

  describe("shared telemetry boundary", () => {
    it("delegates consent enforcement when backend settings report false", () => {
      useSettingsMock.mockReturnValue({
        data: { email: "user@example.com", user_consents_to_analytics: false },
      });

      getTracking().trackPushButtonClick();

      expect(captureMock).toHaveBeenCalledWith(
        "push_button_clicked",
        expect.objectContaining(COMMON),
      );
    });

    it("delegates consent enforcement when backend settings report null", () => {
      useSettingsMock.mockReturnValue({
        data: { email: "user@example.com", user_consents_to_analytics: null },
      });

      getTracking().trackPushButtonClick();

      expect(captureMock).toHaveBeenCalledWith(
        "push_button_clicked",
        expect.objectContaining(COMMON),
      );
    });

    it("does not drop backend milestones while settings are loading", () => {
      useSettingsMock.mockReturnValue({ data: undefined });

      getTracking().trackBackendAdded({
        backendKind: "cloud",
        connectionMethod: "cloud_login",
        hasApiKey: true,
        source: "onboarding",
      });

      expect(captureMock).toHaveBeenCalledWith(
        "backend_added",
        expect.objectContaining({
          backend_kind: "cloud",
          connection_method: "cloud_login",
          source: "onboarding",
        }),
      );
    });
  });
});
