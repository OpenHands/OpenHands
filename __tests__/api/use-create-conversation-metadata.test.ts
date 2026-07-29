import {
  ConversationClient,
  SettingsClient,
} from "@openhands/typescript-client/clients";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import React from "react";
import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { getStoredConversationMetadata } from "#/api/conversation-metadata-store";

const {
  mockHttpPost,
  mockConversationClient,
  mockSettingsClient,
  mockGetSettings,
  mockGetSettingsForConversation,
  mockUseLlmProfiles,
  mockListInstalledPlugins,
} = vi.hoisted(() => ({
  mockHttpPost: vi.fn(),
  mockConversationClient: vi.fn(),
  mockSettingsClient: vi.fn(),
  mockGetSettings: vi.fn(),
  mockGetSettingsForConversation: vi.fn(),
  mockUseLlmProfiles: vi.fn(),
  mockListInstalledPlugins: vi.fn(),
}));

vi.mock("@openhands/typescript-client/clients", async () => {
  const actual = await vi.importActual<
    typeof import("@openhands/typescript-client/clients")
  >("@openhands/typescript-client/clients");
  return {
    ...actual,
    ConversationClient: vi.fn(function ConversationClientMock() {
      return mockConversationClient();
    }),
    SettingsClient: vi.fn(function SettingsClientMock() {
      return mockSettingsClient();
    }),
    VSCodeClient: vi.fn(function VSCodeClientMock() {
      return { getUrl: vi.fn() };
    }),
  };
});

vi.mock("#/api/agent-server-config", () => ({
  DEFAULT_WORKING_DIR: "workspace/project",
  getAgentServerBaseUrl: vi.fn(() => "http://localhost:54928"),
  getBakedSessionApiKey: vi.fn(() => "test-session-key"),
  getAgentServerSessionApiKey: vi.fn(() => "test-session-key"),
  getAgentServerWorkingDir: vi.fn(() => "/workspace/project/agent-canvas"),
  buildConversationWorkingDir: vi.fn(
    (id: string) => `/state/workspaces/${id.replace(/-/g, "")}`,
  ),
  shouldLoadPublicSkills: vi.fn(() => true),
  syncBakedSessionApiKey: vi.fn(),
  getLockedCloudHost: vi.fn(() => null),
}));

vi.mock("#/api/settings-service/settings-service.api", () => ({
  default: {
    getSettings: mockGetSettings,
    getSettingsForConversation: mockGetSettingsForConversation,
  },
}));

vi.mock("#/hooks/use-tracking", () => ({
  useTracking: () => ({ trackConversationCreated: vi.fn() }),
}));

vi.mock("#/hooks/query/use-llm-profiles", () => ({
  useLlmProfiles: () => mockUseLlmProfiles(),
}));

vi.mock("#/api/plugins-management-service", () => ({
  default: {
    listInstalledPlugins: mockListInstalledPlugins,
  },
}));

const wrapper = ({ children }: { children: React.ReactNode }) => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return React.createElement(QueryClientProvider, { client }, children);
};

describe("useCreateConversation persists selected repository metadata", () => {
  beforeEach(() => {
    window.localStorage.clear();
    mockUseLlmProfiles.mockReset();
    // Default: no active profile, so metadata is written only for repo/
    // workspace attachments (as before). Individual tests override this.
    mockUseLlmProfiles.mockReturnValue({ data: { active_profile: null } });
    mockListInstalledPlugins.mockReset();
    // Default: no installed plugins, so the existing repo/workspace/profile
    // assertions are unaffected. Plugin-specific tests override this.
    mockListInstalledPlugins.mockResolvedValue([]);
    mockHttpPost.mockReset();
    mockGetSettings.mockReset();
    mockGetSettingsForConversation.mockReset();
    mockGetSettings.mockResolvedValue({
      agent_settings: { llm: { model: "gpt-4o" } },
      conversation_settings: {},
    });
    mockGetSettingsForConversation.mockResolvedValue({
      agentSettings: { llm: { model: "gpt-4o" } },
      conversationSettings: {},
      secretsEncrypted: true,
    });
    mockConversationClient.mockReset();
    vi.mocked(ConversationClient).mockClear();
    vi.mocked(SettingsClient).mockClear();
    mockConversationClient.mockReturnValue({
      createConversation: async (payload: unknown) => {
        const response = await mockHttpPost("/api/conversations", payload);
        return response.data;
      },
    });
    mockSettingsClient.mockReturnValue({
      listSecrets: vi.fn().mockResolvedValue({ secrets: [] }),
    });
    mockHttpPost.mockResolvedValue({
      data: {
        id: "conv-new",
        created_at: "2026-05-05T00:00:00Z",
        updated_at: "2026-05-05T00:00:00Z",
      },
    });
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it("stores the selected repo/branch/provider in the metadata store after a successful create", async () => {
    const { result } = renderHook(() => useCreateConversation(), { wrapper });

    result.current.mutate({
      query: "ship it",
      repository: {
        name: "octocat/hello-world",
        gitProvider: "github",
        branch: "main",
      },
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(getStoredConversationMetadata("conv-new")).toEqual({
      selected_repository: "octocat/hello-world",
      selected_branch: "main",
      git_provider: "github",
      selected_workspace: null,
      workspace_mode: "new_worktree",
    });
  });

  it("stores the selected workspace path when only a workspace (no repo) is attached", async () => {
    const { result } = renderHook(() => useCreateConversation(), { wrapper });

    result.current.mutate({
      query: "poke at this repo",
      workingDir: "/home/me/code/some-project",
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    // We persist the workspace path so `useHasAttachedSource` can default
    // the Files tab to diff view even when no repo was picked.
    expect(getStoredConversationMetadata("conv-new")).toEqual({
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      selected_workspace: "/home/me/code/some-project",
      workspace_mode: "local_repo",
    });
  });

  it("does not write metadata when neither a repository nor a workspace is attached", async () => {
    const { result } = renderHook(() => useCreateConversation(), { wrapper });

    result.current.mutate({ query: "scratch session" });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(getStoredConversationMetadata("conv-new")).toBeNull();
  });

  it("stamps the active LLM profile even when no repo or workspace is attached (#1082)", async () => {
    mockUseLlmProfiles.mockReturnValue({
      data: { active_profile: "team-default" },
    });

    const { result } = renderHook(() => useCreateConversation(), { wrapper });

    result.current.mutate({ query: "scratch session" });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(getStoredConversationMetadata("conv-new")).toEqual({
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      selected_workspace: null,
      workspace_mode: null,
      active_profile: "team-default",
      plugins: null,
    });
  });

  it("records the enabled installed plugins (excluding disabled ones) so the in-conversation plugins view can show them", async () => {
    // Arrange: one enabled and one disabled plugin are installed locally.
    mockListInstalledPlugins.mockResolvedValue([
      {
        name: "city-weather",
        version: "1.0.0",
        description: null,
        enabled: true,
        source: "github:acme/city-weather",
        resolved_ref: "abc123",
        repo_path: null,
        installed_at: "2026-05-05T00:00:00Z",
        install_path: "/plugins/city-weather",
      },
      {
        name: "stocks",
        version: "1.0.0",
        description: null,
        enabled: false,
        source: "github:acme/stocks",
        resolved_ref: null,
        repo_path: null,
        installed_at: "2026-05-05T00:00:00Z",
        install_path: "/plugins/stocks",
      },
    ]);

    const { result } = renderHook(() => useCreateConversation(), { wrapper });

    result.current.mutate({ query: "what's the weather?" });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    // Only the enabled plugin is snapshotted, mapped to coordinates
    // (resolved_ref -> ref) and keeping its name for display.
    expect(getStoredConversationMetadata("conv-new")?.plugins).toEqual([
      {
        source: "github:acme/city-weather",
        ref: "abc123",
        repo_path: null,
        name: "city-weather",
      },
    ]);
  });

  it("does not duplicate a plugin that is both explicitly attached and enabled-installed", async () => {
    // Arrange: the same plugin is passed explicitly and is enabled-installed.
    mockListInstalledPlugins.mockResolvedValue([
      {
        name: "city-weather",
        version: "1.0.0",
        description: null,
        enabled: true,
        source: "github:acme/city-weather",
        resolved_ref: "main",
        repo_path: null,
        installed_at: "2026-05-05T00:00:00Z",
        install_path: "/plugins/city-weather",
      },
    ]);

    const { result } = renderHook(() => useCreateConversation(), { wrapper });

    result.current.mutate({
      query: "what's the weather?",
      plugins: [
        { source: "github:acme/city-weather", ref: "main", repo_path: null },
      ],
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    // The plugin is recorded exactly once.
    expect(getStoredConversationMetadata("conv-new")?.plugins).toEqual([
      { source: "github:acme/city-weather", ref: "main", repo_path: null },
    ]);
  });
});
