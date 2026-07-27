import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AxiosError, AxiosHeaders } from "axios";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { useSwitchLlmProfile } from "#/hooks/mutation/use-switch-llm-profile";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  LLM_PROFILES_QUERY_KEYS,
  SETTINGS_QUERY_KEYS,
} from "#/hooks/query/query-keys";
import { recordModelSwitchMessage } from "#/hooks/chat/record-model-switch-message";
import {
  getStoredConversationMetadata,
  setStoredConversationMetadata,
} from "#/api/conversation-metadata-store";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

vi.mock("#/utils/custom-toast-handlers");

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      switchProfile: vi.fn(),
    },
  }),
);

vi.mock("#/hooks/chat/record-model-switch-message", () => ({
  recordModelSwitchMessage: vi.fn(),
}));

vi.mock("#/hooks/chat/model-command-event-anchor", () => ({
  getLastRenderableEventId: () => "evt-9",
}));

const renderSwitchHook = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  const invalidateQueriesSpy = vi.spyOn(queryClient, "invalidateQueries");
  const { result } = renderHook(() => useSwitchLlmProfile(), {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
  return { result, invalidateQueriesSpy };
};

describe("useSwitchLlmProfile", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    SettingsService.invalidateCache();
  });

  it("invalidates the settings cache on the home-page activate path (conversationId === null)", async () => {
    vi.mocked(AgentServerConversationService.switchProfile).mockResolvedValue(
      undefined as never,
    );
    const invalidateCacheSpy = vi.spyOn(SettingsService, "invalidateCache");

    const { result, invalidateQueriesSpy } = renderSwitchHook();

    result.current.mutate({ conversationId: null, profileName: "my-profile" });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(AgentServerConversationService.switchProfile).toHaveBeenCalledWith(
      null,
      "my-profile",
    );
    // The stale settings cache must be cleared so conversation-start uses the
    // newly activated profile's LLM (the core of bug #640).
    expect(invalidateCacheSpy).toHaveBeenCalled();
    expect(invalidateQueriesSpy).toHaveBeenCalledWith({
      queryKey: LLM_PROFILES_QUERY_KEYS.all,
    });
    expect(invalidateQueriesSpy).toHaveBeenCalledWith({
      queryKey: SETTINGS_QUERY_KEYS.personal(),
    });

    invalidateCacheSpy.mockRestore();
  });

  it("does not touch the settings cache for the per-conversation switch path", async () => {
    vi.mocked(AgentServerConversationService.switchProfile).mockResolvedValue(
      undefined as never,
    );
    const invalidateCacheSpy = vi.spyOn(SettingsService, "invalidateCache");

    const { result, invalidateQueriesSpy } = renderSwitchHook();

    result.current.mutate({
      conversationId: "conv-1",
      profileName: "my-profile",
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(invalidateCacheSpy).not.toHaveBeenCalled();
    expect(invalidateQueriesSpy).toHaveBeenCalledWith({
      queryKey: LLM_PROFILES_QUERY_KEYS.all,
    });
    expect(invalidateQueriesSpy).toHaveBeenCalledWith({
      queryKey: ["user", "conversation", "conv-1"],
    });
    expect(invalidateQueriesSpy).not.toHaveBeenCalledWith({
      queryKey: SETTINGS_QUERY_KEYS.personal(),
    });

    invalidateCacheSpy.mockRestore();
  });

  it("does not invalidate the settings cache when the switch fails", async () => {
    vi.mocked(AgentServerConversationService.switchProfile).mockRejectedValue(
      new Error("boom"),
    );
    const invalidateCacheSpy = vi.spyOn(SettingsService, "invalidateCache");

    const { result } = renderSwitchHook();

    result.current.mutate({ conversationId: null, profileName: "my-profile" });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(invalidateCacheSpy).not.toHaveBeenCalled();

    invalidateCacheSpy.mockRestore();
  });

  // Errors surface via a tailored onError (not the global mutation toast) so a
  // failed switch keeps the specific "Switched to {name} failed" message
  // (#1571 review).
  it("shows the tailored switch-failed message when the error carries no server detail", async () => {
    // An empty-message Error extracts to "" (see retrieveAxiosErrorMessage),
    // so the tailored fallback is what actually renders.
    vi.mocked(AgentServerConversationService.switchProfile).mockRejectedValue(
      new Error(),
    );

    const { result } = renderSwitchHook();
    result.current.mutate({ conversationId: null, profileName: "Smart" });

    await waitFor(() => expect(result.current.isError).toBe(true));
    // The global i18n mock returns the raw key (see vitest.setup.ts).
    expect(displayErrorToast).toHaveBeenCalledWith("MODEL$SWITCH_FAILED");
  });

  it("prefers the server-provided error detail over the tailored fallback", async () => {
    const axiosError = new AxiosError(
      "Request failed",
      "500",
      undefined,
      undefined,
      {
        status: 404,
        statusText: "Not Found",
        headers: new AxiosHeaders(),
        config: { headers: new AxiosHeaders() },
        data: { message: "LLM profile 'gpt-5' not found" },
      },
    );
    vi.mocked(AgentServerConversationService.switchProfile).mockRejectedValue(
      axiosError,
    );

    const { result } = renderSwitchHook();
    result.current.mutate({ conversationId: null, profileName: "Smart" });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(displayErrorToast).toHaveBeenCalledWith(
      "LLM profile 'gpt-5' not found",
    );
  });

  // The following behaviors run in the mutation-level onSuccess (not
  // mutate-scoped callbacks) so they survive the switcher menu unmounting on
  // select (#1571).
  it("records the inline switch message anchored to the last renderable event", async () => {
    vi.mocked(AgentServerConversationService.switchProfile).mockResolvedValue(
      undefined as never,
    );

    const { result } = renderSwitchHook();
    result.current.mutate({ conversationId: "conv-1", profileName: "Smart" });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(recordModelSwitchMessage).toHaveBeenCalledWith(
      "conv-1",
      "Smart",
      "evt-9",
    );
  });

  it("stamps the switched-to profile onto the conversation metadata, preserving repo/workspace (#1082)", async () => {
    vi.mocked(AgentServerConversationService.switchProfile).mockResolvedValue(
      undefined as never,
    );
    // Repo metadata persisted at creation must survive the merge.
    setStoredConversationMetadata("conv-1", {
      selected_repository: "octocat/hello-world",
      selected_branch: "main",
      git_provider: "github",
    });

    const { result } = renderSwitchHook();
    result.current.mutate({
      conversationId: "conv-1",
      profileName: "claude-sonnet-4.6",
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(getStoredConversationMetadata("conv-1")).toEqual({
      selected_repository: "octocat/hello-world",
      selected_branch: "main",
      git_provider: "github",
      selected_workspace: null,
      active_profile: "claude-sonnet-4.6",
      plugins: null,
    });
  });

  it("preserves the conversation's attached plugins across a profile switch", async () => {
    vi.mocked(AgentServerConversationService.switchProfile).mockResolvedValue(
      undefined as never,
    );
    setStoredConversationMetadata("conv-1", {
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      plugins: [
        { source: "github:acme/city-weather", ref: null, repo_path: null },
      ],
    });

    const { result } = renderSwitchHook();
    result.current.mutate({
      conversationId: "conv-1",
      profileName: "claude-sonnet-4.6",
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(getStoredConversationMetadata("conv-1")?.plugins).toEqual([
      { source: "github:acme/city-weather", ref: null, repo_path: null },
    ]);
  });

  it("does not stamp metadata for the home-page activate path (conversationId === null)", async () => {
    vi.mocked(AgentServerConversationService.switchProfile).mockResolvedValue(
      undefined as never,
    );

    const { result } = renderSwitchHook();
    result.current.mutate({ conversationId: null, profileName: "Smart" });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(getStoredConversationMetadata("conv-1")).toBeNull();
    expect(recordModelSwitchMessage).not.toHaveBeenCalled();
  });

  it("does not stamp metadata when the switch fails", async () => {
    vi.mocked(AgentServerConversationService.switchProfile).mockRejectedValue(
      new Error("boom"),
    );

    const { result } = renderSwitchHook();
    result.current.mutate({ conversationId: "conv-1", profileName: "Smart" });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(getStoredConversationMetadata("conv-1")).toBeNull();
    expect(recordModelSwitchMessage).not.toHaveBeenCalled();
  });
});
