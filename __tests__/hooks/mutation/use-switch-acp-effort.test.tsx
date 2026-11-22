import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { useSwitchAcpEffort } from "#/hooks/mutation/use-switch-acp-effort";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({ default: { switchAcpEffort: vi.fn() } }),
);

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  default: {
    listProfiles: vi.fn(),
    getProfile: vi.fn(),
    saveProfile: vi.fn(),
  },
}));

vi.mock("#/api/settings-service/settings-service.api", () => ({
  default: { saveSettings: vi.fn(), invalidateCache: vi.fn() },
}));

const renderSwitchHook = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const { result } = renderHook(() => useSwitchAcpEffort(), {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
  return result;
};

const ACTIVE_CLAUDE_PROFILE = {
  id: "id-claude",
  name: "claude",
  revision: 3,
  agent_kind: "acp",
  acp_server: "claude-code",
  acp_model: "opus[1m]",
  acp_effort: "high",
};

describe("useSwitchAcpEffort", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [{ id: "id-claude", name: "claude", agent_kind: "acp" }],
      active_agent_profile_id: "id-claude",
    } as never);
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue({
      profile: ACTIVE_CLAUDE_PROFILE,
    } as never);
  });

  it("switches a conversation's effort live without touching the profile", async () => {
    const result = renderSwitchHook();

    result.current.mutate({ conversationId: "conv-1", effort: "low" });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(AgentServerConversationService.switchAcpEffort).toHaveBeenCalledWith(
      "conv-1",
      "low",
    );
    expect(AgentProfilesService.saveProfile).not.toHaveBeenCalled();
  });

  it.each([
    ["low", "low"],
    [null, null],
  ])(
    "saves effort %s on the active ACP profile from the home page",
    async (effort, saved) => {
      const result = renderSwitchHook();

      result.current.mutate({ conversationId: null, effort });

      await waitFor(() => expect(result.current.isSuccess).toBe(true));
      expect(AgentProfilesService.saveProfile).toHaveBeenCalledWith("claude", {
        agent_kind: "acp",
        acp_server: "claude-code",
        acp_model: "opus[1m]",
        acp_effort: saved,
      });
      expect(SettingsService.saveSettings).not.toHaveBeenCalled();
    },
  );

  it("falls back to agent settings when no ACP profile is active", async () => {
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [],
      active_agent_profile_id: null,
    } as never);
    const result = renderSwitchHook();

    result.current.mutate({ conversationId: null, effort: "max" });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(SettingsService.saveSettings).toHaveBeenCalledWith({
      agent_settings_diff: { acp_effort: "max" },
    });
  });
});
