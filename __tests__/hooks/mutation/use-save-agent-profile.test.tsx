import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AgentProfilesService from "#/api/agent-profiles-service/agent-profiles-service.api";
import SettingsService from "#/api/settings-service/settings-service.api";
import { useSaveAgentProfile } from "#/hooks/mutation/use-save-agent-profile";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  __esModule: true,
  default: {
    saveProfile: vi.fn(),
    listProfiles: vi.fn(),
    getProfile: vi.fn(),
  },
}));
vi.mock("#/api/settings-service/settings-service.api", () => ({
  __esModule: true,
  default: { saveSettings: vi.fn(), invalidateCache: vi.fn() },
}));

const useActiveBackendMock = vi.hoisted(() => vi.fn());
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => useActiveBackendMock(),
}));

// The sync chain re-reads the registry before each call so a backend or org
// switch mid-flight can abandon the write; drive it from the same values the
// context mock reports.
const getActiveBackendMock = vi.hoisted(() => vi.fn());
vi.mock("#/api/backend-registry/active-store", () => ({
  getActiveBackend: () => getActiveBackendMock(),
}));
vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: vi.fn(),
}));

const useBackend = (kind: "cloud" | "local") => {
  const active = { backend: { id: "bk-1", kind }, orgId: null };
  useActiveBackendMock.mockReturnValue(active);
  getActiveBackendMock.mockReturnValue(active);
};

const acpSummary = {
  id: "prof-acp",
  name: "claude",
  agent_kind: "acp",
  revision: 2,
  llm_profile_ref: null,
  mcp_server_refs: null,
};

const openHandsSummary = {
  id: "prof-oh",
  name: "default",
  agent_kind: "openhands",
  revision: 1,
  llm_profile_ref: "llm",
  mcp_server_refs: null,
};

const acpDetail = {
  name: "claude",
  profile: {
    agent_kind: "acp",
    id: "prof-acp",
    name: "claude",
    revision: 3,
    mcp_server_refs: null,
    acp_server: "claude-code",
    acp_model: "claude-x",
    acp_session_mode: null,
    acp_prompt_timeout: 60,
    acp_command: null,
    acp_args: null,
  },
};

const openHandsDetail = {
  name: "default",
  profile: {
    agent_kind: "openhands",
    id: "prof-oh",
    name: "default",
    revision: 1,
    mcp_server_refs: null,
    llm_profile_ref: "llm",
  },
};

const renderSaveHook = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  const { result } = renderHook(() => useSaveAgentProfile(), {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
  return { result };
};

const save = async (input: { name: string; profile: unknown }) => {
  const { result } = renderSaveHook();
  result.current.mutate(input as never);
  await waitFor(() => expect(result.current.isPending).toBe(false));
  return { result };
};

describe("useSaveAgentProfile agent-kind sync (#18883)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useBackend("cloud");
  });

  it("on cloud, saving the active profile re-states its kind in agent_settings", async () => {
    vi.mocked(AgentProfilesService.saveProfile).mockResolvedValue({
      name: "claude",
      message: "saved",
    });
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [acpSummary, openHandsSummary],
      active_agent_profile_id: "prof-acp",
    } as never);
    vi.mocked(AgentProfilesService.getProfile).mockResolvedValue(
      acpDetail as never,
    );

    const { result } = await save({
      name: "claude",
      profile: { agent_kind: "acp", acp_server: "claude-code" },
    });

    expect(result.current.isSuccess).toBe(true);
    expect(AgentProfilesService.saveProfile).toHaveBeenCalledWith(
      "claude",
      expect.objectContaining({ agent_kind: "acp" }),
    );
    expect(SettingsService.saveSettings).toHaveBeenCalledWith({
      agent_settings_diff: {
        agent_kind: "acp",
        acp_server: "claude-code",
        acp_command: [],
        acp_args: [],
        acp_model: "claude-x",
      },
    });
  });

  it("on cloud, saving a non-active profile does not write agent_settings", async () => {
    vi.mocked(AgentProfilesService.saveProfile).mockResolvedValue({
      name: "claude",
      message: "saved",
    });
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [acpSummary, openHandsSummary],
      active_agent_profile_id: "prof-oh",
    } as never);

    await save({
      name: "claude",
      profile: { agent_kind: "acp", acp_server: "claude-code" },
    });

    expect(AgentProfilesService.saveProfile).toHaveBeenCalledOnce();
    expect(AgentProfilesService.getProfile).not.toHaveBeenCalled();
    expect(SettingsService.saveSettings).not.toHaveBeenCalled();
  });

  it("on cloud, a profile without a stable id never counts as active", async () => {
    vi.mocked(AgentProfilesService.saveProfile).mockResolvedValue({
      name: "claude",
      message: "saved",
    });
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [{ ...acpSummary, id: null }],
      active_agent_profile_id: null,
    } as never);

    await save({
      name: "claude",
      profile: { agent_kind: "acp", acp_server: "claude-code" },
    });

    expect(SettingsService.saveSettings).not.toHaveBeenCalled();
  });

  it("on local, saving the active profile does not write agent_settings", async () => {
    useBackend("local");
    vi.mocked(AgentProfilesService.saveProfile).mockResolvedValue({
      name: "claude",
      message: "saved",
    });
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [acpDetail.name === "claude" ? acpSummary : openHandsSummary],
      active_agent_profile_id: "prof-acp",
    } as never);

    await save({
      name: "claude",
      profile: { agent_kind: "acp", acp_server: "claude-code" },
    });

    expect(AgentProfilesService.saveProfile).toHaveBeenCalledOnce();
    expect(AgentProfilesService.listProfiles).not.toHaveBeenCalled();
    expect(SettingsService.saveSettings).not.toHaveBeenCalled();
  });

  it("a failed profile lookup toasts and keeps the save successful", async () => {
    vi.mocked(AgentProfilesService.saveProfile).mockResolvedValue({
      name: "claude",
      message: "saved",
    });
    vi.mocked(AgentProfilesService.listProfiles).mockRejectedValue(
      new Error("list failed"),
    );

    const { result } = await save({
      name: "claude",
      profile: { agent_kind: "acp", acp_server: "claude-code" },
    });

    // The save stands; the lookup failure is surfaced so the stale kind is
    // not silently left in place.
    expect(result.current.isSuccess).toBe(true);
    expect(SettingsService.saveSettings).not.toHaveBeenCalled();
    expect(displayErrorToast).toHaveBeenCalledOnce();
  });

  it("abandons the write when the selection changes while the profile read is pending", async () => {
    vi.mocked(AgentProfilesService.saveProfile).mockResolvedValue({
      name: "claude",
      message: "saved",
    });
    vi.mocked(AgentProfilesService.listProfiles).mockResolvedValue({
      profiles: [acpSummary, openHandsSummary],
      active_agent_profile_id: "prof-acp",
    } as never);

    let resolveDetail: (value: unknown) => void = () => {};
    vi.mocked(AgentProfilesService.getProfile).mockReturnValue(
      new Promise((resolve) => {
        resolveDetail = resolve;
      }) as never,
    );

    const { result } = renderSaveHook();
    result.current.mutate({
      name: "claude",
      profile: { agent_kind: "acp", acp_server: "claude-code" },
    });
    await waitFor(() =>
      expect(AgentProfilesService.getProfile).toHaveBeenCalled(),
    );

    // The user switches org within the cloud backend while the read is pending.
    getActiveBackendMock.mockReturnValue({
      backend: { id: "bk-1", kind: "cloud" },
      orgId: "org-2",
    });
    resolveDetail(acpDetail);

    await waitFor(() => expect(result.current.isPending).toBe(false));

    expect(SettingsService.saveSettings).not.toHaveBeenCalled();
    expect(displayErrorToast).not.toHaveBeenCalled();
  });
});
