import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useActiveAgentProfile } from "#/hooks/use-active-agent-profile";
import { useAcpModelContext } from "#/hooks/use-acp-model-context";
import { resolveConversationLaunch } from "#/hooks/mutation/resolve-conversation-launch";
import { AGENT_PROFILES_QUERY_KEYS } from "#/hooks/query/query-keys";

const { listProfilesMock, settingsKind, backendKind } = vi.hoisted(() => ({
  listProfilesMock: vi.fn(),
  settingsKind: { value: "openhands" as "openhands" | "acp" },
  backendKind: { value: "local" as "local" | "cloud" },
}));

vi.mock("#/api/agent-profiles-service/agent-profiles-service.api", () => ({
  __esModule: true,
  default: { listProfiles: listProfilesMock, getProfile: vi.fn() },
  WELL_KNOWN_DEFAULT_AGENT_PROFILE_NAME: "default",
}));
vi.mock("#/api/profiles-service/profiles-service.api", () => ({
  __esModule: true,
  default: {
    listProfiles: vi.fn().mockResolvedValue({
      profiles: [{ name: "mock-llm" }],
      active_profile: "mock-llm",
    }),
  },
}));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "local-1", kind: backendKind.value },
    orgId: null,
  }),
}));
vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => ({
    data: { agent_settings: { agent_kind: settingsKind.value } },
  }),
}));
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: undefined }),
}));

const OPENHANDS = {
  id: "oh-1",
  name: "coder",
  agent_kind: "openhands",
  revision: 1,
  llm_profile_ref: "mock-llm",
  mcp_server_refs: null,
};
const ACP = {
  id: "acp-1",
  name: "claude",
  agent_kind: "acp",
  revision: 1,
  llm_profile_ref: null,
  mcp_server_refs: null,
};
const listWithActive = (id: string) => ({
  profiles: [OPENHANDS, ACP],
  active_agent_profile_id: id,
});

function renderActiveAgent(queryClient: QueryClient) {
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  return renderHook(
    () => ({ active: useActiveAgentProfile(), acp: useAcpModelContext() }),
    { wrapper },
  );
}

const launch = (queryClient: QueryClient) =>
  resolveConversationLaunch({
    queryClient,
    backendId: "local-1",
    orgId: null,
    isCloud: backendKind.value === "cloud",
  });

describe("active agent profile after a failed refetch", () => {
  afterEach(() => {
    listProfilesMock.mockReset();
    settingsKind.value = "openhands";
    backendKind.value = "local";
  });

  it("keeps cached data across a failed refetch, and the launch agrees", async () => {
    const queryClient = new QueryClient();
    listProfilesMock.mockResolvedValue(listWithActive("oh-1"));
    const { result } = renderActiveAgent(queryClient);
    await waitFor(() =>
      expect(result.current.active.activeProfile?.id).toBe("oh-1"),
    );

    // Activating ACP invalidates the list, and that refetch fails.
    settingsKind.value = "acp";
    listProfilesMock.mockRejectedValue(new Error("network down"));
    await act(() =>
      queryClient
        .refetchQueries({ queryKey: AGENT_PROFILES_QUERY_KEYS.all })
        .catch(() => undefined),
    );

    expect(result.current.active.isError).toBe(false);
    expect(result.current.active.agentKind).toBe("openhands");
    expect(result.current.acp.isHomeAcp).toBe(false);
    await expect(launch(queryClient)).resolves.toMatchObject({
      agentProfileId: "oh-1",
      agentProfileKind: "openhands",
    });

    listProfilesMock.mockResolvedValue(listWithActive("acp-1"));
    await act(() =>
      queryClient.refetchQueries({ queryKey: AGENT_PROFILES_QUERY_KEYS.all }),
    );
    expect(result.current.active.agentKind).toBe("acp");
    expect(result.current.acp.isHomeAcp).toBe(true);
    await expect(launch(queryClient)).resolves.toMatchObject({
      agentProfileId: "acp-1",
      agentProfileKind: "acp",
    });
  });

  it("does not fall back to global settings when nothing is cached", async () => {
    const queryClient = new QueryClient();
    settingsKind.value = "acp";
    listProfilesMock.mockRejectedValue(new Error("network down"));
    const { result } = renderActiveAgent(queryClient);

    await waitFor(() => expect(result.current.active.isError).toBe(true));
    expect(result.current.active.agentKind).toBeUndefined();
    expect(result.current.acp.isHomeAcp).toBe(false);
    await expect(launch(queryClient)).rejects.toThrow("network down");
  });

  it("leaves the agent kind unknown when a local server has no active profile", async () => {
    const queryClient = new QueryClient();
    settingsKind.value = "acp";
    listProfilesMock.mockResolvedValue({
      profiles: [OPENHANDS],
      active_agent_profile_id: null,
    });
    const { result } = renderActiveAgent(queryClient);

    await waitFor(() => expect(result.current.active.isLoading).toBe(false));
    expect(result.current.active.agentKind).toBeUndefined();
    expect(result.current.acp.isHomeAcp).toBe(false);
    await expect(launch(queryClient)).rejects.toThrow();
  });

  it("uses global settings on Cloud when no profile is active", async () => {
    const queryClient = new QueryClient();
    backendKind.value = "cloud";
    settingsKind.value = "acp";
    listProfilesMock.mockResolvedValue({
      profiles: [OPENHANDS],
      active_agent_profile_id: null,
    });
    const { result } = renderActiveAgent(queryClient);

    await waitFor(() => expect(result.current.active.isLoading).toBe(false));
    expect(result.current.active.agentKind).toBe("acp");
    expect(result.current.acp.isHomeAcp).toBe(true);
    await expect(launch(queryClient)).resolves.not.toHaveProperty(
      "agentProfileId",
    );
  });
});
