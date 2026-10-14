import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { useLlmConfigured } from "#/hooks/use-llm-configured";

const useSettingsMock = vi.fn();
const useConfigMock = vi.fn();
const useLlmProfilesMock = vi.fn();
const useActiveBackendMock = vi.fn();
const useActiveAgentProfileMock = vi.fn();

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => useSettingsMock(),
}));
vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => useConfigMock(),
}));
vi.mock("#/hooks/query/use-llm-profiles", () => ({
  useLlmProfiles: () => useLlmProfilesMock(),
}));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => useActiveBackendMock(),
}));
vi.mock("#/hooks/use-active-agent-profile", () => ({
  useActiveAgentProfile: () => {
    const value = useActiveAgentProfileMock();
    return {
      isError: false,
      agentKind: value.activeProfile?.agent_kind,
      ...value,
    };
  },
}));

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("useLlmConfigured (local, agent-profile-driven)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useConfigMock.mockReturnValue({ data: { feature_flags: {} } });
    useActiveBackendMock.mockReturnValue({
      backend: { kind: "local", id: "b1" },
      orgId: null,
    });
    useSettingsMock.mockReturnValue({
      data: {
        agent_settings: { agent_kind: "openhands" },
        llm_api_key_set: false,
      },
    });
  });

  const llmProfiles = (
    active: string,
    entries: { name: string; api_key_set: boolean }[],
  ) => ({
    data: {
      active_profile: active,
      profiles: entries.map((e) => ({
        name: e.name,
        model: "m",
        api_key_set: e.api_key_set,
      })),
    },
  });

  it("is configured when the active LLM profile, which a home launch runs, has a key", () => {
    // Discriminating case: the agent profile references "sonnet", which has
    // no key, but a local home launch runs the active LLM profile ("default").
    useLlmProfilesMock.mockReturnValue(
      llmProfiles("default", [
        { name: "default", api_key_set: true },
        { name: "sonnet", api_key_set: false },
      ]),
    );
    useActiveAgentProfileMock.mockReturnValue({
      activeProfile: {
        agent_kind: "openhands",
        llm_profile_ref: "sonnet",
        name: "MyOH",
      },
    });

    const { result } = renderHook(() => useLlmConfigured(), { wrapper });
    expect(result.current.isConfigured).toBe(true);
  });
  it("is NOT configured when the active LLM profile has no key, even if the agent profile's ref has one", () => {
    useLlmProfilesMock.mockReturnValue(
      llmProfiles("default", [
        { name: "default", api_key_set: false },
        { name: "sonnet", api_key_set: true },
      ]),
    );
    useActiveAgentProfileMock.mockReturnValue({
      activeProfile: {
        agent_kind: "openhands",
        llm_profile_ref: "sonnet",
        name: "MyOH",
      },
    });

    const { result } = renderHook(() => useLlmConfigured(), { wrapper });
    expect(result.current.isConfigured).toBe(false);
  });
  it("checks the active LLM profile when the agent profile's ref is stale", () => {
    // The agent profile references "ghost", which no longer exists; the launch
    // runs the active LLM profile ("default", which HAS a key) instead.
    useLlmProfilesMock.mockReturnValue(
      llmProfiles("default", [{ name: "default", api_key_set: true }]),
    );
    useActiveAgentProfileMock.mockReturnValue({
      activeProfile: {
        agent_kind: "openhands",
        llm_profile_ref: "ghost",
        name: "MyOH",
      },
    });

    const { result } = renderHook(() => useLlmConfigured(), { wrapper });
    expect(result.current.isConfigured).toBe(true);
  });
  it.each([
    [true, true],
    [false, false],
  ])(
    "falls back to the agent profile's ref when no LLM profile is active (key set: %s)",
    (keySet, expected) => {
      useLlmProfilesMock.mockReturnValue({
        data: {
          active_profile: null,
          profiles: [{ name: "B", model: "m", api_key_set: keySet }],
        },
      });
      useActiveAgentProfileMock.mockReturnValue({
        activeProfile: {
          agent_kind: "openhands",
          llm_profile_ref: "B",
          name: "default",
        },
      });

      const { result } = renderHook(() => useLlmConfigured(), { wrapper });
      expect(result.current.isConfigured).toBe(expected);
    },
  );
  it("is configured for an ACP agent profile regardless of LLM keys", () => {
    useLlmProfilesMock.mockReturnValue(
      llmProfiles("default", [{ name: "default", api_key_set: false }]),
    );
    useActiveAgentProfileMock.mockReturnValue({
      activeProfile: {
        agent_kind: "acp",
        llm_profile_ref: null,
        name: "MyClaude",
      },
    });

    const { result } = renderHook(() => useLlmConfigured(), { wrapper });
    expect(result.current.isConfigured).toBe(true);
  });

  it("stays indeterminate when the profile list failed with nothing cached", () => {
    useSettingsMock.mockReturnValue({
      data: {
        agent_settings: { agent_kind: "acp" },
        llm_api_key_set: false,
      },
    });
    useLlmProfilesMock.mockReturnValue(
      llmProfiles("default", [{ name: "default", api_key_set: false }]),
    );
    useActiveAgentProfileMock.mockReturnValue({
      activeProfile: null,
      agentKind: undefined,
      isLoading: false,
      isError: true,
    });

    const { result } = renderHook(() => useLlmConfigured(), { wrapper });
    expect(result.current.isLoading).toBe(true);
    expect(result.current.isConfigured).toBe(false);
  });
});
