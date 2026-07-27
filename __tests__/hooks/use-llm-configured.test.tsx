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
  useActiveAgentProfile: () => useActiveAgentProfileMock(),
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

  it("is configured when the LLM profile the AGENT profile references has a key", () => {
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
    expect(result.current.isConfigured).toBe(true);
  });

  it("is NOT configured when the referenced profile has no key, even if the active LLM profile has one", () => {
    // Discriminating case: the standalone active LLM profile ("default") HAS a
    // key, but the active agent profile references "sonnet", which does not.
    // Conversations launch from the agent profile, so the ref is what matters.
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
    expect(result.current.isConfigured).toBe(false);
  });

  it("falls back to the active LLM profile when the ref is stale (matches the launch fallback)", () => {
    // The agent profile references "ghost", which no longer exists in the list
    // (deleted profile / seed ref pointing at a named profile that was never
    // created). The launch path drops a stale-ref profile launch to an
    // agent_settings launch on the active LLM ("default", which HAS a key), so
    // this hook must report configured too — otherwise the composer is
    // spuriously disabled even though launch succeeds (VascoSch92 review #1571).
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
});
