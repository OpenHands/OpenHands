import { renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";

const useOptionalConversationIdMock = vi.fn();
const useActiveConversationMock = vi.fn();
const useLlmProfilesMock = vi.fn();
const switchAndLog = vi.fn();
const setHomeLlmProfile = vi.fn();

const freshModelStoreState = () => ({
  activeProfileByConversation: {} as Record<string, string>,
  homeLlmProfile: null as string | null,
  homeManualModelPick: false,
  setHomeLlmProfile,
  setHomeManualModelPick: (value: boolean) => {
    modelStoreState.homeManualModelPick = value;
  },
});

let modelStoreState: ReturnType<typeof freshModelStoreState> =
  freshModelStoreState();

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => useOptionalConversationIdMock(),
}));
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));
vi.mock("#/hooks/query/use-llm-profiles", () => ({
  useLlmProfiles: () => useLlmProfilesMock(),
}));
vi.mock("#/hooks/mutation/use-switch-llm-profile-and-log", () => ({
  useSwitchLlmProfileAndLog: () => ({ switchAndLog, isPending: false }),
}));
vi.mock("#/stores/model-store", () => ({
  // `selectProfile` writes via getState() (the synchronous, non-reactive
  // path) so the flag is set before the async switch resolves; the hook
  // subscription form is still used for reactive reads elsewhere.
  useModelStore: Object.assign(
    (selector: (s: typeof modelStoreState) => unknown) =>
      selector(modelStoreState),
    { getState: () => modelStoreState },
  ),
}));
// The org-permission check reads the active backend from a context this
// wrapper-less harness doesn't provide; drive it per test instead.
const useCanManageOrgProfilesMock = vi.fn();
vi.mock("#/hooks/use-can-manage-org-profiles", () => ({
  useCanManageOrgProfiles: () => useCanManageOrgProfilesMock(),
}));

const useAgentProfilesMock = vi.fn();
vi.mock("#/hooks/query/use-agent-profiles", () => ({
  useAgentProfiles: () => useAgentProfilesMock(),
}));
const backendKind = vi.hoisted(() => ({ current: "local" }));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "b1", kind: backendKind.current },
  }),
}));
const updateAgentProfileMock = vi.hoisted(() => vi.fn());
vi.mock("#/hooks/mutation/sync-agent-profiles", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/hooks/mutation/sync-agent-profiles")
  >()),
  updateAgentProfile: updateAgentProfileMock,
}));

// eslint-disable-next-line import/first
import { useChatInputLlmProfileState } from "#/hooks/use-chat-input-llm-profile-state";

// The hook reads the switch's pending state via `useIsMutating`, which needs a
// QueryClient in context (the switch mutation itself is mocked out above).
const renderState = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return renderHook(() => useChatInputLlmProfileState(), {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
};

const PROFILES = [
  { name: "Fast", model: "gpt-4o-mini", base_url: null, api_key_set: true },
  { name: "Smart", model: "claude-opus", base_url: null, api_key_set: true },
];

describe("useChatInputLlmProfileState", () => {
  beforeEach(() => {
    switchAndLog.mockReset();
    setHomeLlmProfile.mockReset();
    modelStoreState = freshModelStoreState();
    useOptionalConversationIdMock.mockReturnValue({ conversationId: "c1" });
    useActiveConversationMock.mockReturnValue({ data: undefined });
    useLlmProfilesMock.mockReturnValue({
      data: { profiles: PROFILES, active_profile: "Fast" },
      isLoading: false,
    });
    useCanManageOrgProfilesMock.mockReset();
    // Local backends always own their profiles.
    useCanManageOrgProfilesMock.mockReturnValue(true);
    useAgentProfilesMock.mockReturnValue({ data: undefined });
    updateAgentProfileMock.mockReset();
    backendKind.current = "local";
  });

  const activeAgent = (name: string, llmProfileRef: string) =>
    useAgentProfilesMock.mockReturnValue({
      data: {
        profiles: [
          {
            id: `id-${name}`,
            name,
            agent_kind: "openhands",
            llm_profile_ref: llmProfileRef,
          },
        ],
        active_agent_profile_id: `id-${name}`,
      },
    });

  it("shows the active agent profile's own LLM on the home page", () => {
    useOptionalConversationIdMock.mockReturnValue({ conversationId: null });
    activeAgent("reviewer", "Smart");

    const { result } = renderState();

    expect(result.current.currentProfileName).toBe("Smart");
  });

  it("saves a home pick as a named agent profile's own LLM", () => {
    useOptionalConversationIdMock.mockReturnValue({ conversationId: null });
    activeAgent("reviewer", "Smart");

    const { result } = renderState();
    result.current.selectProfile("Fast");

    expect(switchAndLog).not.toHaveBeenCalled();
    const [, name, change] = updateAgentProfileMock.mock.calls[0];
    expect(name).toBe("reviewer");
    expect(
      change({ agent_kind: "openhands", llm_profile_ref: "Smart" }),
    ).toEqual({ llm_profile_ref: "Fast" });
  });

  it("activates a home pick account-wide while default is active", () => {
    useOptionalConversationIdMock.mockReturnValue({ conversationId: null });
    activeAgent("default", "Fast");

    const { result } = renderState();
    result.current.selectProfile("Smart");

    expect(updateAgentProfileMock).not.toHaveBeenCalled();
    expect(switchAndLog).toHaveBeenCalledWith(null, "Smart");
  });

  it("treats a cloud profile named default like any other profile", () => {
    backendKind.current = "cloud";
    useOptionalConversationIdMock.mockReturnValue({ conversationId: null });
    activeAgent("default", "Smart");

    const { result } = renderState();
    expect(result.current.currentProfileName).toBe("Smart");
    result.current.selectProfile("Fast");

    expect(switchAndLog).not.toHaveBeenCalled();
    expect(updateAgentProfileMock.mock.calls[0][1]).toBe("default");
  });

  it("prefers the profile stamped on the conversation over a model match", () => {
    useActiveConversationMock.mockReturnValue({
      data: { active_profile: "Smart", llm_model: "gpt-4o-mini" },
    });
    const { result } = renderState();
    // model "gpt-4o-mini" would match "Fast", but the stamped profile wins.
    expect(result.current.currentProfileName).toBe("Smart");
  });

  it("falls back to the profile whose model matches the running llm_model", () => {
    useActiveConversationMock.mockReturnValue({
      data: { llm_model: "claude-opus" },
    });
    const { result } = renderState();
    expect(result.current.currentProfileName).toBe("Smart");
  });

  it("falls back to the account active_profile when there is no conversation model", () => {
    useActiveConversationMock.mockReturnValue({ data: { llm_model: null } });
    const { result } = renderState();
    expect(result.current.currentProfileName).toBe("Fast");
  });

  it("prefers the optimistic just-switched profile", () => {
    modelStoreState = {
      ...freshModelStoreState(),
      activeProfileByConversation: { c1: "Smart" },
    };
    useActiveConversationMock.mockReturnValue({
      data: { active_profile: "Fast", llm_model: "gpt-4o-mini" },
    });
    const { result } = renderState();
    expect(result.current.currentProfileName).toBe("Smart");
  });

  it("ignores a stamped profile that is no longer in the list", () => {
    useActiveConversationMock.mockReturnValue({
      data: { active_profile: "Deleted", llm_model: "claude-opus" },
    });
    const { result } = renderState();
    // Deleted is not in PROFILES → fall through to the model match.
    expect(result.current.currentProfileName).toBe("Smart");
  });

  // A conversation the server returns without agent LLM metadata reports no
  // model at all, so the picker must still name a profile rather than fall
  // back to the "Select a model" placeholder (#16263).
  it("names the account active profile when the conversation has no model", () => {
    useActiveConversationMock.mockReturnValue({
      data: { active_profile: null, llm_model: null },
    });
    const { result } = renderState();
    expect(result.current.currentProfileName).toBe("Fast");
  });

  it("live-switches a different profile against the conversation id", () => {
    useActiveConversationMock.mockReturnValue({
      data: { active_profile: "Fast", llm_model: "gpt-4o-mini" },
    });
    const { result } = renderState();
    result.current.selectProfile("Smart");
    expect(switchAndLog).toHaveBeenCalledWith("c1", "Smart");
  });

  it("does not switch when the current profile is re-selected", () => {
    useActiveConversationMock.mockReturnValue({
      data: { active_profile: "Fast", llm_model: "gpt-4o-mini" },
    });
    const { result } = renderState();
    result.current.selectProfile("Fast");
    expect(switchAndLog).not.toHaveBeenCalled();
  });

  it("activates the picked profile account-wide from the home page", () => {
    useOptionalConversationIdMock.mockReturnValue({ conversationId: null });

    const { result } = renderState();
    result.current.selectProfile("Smart");

    // No conversation → the switch activates the profile globally so the next
    // conversation launches with it.
    expect(switchAndLog).toHaveBeenCalledWith(null, "Smart");
  });

  it("holds a member's home pick for the next launch instead of activating it", () => {
    // A home activate hits the org-gated endpoint, which a cloud member could
    // only 403 on; the pick is sent with the next home launch instead.
    useOptionalConversationIdMock.mockReturnValue({ conversationId: null });
    useCanManageOrgProfilesMock.mockReturnValue(false);

    const { result } = renderState();
    result.current.selectProfile("Smart");

    expect(result.current.canSwitchProfile).toBe(true);
    expect(setHomeLlmProfile).toHaveBeenCalledWith("Smart");
    expect(switchAndLog).not.toHaveBeenCalled();
  });

  it("holds a member's home pick instead of editing the org's agent profile", () => {
    backendKind.current = "cloud";
    useOptionalConversationIdMock.mockReturnValue({ conversationId: null });
    useCanManageOrgProfilesMock.mockReturnValue(false);
    activeAgent("reviewer", "Fast");

    const { result } = renderState();
    result.current.selectProfile("Smart");

    expect(setHomeLlmProfile).toHaveBeenCalledWith("Smart");
    expect(updateAgentProfileMock).not.toHaveBeenCalled();
  });

  it("marks a home pick so the next launch suppresses route-at-start (every pick path)", () => {
    // The flag is set at the UI chokepoint (selectProfile) so every home pick
    // path is covered — a cloud member's held pick, an agent-profile-owned
    // LLM pick, and a global activate. `useCreateConversation` reads-and-
    // clears it.
    useOptionalConversationIdMock.mockReturnValue({ conversationId: null });

    // 1. Cloud member (held pick) — can't manage org profiles.
    useCanManageOrgProfilesMock.mockReturnValue(false);
    const { result: memberResult } = renderState();
    memberResult.current.selectProfile("Smart");
    expect(modelStoreState.homeManualModelPick).toBe(true);

    // 2. Admin with an agent-profile-owned LLM — edits the agent profile.
    modelStoreState.homeManualModelPick = false;
    useCanManageOrgProfilesMock.mockReturnValue(true);
    backendKind.current = "local";
    activeAgent("reviewer", "Fast");
    const { result: adminResult } = renderState();
    adminResult.current.selectProfile("Smart");
    expect(modelStoreState.homeManualModelPick).toBe(true);

    // 3. Global activate — default active, no agent-profile ownership.
    modelStoreState.homeManualModelPick = false;
    const { result: activateResult } = renderState();
    activateResult.current.selectProfile("Smart");
    expect(modelStoreState.homeManualModelPick).toBe(true);
  });

  it("does not mark a pick made inside an existing conversation", () => {
    useOptionalConversationIdMock.mockReturnValue({ conversationId: "c1" });

    const { result } = renderState();
    result.current.selectProfile("Smart");

    expect(modelStoreState.homeManualModelPick).toBe(false);
    expect(switchAndLog).toHaveBeenCalledWith("c1", "Smart");
  });

  it.each([
    {
      description: "names a member's home pick on the home page",
      conversationId: null,
      pick: "Smart",
      expected: "Smart",
    },
    {
      description: "names a member's home pick while its start task runs",
      conversationId: "task-abc",
      pick: "Smart",
      expected: "Smart",
    },
    {
      description:
        "falls back to the active profile when the picked one is gone",
      conversationId: null,
      pick: "Deleted",
      expected: "Fast",
    },
  ])("$description", ({ conversationId, pick, expected }) => {
    useOptionalConversationIdMock.mockReturnValue({ conversationId });
    modelStoreState = { ...modelStoreState, homeLlmProfile: pick };

    const { result } = renderState();

    expect(result.current.currentProfileName).toBe(expected);
  });

  it("keeps the picker read-only while a cloud start task is provisioning", () => {
    // `task-…` is not a real conversation id yet, so /switch_profile would 404.
    useOptionalConversationIdMock.mockReturnValue({
      conversationId: "task-abc",
    });

    const { result } = renderState();
    result.current.selectProfile("Smart");

    expect(result.current.canSwitchProfile).toBe(false);
    expect(switchAndLog).not.toHaveBeenCalled();
  });
});
