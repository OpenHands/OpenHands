import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  getStoredConversationMetadata,
  setStoredConversationMetadata,
} from "#/api/conversation-metadata-store";

const useOptionalConversationIdMock = vi.fn();
const useActiveConversationMock = vi.fn();
const useLlmProfilesMock = vi.fn();
const switchAndLog = vi.fn();
const setHomeLlmProfile = vi.fn();
const clearActiveProfile = vi.fn((id: string) => {
  delete modelStoreState.activeProfileByConversation[id];
});

const freshModelStoreState = () => ({
  activeProfileByConversation: {} as Record<string, string>,
  homeLlmProfile: null as string | null,
  setHomeLlmProfile,
  clearActiveProfile,
});

let modelStoreState: ReturnType<typeof freshModelStoreState> =
  freshModelStoreState();

const useIsMutatingMock = vi.fn().mockReturnValue(0);
vi.mock("@tanstack/react-query", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tanstack/react-query")>();
  return {
    ...actual,
    useIsMutating: (...args: unknown[]) => useIsMutatingMock(...args),
  };
});

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
  // `selectProfile` and the reconciliation effect write via getState()
  // (the synchronous, non-reactive path); the hook subscription form is
  // still used for reactive reads elsewhere.
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
const backendKind = vi.hoisted(() => ({
  id: "b1",
  kind: "local",
  orgId: null as string | null,
}));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: backendKind.id, kind: backendKind.kind },
    orgId: backendKind.orgId,
  }),
}));
const updateAgentProfileMock = vi.hoisted(() => vi.fn());
vi.mock("#/hooks/mutation/sync-agent-profiles", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/hooks/mutation/sync-agent-profiles")
  >()),
  updateAgentProfile: updateAgentProfileMock,
}));
// The instance config a cloud backend reports, mocked at the service: it is
// what tells a self-hosted instance with Agent Profiles off from OpenHands Cloud.
const fetchCloudIntegrationsConfigMock = vi.hoisted(() => vi.fn());
vi.mock("#/api/cloud/settings-service.api", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/api/cloud/settings-service.api")
  >()),
  fetchCloudIntegrationsConfig: fetchCloudIntegrationsConfigMock,
}));

// eslint-disable-next-line import/first
import {
  clearPendingHomeModelPick,
  hasPendingHomeModelPick,
  useChatInputLlmProfileState,
} from "#/hooks/use-chat-input-llm-profile-state";

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
    fetchCloudIntegrationsConfigMock.mockReset();
    // OpenHands Cloud, which runs agent profiles.
    fetchCloudIntegrationsConfigMock.mockResolvedValue({
      deploymentMode: "cloud",
      isAgentProfilesEnabled: false,
    });
    backendKind.id = "b1";
    backendKind.kind = "local";
    backendKind.orgId = null;
    clearPendingHomeModelPick();
    useIsMutatingMock.mockReturnValue(0);
    window.localStorage.clear();
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
    backendKind.kind = "cloud";
    useOptionalConversationIdMock.mockReturnValue({ conversationId: null });
    activeAgent("default", "Smart");

    const { result } = renderState();
    expect(result.current.currentProfileName).toBe("Smart");
    result.current.selectProfile("Fast");

    expect(switchAndLog).not.toHaveBeenCalled();
    expect(updateAgentProfileMock.mock.calls[0][1]).toBe("default");
  });

  describe("on a self-hosted instance with Agent Profiles turned off", () => {
    beforeEach(() => {
      backendKind.kind = "cloud";
      useOptionalConversationIdMock.mockReturnValue({ conversationId: null });
      fetchCloudIntegrationsConfigMock.mockResolvedValue({
        deploymentMode: "self_hosted",
        isAgentProfilesEnabled: false,
      });
      // Activated before the feature was turned off.
      activeAgent("reviewer", "Smart");
    });

    it("names the active LLM profile, not the activated agent profile's", async () => {
      const { result } = renderState();

      await waitFor(() =>
        expect(result.current.currentProfileName).toBe("Fast"),
      );
    });

    it("activates a home pick instead of editing the activated agent profile", async () => {
      const { result } = renderState();
      // Until the instance config arrives the pill still names "Smart".
      await waitFor(() =>
        expect(result.current.currentProfileName).toBe("Fast"),
      );

      result.current.selectProfile("Smart");

      expect(switchAndLog).toHaveBeenCalledWith(null, "Smart");
      expect(updateAgentProfileMock).not.toHaveBeenCalled();
    });
  });

  it("falls back to model-matching when stamped profile's model differs from llm_model", () => {
    useActiveConversationMock.mockReturnValue({
      data: { active_profile: "Smart", llm_model: "gpt-4o-mini" },
    });
    const { result } = renderState();
    // Stamped profile "Smart" has model "claude-opus" != "gpt-4o-mini".
    // Backend llm_model "gpt-4o-mini" matches "Fast", which wins.
    expect(result.current.currentProfileName).toBe("Fast");
  });

  it("preserves exact identity when stamped profile's model matches llm_model", () => {
    useActiveConversationMock.mockReturnValue({
      data: { active_profile: "Fast", llm_model: "gpt-4o-mini" },
    });
    const { result } = renderState();
    expect(result.current.currentProfileName).toBe("Fast");
  });

  it("disambiguates two profiles sharing the same model using the stamp", () => {
    const PROFILES_WITH_SHARED_MODEL = [
      {
        name: "Fast-A",
        model: "gpt-4o-mini",
        base_url: null,
        api_key_set: true,
      },
      {
        name: "Fast-B",
        model: "gpt-4o-mini",
        base_url: null,
        api_key_set: true,
      },
    ];
    useLlmProfilesMock.mockReturnValue({
      data: { profiles: PROFILES_WITH_SHARED_MODEL, active_profile: "Fast-A" },
      isLoading: false,
    });
    useActiveConversationMock.mockReturnValue({
      data: { active_profile: "Fast-B", llm_model: "gpt-4o-mini" },
    });
    const { result } = renderState();
    // Both profiles have model "gpt-4o-mini", stamp "Fast-B" matches model so it wins.
    expect(result.current.currentProfileName).toBe("Fast-B");
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

  it("does not let a stale optimistic profile override conflicting confirmed data outside switch window", () => {
    modelStoreState = {
      ...freshModelStoreState(),
      activeProfileByConversation: { c1: "Smart" },
    };
    useActiveConversationMock.mockReturnValue({
      data: { active_profile: "Fast", llm_model: "gpt-4o-mini" },
    });
    // isSwitching is 0 (outside switch window)
    useIsMutatingMock.mockReturnValue(0);
    const { result } = renderState();
    // Confirmed data has model "gpt-4o-mini" (Fast), conflicting optimistic "Smart" (claude-opus) is ignored
    expect(result.current.currentProfileName).toBe("Fast");
  });

  it("allows in-flight switch optimistic profile to win during the pending switch window", () => {
    modelStoreState = {
      ...freshModelStoreState(),
      activeProfileByConversation: { c1: "Smart" },
    };
    useActiveConversationMock.mockReturnValue({
      data: { active_profile: "Fast", llm_model: "gpt-4o-mini" },
    });
    // In-flight switch is active
    useIsMutatingMock.mockReturnValue(1);
    const { result } = renderState();
    expect(result.current.currentProfileName).toBe("Smart");
  });

  it("does not drop the optimistic entry or rewrite a fresh stamp while the post-switch refetch is still in flight", () => {
    // Race window identified in review: onSuccess has run (mutation done,
    // isSwitching=0; the new stamp + optimistic entry are written), but
    // invalidateQueries' refetch hasn't landed, so the query still serves the
    // stale pre-switch llm_model. The repair effect must not run against that
    // stale model.
    setStoredConversationMetadata("c1", {
      selected_repository: "owner/repo",
      selected_branch: "feat",
      git_provider: null,
      active_profile: "Smart",
      stamped_at: "2026-11-09T00:00:00.000Z",
    });
    modelStoreState = {
      ...freshModelStoreState(),
      activeProfileByConversation: { c1: "Smart" },
    };
    // Stale cached conversation: backend already switched to Smart (claude-opus)
    // but the refetch hasn't landed, so llm_model is still the pre-switch
    // gpt-4o-mini (Fast). isFetching=true signals the refetch is pending.
    useActiveConversationMock.mockReturnValue({
      data: { active_profile: "Smart", llm_model: "gpt-4o-mini" },
      isFetching: true,
    });
    useIsMutatingMock.mockReturnValue(0);
    const { result } = renderState();
    // The optimistic entry (Smart) survives — the pill does NOT snap back to
    // the pre-switch profile (Fast) while the refetch is pending.
    expect(result.current.currentProfileName).toBe("Smart");
    // The just-persisted Smart stamp is NOT rewritten back to Fast, and its
    // stamped_at is NOT refreshed (which would suppress
    // seedModelSwitchesFromHistory's corrective re-stamp on reload).
    const stored = getStoredConversationMetadata("c1");
    expect(stored?.active_profile).toBe("Smart");
    expect(stored?.stamped_at).toBe("2026-11-09T00:00:00.000Z");
    expect(stored?.selected_repository).toBe("owner/repo");
    // The optimistic entry is NOT cleared.
    expect(
      modelStoreState.activeProfileByConversation.c1,
    ).toBe("Smart");
  });

  it("runs the repair once the post-switch refetch lands and the stamp still conflicts", () => {
    // Same setup as above, but now the refetch has landed: isFetching=false and
    // the confirmed llm_model is gpt-4o-mini (Fast) while the browser stamp is
    // still Smart (claude-opus). This is the genuine stale-stamp case the PR
    // targets — the repair should now run.
    setStoredConversationMetadata("c1", {
      selected_repository: "owner/repo",
      selected_branch: "feat",
      git_provider: null,
      active_profile: "Smart",
      stamped_at: "2026-11-09T00:00:00.000Z",
    });
    useActiveConversationMock.mockReturnValue({
      data: { active_profile: "Smart", llm_model: "gpt-4o-mini" },
      isFetching: false,
    });
    useIsMutatingMock.mockReturnValue(0);
    renderState();
    const stored = getStoredConversationMetadata("c1");
    // Confirmed model is gpt-4o-mini (Fast); the Smart stamp is repaired to
    // Fast with a fresh stamped_at, and other metadata fields are preserved.
    expect(stored?.active_profile).toBe("Fast");
    expect(stored?.stamped_at).not.toBe("2026-11-09T00:00:00.000Z");
    expect(stored?.selected_repository).toBe("owner/repo");
  });

  it("repairs conflicting stamp in conversation metadata store without dropping other fields", () => {
    setStoredConversationMetadata("c1", {
      selected_repository: "owner/repo",
      selected_branch: "feat",
      git_provider: null,
      selected_workspace: "/workspace",
      plugins: [],
      local_planning_conversation_id: "plan-123",
      active_profile: "Smart",
    });

    useActiveConversationMock.mockReturnValue({
      data: { active_profile: "Smart", llm_model: "gpt-4o-mini" },
    });

    renderState();

    const stored = getStoredConversationMetadata("c1");
    expect(stored?.active_profile).toBe("Fast");
    expect(stored?.selected_repository).toBe("owner/repo");
    expect(stored?.selected_branch).toBe("feat");
    expect(stored?.selected_workspace).toBe("/workspace");
    expect(stored?.plugins).toEqual([]);
    expect(stored?.local_planning_conversation_id).toBe("plan-123");
  });

  it("issues a switch when re-selecting the backend-consistent profile instead of falsely no-oping", () => {
    // Backend is running claude-opus, but local stamp was Fast (gpt-4o-mini).
    // Now currentProfileName resolves to Smart. Selecting Fast should switch cleanly.
    useActiveConversationMock.mockReturnValue({
      data: { active_profile: "Fast", llm_model: "claude-opus" },
    });
    const { result } = renderState();
    expect(result.current.currentProfileName).toBe("Smart");

    result.current.selectProfile("Fast");
    expect(switchAndLog).toHaveBeenCalledWith("c1", "Fast");
  });

  it("ignores a stamped profile that is no longer in the list", () => {
    useActiveConversationMock.mockReturnValue({
      data: { active_profile: "Deleted", llm_model: "claude-opus" },
    });
    const { result } = renderState();
    // Deleted is not in PROFILES → fall through to the model match.
    expect(result.current.currentProfileName).toBe("Smart");
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
    backendKind.kind = "cloud";
    useOptionalConversationIdMock.mockReturnValue({ conversationId: null });
    useCanManageOrgProfilesMock.mockReturnValue(false);
    activeAgent("reviewer", "Fast");

    const { result } = renderState();
    result.current.selectProfile("Smart");

    expect(setHomeLlmProfile).toHaveBeenCalledWith("Smart");
    expect(updateAgentProfileMock).not.toHaveBeenCalled();
  });

  it("marks a home pick scoped to the current backend/org so the next home launch suppresses route-at-start (every pick path)", () => {
    // The pending pick is recorded at the UI chokepoint (selectProfile) so
    // every home pick path is covered — a cloud member's held pick, an
    // agent-profile-owned LLM pick, and a global activate — and is scoped to
    // (backendId, orgId) so it never leaks to another backend or org.
    useOptionalConversationIdMock.mockReturnValue({ conversationId: null });

    // 1. Cloud member (held pick) — can't manage org profiles.
    useCanManageOrgProfilesMock.mockReturnValue(false);
    const { result: memberResult } = renderState();
    memberResult.current.selectProfile("Smart");
    expect(hasPendingHomeModelPick("b1", null)).toBe(true);
    expect(hasPendingHomeModelPick("b2", null)).toBe(false);
    expect(hasPendingHomeModelPick("b1", "org-other")).toBe(false);

    // 2. Admin with an agent-profile-owned LLM — edits the agent profile.
    clearPendingHomeModelPick();
    useCanManageOrgProfilesMock.mockReturnValue(true);
    backendKind.kind = "local";
    activeAgent("reviewer", "Fast");
    const { result: adminResult } = renderState();
    adminResult.current.selectProfile("Smart");
    expect(hasPendingHomeModelPick("b1", null)).toBe(true);

    // 3. Global activate — default active, no agent-profile ownership.
    clearPendingHomeModelPick();
    const { result: activateResult } = renderState();
    activateResult.current.selectProfile("Smart");
    expect(hasPendingHomeModelPick("b1", null)).toBe(true);
  });

  it("does not mark a pick made inside an existing conversation", () => {
    useOptionalConversationIdMock.mockReturnValue({ conversationId: "c1" });

    const { result } = renderState();
    result.current.selectProfile("Smart");

    expect(hasPendingHomeModelPick("b1", null)).toBe(false);
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
