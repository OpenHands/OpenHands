import { renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";

const useOptionalConversationIdMock = vi.fn();
const useActiveConversationMock = vi.fn();
const useLlmProfilesMock = vi.fn();
const switchAndLog = vi.fn();

let modelStoreState: { activeProfileByConversation: Record<string, string> } = {
  activeProfileByConversation: {},
};

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
  useModelStore: (selector: (s: typeof modelStoreState) => unknown) =>
    selector(modelStoreState),
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
    modelStoreState = { activeProfileByConversation: {} };
    useOptionalConversationIdMock.mockReturnValue({ conversationId: "c1" });
    useActiveConversationMock.mockReturnValue({ data: undefined });
    useLlmProfilesMock.mockReturnValue({
      data: { profiles: PROFILES, active_profile: "Fast" },
      isLoading: false,
    });
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
    modelStoreState = { activeProfileByConversation: { c1: "Smart" } };
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
});
