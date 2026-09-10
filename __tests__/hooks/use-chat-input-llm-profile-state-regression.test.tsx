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
const useCanManageOrgProfilesMock = vi.fn();
vi.mock("#/hooks/use-can-manage-org-profiles", () => ({
  useCanManageOrgProfiles: () => useCanManageOrgProfilesMock(),
}));

// eslint-disable-next-line import/first
import { useChatInputLlmProfileState } from "#/hooks/use-chat-input-llm-profile-state";

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
  { name: "Fable", model: "fable-model", base_url: null, api_key_set: true },
  { name: "Opus", model: "opus-model", base_url: null, api_key_set: true },
];

describe("useChatInputLlmProfileState regression #16851", () => {
  beforeEach(() => {
    switchAndLog.mockReset();
    modelStoreState = { activeProfileByConversation: {} };
    useOptionalConversationIdMock.mockReturnValue({ conversationId: "c1" });
    useActiveConversationMock.mockReturnValue({ data: undefined });
    useLlmProfilesMock.mockReturnValue({
      data: { profiles: PROFILES, active_profile: "Fable" },
      isLoading: false,
    });
    useCanManageOrgProfilesMock.mockReturnValue(true);
  });

  it("reopening shows conversation active_profile Opus even though global default is Fable", () => {
    useActiveConversationMock.mockReturnValue({
      data: { active_profile: "Opus", llm_model: "opus-model" },
    });
    const { result } = renderState();
    expect(result.current.currentProfileName).toBe("Opus");
  });

  it("reopening shows Opus via model match when stamped profile is null, not global Fable", () => {
    useActiveConversationMock.mockReturnValue({
      data: { active_profile: null, llm_model: "opus-model" },
    });
    const { result } = renderState();
    expect(result.current.currentProfileName).toBe("Opus");
  });

  it("inside conversation with pending history (conversation undefined) does NOT fallback to global Fable", () => {
    // Simulates initial reload: conversation query still pending, profiles already loaded, global is Fable
    useActiveConversationMock.mockReturnValue({ data: undefined });
    useLlmProfilesMock.mockReturnValue({
      data: { profiles: PROFILES, active_profile: "Fable" },
      isLoading: false,
    });
    const { result } = renderState();
    // Bug: would show Fable (global). Fix: should show null until conversation loads.
    expect(result.current.currentProfileName).not.toBe("Fable");
    expect(result.current.currentProfileName).toBeNull();
  });

  it("home page (no conversation) still shows global default Fable", () => {
    useOptionalConversationIdMock.mockReturnValue({ conversationId: null });
    useActiveConversationMock.mockReturnValue({ data: undefined });
    const { result } = renderState();
    expect(result.current.currentProfileName).toBe("Fable");
  });
});
