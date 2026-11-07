import { renderHook } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import type { AcpModelContext } from "#/hooks/use-acp-model-context";
import { ACP_VERTEX_SAFE_MODEL } from "#/constants/acp-providers";

const useActiveConversationMock = vi.fn();
const useSettingsMock = vi.fn();
const useActiveBackendMock = vi.fn();
const useAcpModelContextMock = vi.fn();
const useOptionalConversationIdMock = vi.fn();

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => useSettingsMock(),
}));

vi.mock("#/contexts/active-backend-context", async () => {
  const actual = await vi.importActual<
    typeof import("#/contexts/active-backend-context")
  >("#/contexts/active-backend-context");
  return {
    ...actual,
    useActiveBackend: () => useActiveBackendMock(),
  };
});

vi.mock("#/hooks/use-acp-model-context", () => ({
  useAcpModelContext: () => useAcpModelContextMock(),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => useOptionalConversationIdMock(),
}));

// The detail query and the org-permission check need a QueryClient this
// wrapper-less harness doesn't provide; both are driven per test (detail null
// → the settings fallback the older tests exercise).
const useActiveAcpProfileDetailMock = vi.fn();
vi.mock("#/hooks/query/use-active-acp-profile-detail", () => ({
  useActiveAcpProfileDetail: () => useActiveAcpProfileDetailMock(),
}));

const useCanManageOrgProfilesMock = vi.fn();
vi.mock("#/hooks/use-can-manage-org-profiles", () => ({
  useCanManageOrgProfiles: () => useCanManageOrgProfilesMock(),
}));

const useAcpModelDiscoveryMock = vi.fn();
vi.mock("#/hooks/query/use-acp-model-discovery", () => ({
  useAcpModelDiscovery: (...args: unknown[]) =>
    useAcpModelDiscoveryMock(...args),
}));

const useAcpSessionModelsMock = vi.fn();
vi.mock("#/hooks/query/use-acp-session-models", () => ({
  useAcpSessionModels: (...args: unknown[]) => useAcpSessionModelsMock(...args),
}));

const NO_LIVE_MODELS = { models: [], defaultModelId: null };
const CLAUDE_MODELS = [
  { id: "default", label: "Default (recommended)" },
  { id: "sonnet", label: "Sonnet" },
];

import { useChatInputModelState } from "#/hooks/use-chat-input-model-state";

// `useAcpModelContext` derives these booleans; here we drive them directly so
// each branch of `useChatInputModelState` is documented in isolation.
const acpContext = (
  overrides: Partial<AcpModelContext> = {},
): AcpModelContext => ({
  isActiveAcpConversation: false,
  isHomeAcp: false,
  isAcpContext: false,
  destinationPath: "/settings/llm",
  destinationLabel: "LLM Profiles",
  ...overrides,
});

describe("useChatInputModelState", () => {
  beforeEach(() => {
    useActiveConversationMock.mockReset();
    useActiveConversationMock.mockReturnValue({ data: undefined });
    useSettingsMock.mockReset();
    useSettingsMock.mockReturnValue({ data: undefined });
    useActiveBackendMock.mockReset();
    // Default to a local backend — live ACP switching is local-only.
    useActiveBackendMock.mockReturnValue({ backend: { kind: "local" } });
    useAcpModelContextMock.mockReset();
    useAcpModelContextMock.mockReturnValue(acpContext());
    useOptionalConversationIdMock.mockReset();
    useOptionalConversationIdMock.mockReturnValue({ conversationId: null });
    useActiveAcpProfileDetailMock.mockReset();
    useActiveAcpProfileDetailMock.mockReturnValue(null);
    useCanManageOrgProfilesMock.mockReset();
    useCanManageOrgProfilesMock.mockReturnValue(true);
    useAcpModelDiscoveryMock.mockReset();
    useAcpModelDiscoveryMock.mockReturnValue(NO_LIVE_MODELS);
    useAcpSessionModelsMock.mockReset();
    useAcpSessionModelsMock.mockImplementation(
      (conversation) => conversation?.acp_available_models ?? [],
    );
  });

  it("active ACP: lists the models the conversation's session reported", () => {
    const live = [
      { id: "default", label: "Default (recommended)" },
      { id: "claude-fable-5[1m]", label: "Fable 5 (1M)" },
    ];
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "c1",
        agent_kind: "acp",
        acp_server: "claude-code",
        llm_model: "claude-fable-5[1m]",
        acp_available_models: live,
      },
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isActiveAcpConversation: true, isAcpContext: true }),
    );

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.availableAcpModels).toEqual(live);
    expect(result.current.displayModel).toBe("Fable 5 (1M)");
  });

  it("active ACP: a custom server's session list makes the picker selectable", () => {
    const live = [
      { id: "swe-2-high", label: "SWE-2 High" },
      { id: "swe-2-low", label: "SWE-2 Low" },
    ];
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "c1",
        agent_kind: "acp",
        acp_server: "custom",
        llm_model: "swe-2-high",
        acp_available_models: live,
      },
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isActiveAcpConversation: true, isAcpContext: true }),
    );

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.availableAcpModels).toEqual(live);
    expect(result.current.displayModel).toBe("SWE-2 High");
    expect(result.current.showAcpPicker).toBe(true);
    expect(useAcpModelDiscoveryMock).toHaveBeenCalledWith(null, {
      enabled: true,
      secretRefs: null,
    });
  });

  it("active ACP: uses the agent's reported list until the session reports its own", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "c1",
        agent_kind: "acp",
        acp_server: "codex",
        llm_model: "gpt-5.6-terra",
        acp_available_models: [],
      },
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isActiveAcpConversation: true, isAcpContext: true }),
    );
    const live = [
      { id: "gpt-6-astra", label: "GPT-6 Astra" },
      { id: "gpt-5.6-terra", label: "GPT-5.6 Terra" },
    ];
    useAcpModelDiscoveryMock.mockReturnValue({
      models: live,
      defaultModelId: "gpt-6-astra",
    });

    const { result } = renderHook(() => useChatInputModelState());

    expect(useAcpModelDiscoveryMock).toHaveBeenCalledWith("codex", {
      enabled: true,
      secretRefs: null,
    });
    expect(result.current.availableAcpModels).toEqual(live);
    expect(result.current.currentModelId).toBe("gpt-5.6-terra");
  });

  it("active ACP: asks for models with the secrets the conversation launched with", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "c1",
        agent_kind: "acp",
        acp_server: "codex",
        llm_model: "gpt-5.6-terra",
        launched_agent_profile: {
          agent_profile_id: "id-codex",
          revision: 3,
          secret_refs: ["OPENAI_API_KEY"],
        },
      },
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isActiveAcpConversation: true, isAcpContext: true }),
    );
    // The home page's active profile is not the one this conversation launched.
    useActiveAcpProfileDetailMock.mockReturnValue({
      id: "id-other",
      name: "other",
      agent_kind: "acp",
      acp_server: "codex",
      acp_model: null,
      secret_refs: null,
    });

    renderHook(() => useChatInputModelState());

    expect(useAcpModelDiscoveryMock).toHaveBeenCalledWith("codex", {
      enabled: true,
      secretRefs: ["OPENAI_API_KEY"],
    });
  });

  it("home ACP: lists the models the agent reported and shows its own default", () => {
    useActiveAcpProfileDetailMock.mockReturnValue({
      id: "id-pi",
      name: "pi",
      agent_kind: "acp",
      acp_server: "pi",
      acp_model: null,
      secret_refs: null,
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isHomeAcp: true, isAcpContext: true }),
    );
    const live = [
      { id: "anthropic/claude-opus-4-8", label: "Claude Opus 4.8" },
      { id: "anthropic/claude-sonnet-5", label: "Claude Sonnet 5" },
    ];
    useAcpModelDiscoveryMock.mockReturnValue({
      models: live,
      defaultModelId: "anthropic/claude-opus-4-8",
    });

    const { result } = renderHook(() => useChatInputModelState());

    expect(useAcpModelDiscoveryMock).toHaveBeenCalledWith("pi", {
      enabled: true,
      secretRefs: null,
    });
    expect(result.current.availableAcpModels).toEqual(live);
    expect(result.current.currentModelId).toBe("anthropic/claude-opus-4-8");
    expect(result.current.displayModel).toBe("Claude Opus 4.8");
  });

  it("non-ACP: shows the conversation/settings llm_model with no picker", () => {
    useActiveConversationMock.mockReturnValue({
      data: { conversation_id: "c1", llm_model: "openai/gpt-4o" },
    });
    useOptionalConversationIdMock.mockReturnValue({ conversationId: "c1" });

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.isAcpContext).toBe(false);
    expect(result.current.currentModelId).toBe("openai/gpt-4o");
    expect(result.current.displayModel).toBe("openai/gpt-4o");
    expect(result.current.availableAcpModels).toEqual([]);
    expect(result.current.showAcpPicker).toBe(false);
    // switchConversationId is ACP-only — null for native conversations.
    expect(result.current.switchConversationId).toBeNull();
    expect(result.current.destinationPath).toBe("/settings/llm");
  });

  it("non-ACP: falls back to settings.llm_model when the conversation has none", () => {
    useActiveConversationMock.mockReturnValue({ data: undefined });
    useSettingsMock.mockReturnValue({ data: { llm_model: "openai/gpt-4o" } });

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.currentModelId).toBe("openai/gpt-4o");
  });

  it("active ACP: offers no hardcoded models before the agent reports any", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "c1",
        agent_kind: "acp",
        acp_server: "claude-code",
        llm_model: "sonnet",
      },
    });
    useOptionalConversationIdMock.mockReturnValue({ conversationId: "c1" });
    useAcpModelContextMock.mockReturnValue(
      acpContext({
        isActiveAcpConversation: true,
        isAcpContext: true,
        destinationPath: "/settings/agents",
        destinationLabel: "Agent",
      }),
    );

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.isAcpContext).toBe(true);
    expect(result.current.currentModelId).toBe("sonnet");
    expect(result.current.displayModel).toBe("sonnet");
    expect(result.current.availableAcpModels).toEqual([]);
    expect(result.current.showAcpPicker).toBe(false);
    // Live switch targets the navigation conversation id.
    expect(result.current.switchConversationId).toBe("c1");
    expect(result.current.destinationPath).toBe("/settings/agents");
  });

  it("home ACP: resolves the configured acp_model and exposes the picker, but no live-switch target", () => {
    useActiveConversationMock.mockReturnValue({ data: undefined });
    useSettingsMock.mockReturnValue({
      data: {
        agent_settings: {
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_model: "claude-sonnet-4-6",
        },
      },
    });
    useOptionalConversationIdMock.mockReturnValue({ conversationId: null });
    useAcpModelContextMock.mockReturnValue(
      acpContext({
        isHomeAcp: true,
        isAcpContext: true,
        destinationPath: "/settings/agents",
        destinationLabel: "Agent",
      }),
    );
    useAcpModelDiscoveryMock.mockReturnValue({
      models: CLAUDE_MODELS,
      defaultModelId: "default",
    });

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.currentModelId).toBe("claude-sonnet-4-6");
    expect(result.current.showAcpPicker).toBe(true);
    // Home / no session → there is no conversation to switch in place.
    expect(result.current.switchConversationId).toBeNull();
  });

  it("home ACP: shows the agent's own default when no acp_model is saved", () => {
    useActiveConversationMock.mockReturnValue({ data: undefined });
    useSettingsMock.mockReturnValue({
      data: {
        agent_settings: { agent_kind: "acp", acp_server: "claude-code" },
      },
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isHomeAcp: true, isAcpContext: true }),
    );
    useAcpModelDiscoveryMock.mockReturnValue({
      models: CLAUDE_MODELS,
      defaultModelId: "default",
    });

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.currentModelId).toBe("default");
    expect(result.current.displayModel).toBe("Default (recommended)");
  });

  it("home ACP: names the agent's default when no one has reported it yet", () => {
    useActiveBackendMock.mockReturnValue({ backend: { kind: "cloud" } });
    useActiveConversationMock.mockReturnValue({ data: undefined });
    useSettingsMock.mockReturnValue({
      data: {
        agent_settings: { agent_kind: "acp", acp_server: "claude-code" },
      },
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isHomeAcp: true, isAcpContext: true }),
    );
    useAcpModelDiscoveryMock.mockReturnValue({
      models: CLAUDE_MODELS,
      defaultModelId: null,
    });

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.currentModelId).toBeNull();
    expect(result.current.displayModel).toBe(
      "SETTINGS$AGENT_MODEL_AGENT_DEFAULT",
    );
    expect(result.current.showAcpPicker).toBe(true);
  });

  it("home ACP: Gemini keeps its Vertex-safe model over the agent's default", () => {
    useActiveConversationMock.mockReturnValue({ data: undefined });
    useSettingsMock.mockReturnValue({
      data: {
        agent_settings: { agent_kind: "acp", acp_server: "gemini-cli" },
      },
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isHomeAcp: true, isAcpContext: true }),
    );
    useAcpModelDiscoveryMock.mockReturnValue({
      models: [],
      defaultModelId: "gemini-3-flash-preview",
    });

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.currentModelId).toBe(ACP_VERTEX_SAFE_MODEL);
  });

  it("home ACP: the active profile's detail overrides stale agent settings for provider and model", () => {
    // Activation is pointer-only: settings still describe claude-code, but the
    // active ACP profile is codex — the picker must follow the profile (the
    // conversation launch source), not the settings.
    useActiveConversationMock.mockReturnValue({ data: undefined });
    useSettingsMock.mockReturnValue({
      data: {
        agent_settings: {
          agent_kind: "acp",
          acp_server: "claude-code",
          acp_model: "claude-sonnet-4-6",
        },
      },
    });
    useActiveAcpProfileDetailMock.mockReturnValue({
      id: "id-codex",
      name: "codex-test",
      agent_kind: "acp",
      acp_server: "codex",
      acp_model: "gpt-5.5",
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isHomeAcp: true, isAcpContext: true }),
    );
    const codexModels = [{ id: "gpt-5.5", label: "GPT-5.5" }];
    useAcpModelDiscoveryMock.mockReturnValue({
      models: codexModels,
      defaultModelId: null,
    });

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.currentModelId).toBe("gpt-5.5");
    expect(useAcpModelDiscoveryMock).toHaveBeenCalledWith("codex", {
      enabled: true,
      secretRefs: null,
    });
    expect(result.current.availableAcpModels).toEqual(codexModels);
  });

  it("home ACP: asks for models with the active profile's secret scope", () => {
    useActiveAcpProfileDetailMock.mockReturnValue({
      id: "id-codex",
      name: "codex-test",
      agent_kind: "acp",
      acp_server: "codex",
      acp_model: null,
      secret_refs: ["OPENAI_API_KEY"],
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isHomeAcp: true, isAcpContext: true }),
    );

    renderHook(() => useChatInputModelState());

    expect(useAcpModelDiscoveryMock).toHaveBeenCalledWith("codex", {
      enabled: true,
      secretRefs: ["OPENAI_API_KEY"],
    });
  });

  it("home ACP: does not ask for models until the active profile loads", () => {
    useSettingsMock.mockReturnValue({
      data: { agent_settings: { acp_server: "codex", acp_model: null } },
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isHomeAcp: true, isAcpContext: true }),
    );

    renderHook(() => useChatInputModelState());

    expect(useAcpModelDiscoveryMock).toHaveBeenCalledWith("codex", {
      enabled: false,
      secretRefs: null,
    });
  });

  it("home ACP on cloud: hides the selectable rows from members who cannot manage org profiles", () => {
    // A home pick persists into the org-owned profile; a member's pick would
    // only 403. The chip and settings link remain (showAcpPicker false).
    useActiveBackendMock.mockReturnValue({ backend: { kind: "cloud" } });
    useCanManageOrgProfilesMock.mockReturnValue(false);
    useActiveConversationMock.mockReturnValue({ data: undefined });
    useSettingsMock.mockReturnValue({
      data: {
        agent_settings: { agent_kind: "acp", acp_server: "claude-code" },
      },
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isHomeAcp: true, isAcpContext: true }),
    );
    useAcpModelDiscoveryMock.mockReturnValue({
      models: CLAUDE_MODELS,
      defaultModelId: null,
    });

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.availableAcpModels.length).toBeGreaterThan(0);
    expect(result.current.showAcpPicker).toBe(false);
  });

  it("showAcpPicker: a cloud conversation's session list enables the picker", () => {
    useActiveBackendMock.mockReturnValue({ backend: { kind: "cloud" } });
    const conversation = {
      conversation_id: "c1",
      agent_kind: "acp",
      acp_server: "claude-code",
      llm_model: "sonnet",
    };
    useActiveConversationMock.mockReturnValue({ data: conversation });
    useAcpSessionModelsMock.mockReturnValue(CLAUDE_MODELS);
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isActiveAcpConversation: true, isAcpContext: true }),
    );

    const { result } = renderHook(() => useChatInputModelState());

    expect(useAcpSessionModelsMock).toHaveBeenCalledWith(conversation);
    expect(result.current.availableAcpModels).toEqual(CLAUDE_MODELS);
    expect(result.current.displayModel).toBe("Sonnet");
    expect(result.current.showAcpPicker).toBe(true);
  });

  it("showAcpPicker tri-condition: an unknown ACP provider has no model list → no picker", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "c1",
        agent_kind: "acp",
        acp_server: "some-custom-server",
        llm_model: "custom-model",
      },
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isActiveAcpConversation: true, isAcpContext: true }),
    );

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.availableAcpModels).toEqual([]);
    expect(result.current.showAcpPicker).toBe(false);
    // No reported label → falls back to the raw id.
    expect(result.current.displayModel).toBe("custom-model");
  });
});
