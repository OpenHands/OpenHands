import { renderHook } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import type { AcpModelContext } from "#/hooks/use-acp-model-context";
import { getAcpProvider } from "#/constants/acp-providers";

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

// `getAcpProvider`/`labelForAcpModel`/`resolveEffectiveAcpModel` are exercised
// for real (not mocked) so the test pins the actual registry-sourced model
// list the picker shows.
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

  it("active ACP: resolves the provider's available models (getAcpProvider called for active contexts, not just home)", () => {
    // Regression guard: in the old ChatInputModel `getAcpProvider` ran only on
    // the home branch. The shared hook calls it for ANY ACP context so the
    // picker has a model list on active conversations too. Pin that contract.
    const provider = getAcpProvider("claude-code");
    expect(provider?.available_models?.length).toBeGreaterThan(0);

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
    // Human label resolved from the registry (matches the conversation chip).
    expect(result.current.displayModel).toBe("Claude Sonnet");
    expect(result.current.availableAcpModels).toEqual(
      provider?.available_models,
    );
    // Local backend + ACP + a non-empty model list → picker is enabled.
    expect(result.current.showAcpPicker).toBe(true);
    // Live switch targets the navigation conversation id.
    expect(result.current.switchConversationId).toBe("c1");
    expect(result.current.destinationPath).toBe("/settings/agents");
  });

  it("active ACP: prefers the conversation's runtime available_models over the static catalogue (#18053)", () => {
    // A ``custom`` ACP profile has no static registry entry, so the runtime
    // catalogue the server reported at ``session/new`` (e.g. ``devin acp``)
    // is the only model list the picker can show.
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "c1",
        agent_kind: "acp",
        acp_server: "some-custom-server",
        llm_model: "swe-2-high",
        available_models: [
          { model_id: "swe-2-high", name: "SWE 2 High" },
          { model_id: "model-b", name: null },
          { model_id: "model-c" },
        ],
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

    // Rows: one per runtime entry, labelled by ``name`` with a ``model_id``
    // fallback.
    expect(result.current.availableAcpModels).toEqual([
      { id: "swe-2-high", label: "SWE 2 High" },
      { id: "model-b", label: "model-b" },
      { id: "model-c", label: "model-c" },
    ]);
    // A custom profile with a runtime list now shows the picker.
    expect(result.current.showAcpPicker).toBe(true);
    // Chip: the matching runtime entry's ``name``, not the raw id.
    expect(result.current.displayModel).toBe("SWE 2 High");
    expect(result.current.switchConversationId).toBe("c1");
  });

  it("active ACP: a runtime list wins over the static catalogue for built-in providers", () => {
    // Runtime lists reflect the credentials actually configured (e.g.
    // account-tier model availability), not the build-time snapshot.
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "c1",
        agent_kind: "acp",
        acp_server: "claude-code",
        llm_model: "sonnet",
        available_models: [
          { model_id: "sonnet", name: "Claude Sonnet" },
          { model_id: "account-tier-model", name: "Account-tier model" },
        ],
      },
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isActiveAcpConversation: true, isAcpContext: true }),
    );

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.availableAcpModels).toEqual([
      { id: "sonnet", label: "Claude Sonnet" },
      { id: "account-tier-model", label: "Account-tier model" },
    ]);
  });

  it("active ACP: falls back to the static catalogue when the runtime list is empty", () => {
    const provider = getAcpProvider("claude-code");
    expect(provider?.available_models?.length).toBeGreaterThan(0);

    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "c1",
        agent_kind: "acp",
        acp_server: "claude-code",
        llm_model: "sonnet",
        available_models: [],
      },
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isActiveAcpConversation: true, isAcpContext: true }),
    );

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.availableAcpModels).toEqual(
      provider?.available_models,
    );
  });

  it("active ACP: chip label falls back to the static label, then the raw id, when the runtime entry has no name", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "c1",
        agent_kind: "acp",
        acp_server: "claude-code",
        llm_model: "sonnet",
        available_models: [{ model_id: "sonnet" }],
      },
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isActiveAcpConversation: true, isAcpContext: true }),
    );

    const { result } = renderHook(() => useChatInputModelState());

    // The row label has no runtime name to use, so it falls back to the id…
    expect(result.current.availableAcpModels).toEqual([
      { id: "sonnet", label: "sonnet" },
    ]);
    // …while the chip prefers the static registry label.
    expect(result.current.displayModel).toBe("Claude Sonnet");
  });

  it("active ACP: chip keeps the raw id for a nameless custom entry (there is no static label)", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "c1",
        agent_kind: "acp",
        acp_server: "some-custom-server",
        llm_model: "model-b",
        available_models: [{ model_id: "model-b" }],
      },
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isActiveAcpConversation: true, isAcpContext: true }),
    );

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.displayModel).toBe("model-b");
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

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.currentModelId).toBe("claude-sonnet-4-6");
    expect(result.current.showAcpPicker).toBe(true);
    // Home / no session → there is no conversation to switch in place.
    expect(result.current.switchConversationId).toBeNull();
  });

  it("home ACP: falls back to the provider default when no acp_model is saved", () => {
    const provider = getAcpProvider("claude-code");
    useActiveConversationMock.mockReturnValue({ data: undefined });
    useSettingsMock.mockReturnValue({
      data: {
        agent_settings: { agent_kind: "acp", acp_server: "claude-code" },
      },
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isHomeAcp: true, isAcpContext: true }),
    );

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.currentModelId).toBe(provider?.default_model);
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

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.currentModelId).toBe("gpt-5.5");
    expect(result.current.availableAcpModels).toEqual(
      getAcpProvider("codex")?.available_models,
    );
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

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.availableAcpModels.length).toBeGreaterThan(0);
    expect(result.current.showAcpPicker).toBe(false);
  });

  it("showAcpPicker: cloud backend shows the picker when a model list is present (cloud ACP supports mid-conversation switching)", () => {
    useActiveBackendMock.mockReturnValue({ backend: { kind: "cloud" } });
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "c1",
        agent_kind: "acp",
        acp_server: "claude-code",
        llm_model: "claude-sonnet-4-6",
      },
    });
    useAcpModelContextMock.mockReturnValue(
      acpContext({ isActiveAcpConversation: true, isAcpContext: true }),
    );

    const { result } = renderHook(() => useChatInputModelState());

    expect(result.current.availableAcpModels.length).toBeGreaterThan(0);
    // ACP + model list present → picker is enabled on all backends
    // (cloud ACP conversations support mid-conversation model switching).
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
    // Unknown model id has no registry label → falls back to the raw id.
    expect(result.current.displayModel).toBe("custom-model");
  });
});
