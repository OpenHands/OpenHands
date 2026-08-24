import React from "react";
import { fireEvent, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderWithProviders } from "test-utils";
import { fetchModelsDevCatalog } from "#/api/models-dev-catalog";
import { useAcpCustomModelsStore } from "#/stores/acp-custom-models-store";

// Never resolves: keeps the models.dev catalog request pending for the
// life of every test here (none of them assert on catalog behavior — that's
// use-acp-model-choices.test.tsx's job), so the picker's contents stay
// deterministic (curated + live + custom only) without a real network call.
vi.mock("#/api/models-dev-catalog", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/api/models-dev-catalog")>();
  return {
    ...actual,
    fetchModelsDevCatalog: vi.fn(() => new Promise(() => {})),
  };
});

const useActiveConversationMock = vi.fn();
const useSettingsMock = vi.fn();
const useActiveBackendMock = vi.fn();
const switchAcpModelMutate = vi.fn();

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

vi.mock("#/hooks/mutation/use-switch-acp-model", () => ({
  useSwitchAcpModel: () => ({ mutate: switchAcpModelMutate }),
}));

import { ChatInputModel } from "#/components/features/chat/components/chat-input-model";

describe("ChatInputModel", () => {
  beforeEach(() => {
    useActiveConversationMock.mockReset();
    useSettingsMock.mockReset();
    useSettingsMock.mockReturnValue({ data: undefined });
    useActiveBackendMock.mockReset();
    // Default to a local backend (mirrors useActiveBackend's standalone
    // fallback): live ACP model switching is local-only.
    useActiveBackendMock.mockReturnValue({ backend: { kind: "local" } });
    switchAcpModelMutate.mockReset();
    vi.mocked(fetchModelsDevCatalog).mockClear();
    useAcpCustomModelsStore.setState({ customModelsByProfileId: {} });
  });

  it("renders the active conversation's llm_model when present", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-conversation-id",
        llm_model: "openai/gpt-4o",
      },
    });

    renderWithProviders(<ChatInputModel />);

    const model = screen.getByTestId("chat-input-llm-model");
    expect(model).toBeInTheDocument();
    expect(model).toHaveTextContent("openai/gpt…");
    expect(model).toHaveAttribute("title", "openai/gpt-4o");
    expect(
      screen.queryByTestId("chat-input-llm-model-popover"),
    ).not.toBeInTheDocument();

    fireEvent.click(model);
    const popover = screen.getByTestId("chat-input-llm-model-popover");
    expect(popover).toHaveTextContent("openai/gpt-4o");
    const llmSettingsLink = screen.getByRole("link", {
      name: /LLM Profiles|SETTINGS\$LLM_PROFILES|LLM Settings|SETTINGS\$LLM_SETTINGS/,
    });
    expect(llmSettingsLink).toHaveAttribute("href", "/settings/llm");
  });

  it("renders nothing when llm_model is missing", () => {
    useActiveConversationMock.mockReturnValue({
      data: { conversation_id: "test-conversation-id" },
    });

    renderWithProviders(<ChatInputModel />);

    expect(
      screen.queryByTestId("chat-input-llm-model"),
    ).not.toBeInTheDocument();
  });

  it("renders an ACP conversation model and links to Agent settings", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-conversation-id",
        agent_kind: "acp",
        acp_server: "claude-code",
        llm_model: "sonnet",
      },
    });

    renderWithProviders(<ChatInputModel />);

    const model = screen.getByTestId("chat-input-llm-model");
    // ACP surfaces show the provider's human label (matching the conversation
    // list chip), resolved from ``acp_server`` + the raw ``acp_model`` id.
    expect(model).toHaveAttribute("title", "Claude Sonnet");
    fireEvent.click(model);
    expect(screen.getByRole("link")).toHaveAttribute(
      "href",
      "/settings/agents",
    );
  });

  it("does not fall back to the OpenHands settings model for active ACP conversations", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-conversation-id",
        agent_kind: "acp",
        llm_model: null,
      },
    });
    useSettingsMock.mockReturnValue({
      data: { llm_model: "openai/gpt-4o" },
    });

    renderWithProviders(<ChatInputModel />);

    expect(
      screen.queryByTestId("chat-input-llm-model"),
    ).not.toBeInTheDocument();
  });

  it("falls back to the user's default model from settings when there is no active conversation", () => {
    // Arrange — home page render: no conversation yet, but the user has
    // a default model configured. The switcher should still show.
    useActiveConversationMock.mockReturnValue({ data: undefined });
    useSettingsMock.mockReturnValue({
      data: { llm_model: "anthropic/claude-sonnet-4-20250514" },
    });

    renderWithProviders(<ChatInputModel />);

    const model = screen.getByTestId("chat-input-llm-model");
    expect(model).toHaveTextContent("anthropic/…");
    expect(model).toHaveAttribute(
      "title",
      "anthropic/claude-sonnet-4-20250514",
    );
  });

  it("uses the ACP settings model on the home page when ACP is active", () => {
    useActiveConversationMock.mockReturnValue({ data: undefined });
    useSettingsMock.mockReturnValue({
      data: {
        llm_model: "openai/gpt-4o",
        agent_settings: {
          agent_kind: "acp",
          acp_model: "gemini-2.5-pro",
        },
      },
    });

    renderWithProviders(<ChatInputModel />);

    const model = screen.getByTestId("chat-input-llm-model");
    expect(model).toHaveAttribute("title", "gemini-2.5-pro");
    fireEvent.click(model);
    expect(screen.getByRole("link")).toHaveAttribute(
      "href",
      "/settings/agents",
    );
  });

  it("renders nothing when neither the conversation nor settings provide an llm_model", () => {
    useActiveConversationMock.mockReturnValue({ data: undefined });
    useSettingsMock.mockReturnValue({ data: undefined });

    renderWithProviders(<ChatInputModel />);

    expect(
      screen.queryByTestId("chat-input-llm-model"),
    ).not.toBeInTheDocument();
  });

  it("renders nothing for ACP conversations and does NOT fall back to settings.llm_model", () => {
    // The ACP subprocess owns its model (via ``acp_model``); ``llm_model``
    // is null on the conversation by design. The previous fallback to
    // ``settings.llm_model`` would have resurrected the user's *default*
    // OpenHands model on, say, a Claude-Code conversation — visibly
    // wrong (the link goes to /settings, which is itself disabled for
    // ACP) and silently lies about what model is actually running.
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-conversation-id",
        agent_kind: "acp",
        llm_model: null,
      },
    });
    useSettingsMock.mockReturnValue({
      data: { llm_model: "anthropic/claude-sonnet-4-20250514" },
    });

    renderWithProviders(<ChatInputModel />);

    expect(
      screen.queryByTestId("chat-input-llm-model"),
    ).not.toBeInTheDocument();
  });

  it("shows the provider default on the home page when ACP is the default agent and no model is saved", () => {
    // Home-screen gating: no active conversation and no saved ``acp_model``.
    // The next-created conversation will inherit the provider's
    // ``default_model`` (see buildConfiguredAcpAgentSettings), so the picker
    // shows that same default — matching what the runtime will actually
    // start. Picker links to /settings/agent (not /settings) since
    // ``settings.llm_model`` doesn't apply to ACP.
    useActiveConversationMock.mockReturnValue({ data: undefined });
    useSettingsMock.mockReturnValue({
      data: {
        agent_settings: { agent_kind: "acp", acp_server: "claude-code" },
        // settings.llm_model is set (user has an OpenHands default
        // configured), but agent_kind=acp suppresses it.
        llm_model: "anthropic/claude-sonnet-4-20250514",
      },
    });

    renderWithProviders(<ChatInputModel />);

    const model = screen.getByTestId("chat-input-llm-model");
    // Claude Code's registered default (``opus[1m]``), shown as its
    // human label to match the conversation list chip. See CLAUDE_MODELS in
    // acp-providers.ts.
    expect(model).toHaveAttribute("title", "Claude Opus (1M)");
    fireEvent.click(model);
    expect(screen.getByRole("link")).toHaveAttribute(
      "href",
      "/settings/agents",
    );
  });

  it("renders the provider's available models as selectable rows for an ACP conversation", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-conversation-id",
        agent_kind: "acp",
        acp_server: "claude-code",
        llm_model: "sonnet",
      },
    });

    renderWithProviders(<ChatInputModel />);

    fireEvent.click(screen.getByTestId("chat-input-llm-model"));

    // Every registered Claude Code model is offered as a row, and the running
    // one (sonnet) is marked selected.
    const selectedRow = screen.getByTestId(
      "chat-input-acp-model-option-sonnet",
    );
    expect(selectedRow).toBeInTheDocument();
    expect(selectedRow).toHaveTextContent("Claude Sonnet");
    expect(
      screen.getByTestId("chat-input-acp-model-option-opus[1m]"),
    ).toBeInTheDocument();
  });

  it("live-switches the model when a row is selected in an active ACP conversation", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-conversation-id",
        agent_kind: "acp",
        acp_server: "claude-code",
        llm_model: "sonnet",
      },
    });

    renderWithProviders(<ChatInputModel />);

    fireEvent.click(screen.getByTestId("chat-input-llm-model"));
    fireEvent.click(screen.getByTestId("chat-input-acp-model-option-opus[1m]"));

    // Active conversation → live switch keyed by the conversation id from the
    // navigation context (test-conversation-id), default-write NOT used.
    expect(switchAcpModelMutate).toHaveBeenCalledWith({
      conversationId: "test-conversation-id",
      model: "opus[1m]",
    });
    // Popover closes after a selection.
    expect(
      screen.queryByTestId("chat-input-llm-model-popover"),
    ).not.toBeInTheDocument();
  });

  it("persists the choice as the default (conversationId null) in the home ACP case", () => {
    useActiveConversationMock.mockReturnValue({ data: undefined });
    useSettingsMock.mockReturnValue({
      data: {
        agent_settings: { agent_kind: "acp", acp_server: "claude-code" },
        llm_model: "anthropic/claude-sonnet-4-20250514",
      },
    });

    renderWithProviders(<ChatInputModel />);

    fireEvent.click(screen.getByTestId("chat-input-llm-model"));
    fireEvent.click(screen.getByTestId("chat-input-acp-model-option-sonnet"));

    // Home / no session → null conversationId routes the hook to the
    // settings-default write path.
    expect(switchAcpModelMutate).toHaveBeenCalledWith({
      conversationId: null,
      model: "sonnet",
    });
  });

  it("offers selectable rows on a cloud backend for ACP conversations (mid-conversation model switching is supported)", () => {
    useActiveBackendMock.mockReturnValue({ backend: { kind: "cloud" } });
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-conversation-id",
        agent_kind: "acp",
        acp_server: "claude-code",
        llm_model: "sonnet",
      },
    });

    renderWithProviders(<ChatInputModel />);

    fireEvent.click(screen.getByTestId("chat-input-llm-model"));

    // Cloud ACP conversations support mid-conversation model switching,
    // so selectable model rows are shown.
    const selectedRow = screen.getByTestId(
      "chat-input-acp-model-option-sonnet",
    );
    expect(selectedRow).toBeInTheDocument();
    expect(selectedRow).toHaveTextContent("Claude Sonnet");
    expect(
      screen.getByTestId("chat-input-acp-model-option-opus[1m]"),
    ).toBeInTheDocument();
    const popover = screen.getByTestId("chat-input-llm-model-popover");
    expect(popover).toHaveTextContent("Claude Sonnet");
    expect(screen.getByRole("link")).toHaveAttribute(
      "href",
      "/settings/agents",
    );
  });

  it("renders a live session model as a selectable row alongside the curated list", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-conversation-id",
        agent_kind: "acp",
        acp_server: "claude-code",
        llm_model: "sonnet",
        acp_live_models: [{ id: "session-only", label: "Session-Only Model" }],
      },
    });

    renderWithProviders(<ChatInputModel />);
    fireEvent.click(screen.getByTestId("chat-input-llm-model"));

    expect(
      screen.getByTestId("chat-input-acp-model-option-session-only"),
    ).toHaveTextContent("Session-Only Model");
    // The curated list is still offered alongside the live-only model.
    expect(
      screen.getByTestId("chat-input-acp-model-option-sonnet"),
    ).toBeInTheDocument();
  });

  it("marks the base entry selected for a composite '<base>/<effort>' session model", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-conversation-id",
        agent_kind: "acp",
        acp_server: "claude-code",
        llm_model: "sonnet/high",
      },
    });

    renderWithProviders(<ChatInputModel />);
    fireEvent.click(screen.getByTestId("chat-input-llm-model"));

    // The composite id itself never appears as a row (choices are always
    // bare base ids) — its base "sonnet" is the one marked current.
    expect(
      screen.queryByTestId("chat-input-acp-model-option-sonnet/high"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("chat-input-acp-model-option-sonnet"),
    ).toHaveClass("bg-[var(--oh-interactive-hover)]");
  });

  it("does not re-switch when selecting the already-current base of a composite session model", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-conversation-id",
        agent_kind: "acp",
        acp_server: "claude-code",
        llm_model: "sonnet/high",
      },
    });

    renderWithProviders(<ChatInputModel />);
    fireEvent.click(screen.getByTestId("chat-input-llm-model"));
    fireEvent.click(screen.getByTestId("chat-input-acp-model-option-sonnet"));

    // Clicking the row already highlighted as current (via the base-id
    // fallback) is a no-op — it must not switch to a bare id that's already
    // effectively selected.
    expect(switchAcpModelMutate).not.toHaveBeenCalled();
    expect(
      screen.queryByTestId("chat-input-llm-model-popover"),
    ).not.toBeInTheDocument();
  });

  it("switching to a different base model from a composite session model preserves the current effort (M5)", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-conversation-id",
        agent_kind: "acp",
        acp_server: "claude-code",
        llm_model: "sonnet/high",
      },
    });

    renderWithProviders(<ChatInputModel />);
    fireEvent.click(screen.getByTestId("chat-input-llm-model"));
    fireEvent.click(screen.getByTestId("chat-input-acp-model-option-opus[1m]"));

    // The current "high" effort rides along onto the newly picked base via
    // composeAcpModelId — upgraded from M3's "drop it" behavior.
    expect(switchAcpModelMutate).toHaveBeenCalledWith({
      conversationId: "test-conversation-id",
      model: "opus[1m]/high",
    });
  });

  describe("M5: effort switching", () => {
    it("renders an effort section for a claude-code conversation, with the current effort checked", () => {
      useActiveConversationMock.mockReturnValue({
        data: {
          conversation_id: "test-conversation-id",
          agent_kind: "acp",
          acp_server: "claude-code",
          llm_model: "sonnet/high",
        },
      });

      renderWithProviders(<ChatInputModel />);
      fireEvent.click(screen.getByTestId("chat-input-llm-model"));

      expect(
        screen.getByTestId("chat-input-acp-effort-option-default"),
      ).toBeInTheDocument();
      expect(
        screen.getByTestId("chat-input-acp-effort-option-low"),
      ).toBeInTheDocument();
      expect(
        screen.getByTestId("chat-input-acp-effort-option-max"),
      ).toBeInTheDocument();
      // The running effort ("high") is the one marked selected.
      expect(
        screen.getByTestId("chat-input-acp-effort-option-high"),
      ).toHaveClass("bg-[var(--oh-interactive-hover)]");
      expect(
        screen.getByTestId("chat-input-acp-effort-option-default"),
      ).not.toHaveClass("bg-[var(--oh-interactive-hover)]");
    });

    it("hides the effort section for a server with no recognized effort levels (gemini-cli)", () => {
      useActiveConversationMock.mockReturnValue({
        data: {
          conversation_id: "test-conversation-id",
          agent_kind: "acp",
          acp_server: "gemini-cli",
          llm_model: "gemini-2.5-pro",
        },
      });

      renderWithProviders(<ChatInputModel />);
      fireEvent.click(screen.getByTestId("chat-input-llm-model"));

      // Model rows are still offered...
      expect(
        screen.getByTestId("chat-input-acp-model-option-gemini-2.5-pro"),
      ).toBeInTheDocument();
      // ...but there is no effort section for this server.
      expect(
        screen.queryByTestId("chat-input-acp-effort-option-default"),
      ).not.toBeInTheDocument();
    });

    it("hides the effort section for a custom server with no live-reported efforts", () => {
      useActiveConversationMock.mockReturnValue({
        data: {
          conversation_id: "test-conversation-id",
          agent_kind: "acp",
          acp_server: "custom",
          llm_model: "my-model",
          acp_live_models: [{ id: "my-model", label: "my-model" }],
        },
      });

      renderWithProviders(<ChatInputModel />);
      fireEvent.click(screen.getByTestId("chat-input-llm-model"));

      expect(
        screen.getByTestId("chat-input-acp-model-option-my-model"),
      ).toBeInTheDocument();
      expect(
        screen.queryByTestId("chat-input-acp-effort-option-default"),
      ).not.toBeInTheDocument();
    });

    it("shows a live-reported effort section for a custom server that reports available efforts", () => {
      useActiveConversationMock.mockReturnValue({
        data: {
          conversation_id: "test-conversation-id",
          agent_kind: "acp",
          acp_server: "custom",
          llm_model: "my-model",
          acp_live_models: [{ id: "my-model", label: "my-model" }],
          acp_available_efforts: ["default", "turbo"],
        },
      });

      renderWithProviders(<ChatInputModel />);
      fireEvent.click(screen.getByTestId("chat-input-llm-model"));

      expect(
        screen.getByTestId("chat-input-acp-effort-option-default"),
      ).toBeInTheDocument();
      // "turbo" has no i18n key mirrored in Canvas's static map — the raw
      // value renders instead of guessing at a translation.
      const turboRow = screen.getByTestId("chat-input-acp-effort-option-turbo");
      expect(turboRow).toBeInTheDocument();
      expect(turboRow).toHaveTextContent("turbo");
    });

    it("live-switches the effort when a row is selected in an active ACP conversation", () => {
      useActiveConversationMock.mockReturnValue({
        data: {
          conversation_id: "test-conversation-id",
          agent_kind: "acp",
          acp_server: "claude-code",
          llm_model: "sonnet",
        },
      });

      renderWithProviders(<ChatInputModel />);
      fireEvent.click(screen.getByTestId("chat-input-llm-model"));
      fireEvent.click(screen.getByTestId("chat-input-acp-effort-option-high"));

      expect(switchAcpModelMutate).toHaveBeenCalledWith({
        conversationId: "test-conversation-id",
        model: "sonnet/high",
      });
      // Popover closes after a selection, same as a model pick.
      expect(
        screen.queryByTestId("chat-input-llm-model-popover"),
      ).not.toBeInTheDocument();
    });

    it("does not re-switch when selecting the already-current effort", () => {
      useActiveConversationMock.mockReturnValue({
        data: {
          conversation_id: "test-conversation-id",
          agent_kind: "acp",
          acp_server: "claude-code",
          llm_model: "sonnet/high",
        },
      });

      renderWithProviders(<ChatInputModel />);
      fireEvent.click(screen.getByTestId("chat-input-llm-model"));
      fireEvent.click(screen.getByTestId("chat-input-acp-effort-option-high"));

      expect(switchAcpModelMutate).not.toHaveBeenCalled();
    });

    it("persists the effort choice as the default (conversationId null) in the home ACP case", () => {
      useActiveConversationMock.mockReturnValue({ data: undefined });
      useSettingsMock.mockReturnValue({
        data: {
          agent_settings: {
            agent_kind: "acp",
            acp_server: "claude-code",
            acp_model: "sonnet",
          },
        },
      });

      renderWithProviders(<ChatInputModel />);
      fireEvent.click(screen.getByTestId("chat-input-llm-model"));
      fireEvent.click(screen.getByTestId("chat-input-acp-effort-option-max"));

      expect(switchAcpModelMutate).toHaveBeenCalledWith({
        conversationId: null,
        model: "sonnet/max",
      });
    });
  });
});
