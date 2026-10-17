import type { FormEvent } from "react";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import ProfilesService from "#/api/profiles-service/profiles-service.api";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { ChatInputLlmProfilePicker } from "#/components/features/chat/components/chat-input-llm-profile-picker";
import { ChatInputModelMenuContent } from "#/components/features/chat/components/chat-input-model";
import type { ChatInputModelState } from "#/hooks/use-chat-input-model-state";
import { ContextMenu } from "#/ui/context-menu";

const profiles = [
  {
    name: "Fast",
    model: "openai/gpt-4o-mini",
    base_url: null,
    api_key_set: true,
  },
  {
    name: "Smart",
    model: "anthropic/claude-opus",
    base_url: null,
    api_key_set: true,
  },
  {
    name: "Free",
    model: "openhands/deepseek-v4-flash",
    base_url: null,
    api_key_set: true,
  },
];

const acpModel: ChatInputModelState = {
  isAcpContext: true,
  displayModel: "Balanced",
  currentModelId: "vendor/alpha",
  selectedModelId: "vendor/alpha",
  availableAcpModels: [
    { id: "vendor/alpha", label: "Balanced" },
    { id: "vendor/beta", label: "Powerful" },
  ],
  agentDefaultLabel: "Agent default",
  showAcpPicker: true,
  switchConversationId: "search-conversation",
  destinationPath: "/settings/agents",
  destinationLabel: "Agent profiles",
};

async function openProfilePicker() {
  const user = userEvent.setup();
  renderWithProviders(<ChatInputLlmProfilePicker />, {
    navigation: { conversationId: null },
  });
  await user.click(await screen.findByTestId("chat-input-llm-profile"));
  return user;
}

describe("chat model search", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(ProfilesService, "listProfiles").mockResolvedValue({
      profiles,
      active_profile: "Fast",
    });
    vi.spyOn(AgentServerConversationService, "switchProfile").mockResolvedValue(
      undefined,
    );
    vi.spyOn(
      AgentServerConversationService,
      "switchAcpModel",
    ).mockResolvedValue(undefined);
  });

  // @spec MDS-001 — Search LLM profiles by name and model
  it.each([
    ["  fAsT  ", "Fast", "Smart"],
    ["ANTHROPIC/CLAUDE", "Smart", "Fast"],
    ["DeepSeek V4", "Free", "Fast"],
  ])(
    "filters profiles for %s without switching the model",
    async (query, visible, hidden) => {
      const user = await openProfilePicker();
      const search = screen.getByRole("searchbox");
      expect(search).toHaveFocus();

      await user.type(search, query);

      expect(
        screen.getByTestId(`chat-input-llm-profile-option-${visible}`),
      ).toBeInTheDocument();
      expect(
        screen.queryByTestId(`chat-input-llm-profile-option-${hidden}`),
      ).not.toBeInTheDocument();
      expect(
        AgentServerConversationService.switchProfile,
      ).not.toHaveBeenCalled();
      expect(screen.getByTestId("chat-input-llm-profile")).toHaveTextContent(
        "Fast",
      );
    },
  );

  // @spec MDS-003 — Empty results and search reset
  it("keeps settings available with no matches, clears the search, and resets on reopen", async () => {
    const user = await openProfilePicker();
    await user.type(screen.getByRole("searchbox"), "not-a-model");

    expect(screen.getByRole("status")).toHaveTextContent("COMMON$NO_RESULTS");
    expect(screen.getByRole("link")).toHaveAttribute("href", "/settings/llm");

    await user.click(screen.getByRole("button", { name: "MCP$SEARCH_CLEAR" }));
    expect(screen.getByRole("searchbox")).toHaveFocus();
    expect(
      screen.getByTestId("chat-input-llm-profile-option-Smart"),
    ).toBeInTheDocument();

    await user.type(screen.getByRole("searchbox"), "Smart");
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("searchbox")).not.toBeInTheDocument();
    expect(screen.getByTestId("chat-input-llm-profile")).toHaveFocus();

    await user.click(screen.getByTestId("chat-input-llm-profile"));
    expect(screen.getByRole("searchbox")).toHaveValue("");
    expect(
      screen.getByTestId("chat-input-llm-profile-option-Fast"),
    ).toBeInTheDocument();
  });

  // @spec MDS-004 — Keyboard search and selection
  it("selects a filtered profile with the keyboard through the existing service", async () => {
    const user = await openProfilePicker();
    await user.type(screen.getByRole("searchbox"), "claude");
    await user.keyboard("{ArrowDown}{Enter}");

    await waitFor(() =>
      expect(AgentServerConversationService.switchProfile).toHaveBeenCalledWith(
        null,
        "Smart",
      ),
    );
    expect(screen.queryByRole("searchbox")).not.toBeInTheDocument();
  });

  it("does not submit the surrounding chat form when Enter is pressed in search", async () => {
    const submit = vi.fn((event: FormEvent) => event.preventDefault());
    const keyDown = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <form onSubmit={submit} onKeyDown={keyDown}>
        <ChatInputLlmProfilePicker />
      </form>,
      { navigation: { conversationId: null } },
    );
    await user.click(await screen.findByTestId("chat-input-llm-profile"));
    await user.type(screen.getByRole("searchbox"), "Smart");
    keyDown.mockClear();

    await user.keyboard("{Enter}");

    expect(submit).not.toHaveBeenCalled();
    expect(keyDown).not.toHaveBeenCalled();
    expect(AgentServerConversationService.switchProfile).not.toHaveBeenCalled();
  });

  // @spec MDS-002 — Search ACP model labels and IDs
  it("filters ACP models by label and ID while preserving live selection", async () => {
    const user = userEvent.setup();
    const close = vi.fn();
    renderWithProviders(
      <ContextMenu>
        <ChatInputModelMenuContent model={acpModel} onClose={close} />
      </ContextMenu>,
    );
    const search = screen.getByRole("searchbox");
    await user.type(search, "  POWERFUL  ");
    expect(
      screen.queryByTestId("chat-input-acp-model-option-vendor/alpha"),
    ).not.toBeInTheDocument();
    expect(
      screen.getByTestId("chat-input-acp-model-option-vendor/beta"),
    ).toBeInTheDocument();

    await user.clear(search);
    await user.type(search, "vendor/beta");
    await user.click(
      screen.getByTestId("chat-input-acp-model-option-vendor/beta"),
    );

    await waitFor(() =>
      expect(
        AgentServerConversationService.switchAcpModel,
      ).toHaveBeenCalledWith("search-conversation", "vendor/beta"),
    );
    expect(close).toHaveBeenCalledOnce();
  });

  it("includes the ACP agent default in search and shows no-results feedback", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ContextMenu>
        <ChatInputModelMenuContent model={acpModel} onClose={vi.fn()} />
      </ContextMenu>,
    );
    await user.type(screen.getByRole("searchbox"), "default");
    expect(
      screen.getByTestId("chat-input-acp-model-option-agent-default"),
    ).toBeInTheDocument();
    await user.type(screen.getByRole("searchbox"), "missing");
    expect(screen.getByRole("status")).toHaveTextContent("COMMON$NO_RESULTS");
    expect(screen.getByRole("link")).toHaveAttribute(
      "href",
      "/settings/agents",
    );
  });

  // @spec MDS-005 — Read-only model surfaces
  it("does not offer search on read-only profile or model menus", async () => {
    const user = userEvent.setup();
    const { unmount } = renderWithProviders(<ChatInputLlmProfilePicker />, {
      navigation: { conversationId: "task-provisioning" },
    });
    await user.click(await screen.findByTestId("chat-input-llm-profile"));
    expect(screen.queryByRole("searchbox")).not.toBeInTheDocument();
    unmount();

    renderWithProviders(
      <ContextMenu>
        <ChatInputModelMenuContent
          model={{ ...acpModel, showAcpPicker: false }}
          onClose={vi.fn()}
        />
      </ContextMenu>,
    );
    expect(screen.queryByRole("searchbox")).not.toBeInTheDocument();
  });
});
