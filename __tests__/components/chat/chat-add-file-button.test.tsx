import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ChatAddFileButton } from "#/components/features/chat/chat-add-file-button";
import { useCloseOnEscape } from "#/hooks/use-close-on-escape";
import { I18nKey } from "#/i18n/declaration";

const state = vi.hoisted(() => ({
  conversation: undefined as
    | { id?: string; agent_kind?: string; tags?: Record<string, string> }
    | undefined,
  conversationId: undefined as string | undefined,
  canConfigure: false,
  canInstall: false,
  canRoute: false,
  projectsEnabled: false,
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: state.conversation }),
}));

vi.mock("#/hooks/query/use-projects", () => ({
  useProjectsEnabled: () => state.projectsEnabled,
}));

// This isolated menu test has no QueryClient; capability fetching is covered by integration tests.
vi.mock("#/hooks/query/use-conversation-configuration-capability", () => ({
  useConversationConfigurationCapability: () => ({ data: state.canConfigure }),
  useConversationMcpCapabilities: () => ({ canInstall: state.canInstall }),
  useConversationLlmCapabilities: () => ({ canRoute: state.canRoute }),
}));

vi.mock("#/components/features/projects/project-file-picker", () => ({
  ProjectFilePicker: ({ projectId }: { projectId: string }) => (
    <div data-testid="project-file-picker">{projectId}</div>
  ),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: state.conversationId }),
}));

vi.mock("#/hooks/use-conversation-name-context-menu", () => ({
  useConversationNameContextMenu: () => ({
    handleShowAgentTools: vi.fn(),
    handleShowSkills: vi.fn(),
    handleShowHooks: vi.fn(),
    systemModalVisible: false,
    setSystemModalVisible: vi.fn(),
    skillsModalVisible: false,
    setSkillsModalVisible: vi.fn(),
    hooksModalVisible: false,
    setHooksModalVisible: vi.fn(),
    systemMessage: null,
    shouldShowAgentTools: true,
    shouldShowHooks: false,
  }),
}));

vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => ({ providers: [] }),
}));

// Stands in for a layer opened over the tools menu without an outside click,
// such as the command menu opened with Ctrl+K.
function LayerOverMenu({
  isOpen,
  onClose,
}: {
  isOpen: boolean;
  onClose: () => void;
}) {
  useCloseOnEscape(isOpen, onClose);
  return null;
}

describe("ChatAddFileButton", () => {
  beforeEach(() => {
    state.conversation = undefined;
    state.conversationId = undefined;
    state.canConfigure = false;
    state.canInstall = false;
    state.canRoute = false;
    state.projectsEnabled = false;
  });

  it("offers additions only for a supported live OpenHands conversation", async () => {
    state.conversationId = "11111111-1111-4111-8111-111111111111";
    state.conversation = { id: state.conversationId };
    state.canConfigure = true;
    state.canInstall = true;
    state.canRoute = true;
    const user = userEvent.setup();
    render(<ChatAddFileButton handleFileIconClick={vi.fn()} />);

    await user.click(screen.getByTestId("chat-plus-button"));
    expect(
      screen.getByTestId("conversation-add-secret-button"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("conversation-add-mcp-button"),
    ).toBeInTheDocument();
    expect(
      screen.getByTestId("conversation-add-llm-button"),
    ).toBeInTheDocument();
  });

  it("does not offer MCP or model additions for ACP conversations", async () => {
    state.conversationId = "11111111-1111-4111-8111-111111111111";
    state.conversation = { id: state.conversationId, agent_kind: "acp" };
    state.canConfigure = true;
    state.canInstall = true;
    state.canRoute = true;
    const user = userEvent.setup();
    render(<ChatAddFileButton handleFileIconClick={vi.fn()} />);

    await user.click(screen.getByTestId("chat-plus-button"));
    expect(
      screen.getByTestId("conversation-add-secret-button"),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("conversation-add-mcp-button"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("conversation-add-llm-button"),
    ).not.toBeInTheDocument();
  });

  it("uses the translated aria-label for the plus menu trigger", () => {
    render(<ChatAddFileButton handleFileIconClick={vi.fn()} />);

    const button = screen.getByTestId("chat-plus-button");
    expect(button).toHaveAttribute(
      "aria-label",
      I18nKey.CHAT_INTERFACE$PLUS_MENU,
    );
    expect(button).toHaveAttribute("aria-haspopup", "menu");
  });

  it("opens the tools menu and invokes handleFileIconClick from the footer item", async () => {
    const user = userEvent.setup();
    const handleFileIconClick = vi.fn();

    render(<ChatAddFileButton handleFileIconClick={handleFileIconClick} />);

    await user.click(screen.getByTestId("chat-plus-button"));
    expect(screen.getByTestId("tools-context-menu")).toBeInTheDocument();

    await user.click(screen.getByTestId("add-files-and-images-button"));
    expect(handleFileIconClick).toHaveBeenCalledTimes(1);
  });

  it.each(["chat-plus-button", "macros-button"])(
    "closes the tools menu with Escape from %s and returns focus to the + button",
    async (focusedTestId) => {
      const user = userEvent.setup();
      render(<ChatAddFileButton handleFileIconClick={vi.fn()} />);
      const trigger = screen.getByTestId("chat-plus-button");

      await user.click(trigger);
      screen.getByTestId(focusedTestId).focus();
      await user.keyboard("{Escape}");

      expect(
        screen.queryByTestId("tools-context-menu"),
      ).not.toBeInTheDocument();
      expect(trigger).toHaveAttribute("aria-expanded", "false");
      expect(trigger).toHaveFocus();
    },
  );

  it("leaves the tools menu open while a layer opened over it takes Escape", async () => {
    const user = userEvent.setup();
    const closeLayer = vi.fn();
    const renderWithLayer = (isLayerOpen: boolean) => (
      <>
        <ChatAddFileButton handleFileIconClick={vi.fn()} />
        <LayerOverMenu isOpen={isLayerOpen} onClose={closeLayer} />
      </>
    );
    const { rerender } = render(renderWithLayer(false));
    await user.click(screen.getByTestId("chat-plus-button"));
    rerender(renderWithLayer(true));

    await user.keyboard("{Escape}");
    expect(closeLayer).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("tools-context-menu")).toBeInTheDocument();

    // Once the layer above has closed, the next Escape reaches the menu.
    rerender(renderWithLayer(false));
    await user.keyboard("{Escape}");
    expect(screen.queryByTestId("tools-context-menu")).not.toBeInTheDocument();
  });

  it("does not open the menu or invoke the file handler when disabled", async () => {
    const user = userEvent.setup();
    const handleFileIconClick = vi.fn();

    render(
      <ChatAddFileButton handleFileIconClick={handleFileIconClick} disabled />,
    );

    const button = screen.getByTestId("chat-plus-button");
    expect(button).toBeDisabled();

    await user.click(button);
    expect(screen.queryByTestId("tools-context-menu")).not.toBeInTheDocument();
    expect(handleFileIconClick).not.toHaveBeenCalled();
  });

  it.each([
    [true, { project: "aa01" }, true],
    [false, { project: "aa01" }, false],
    [true, {}, false],
  ])(
    "with Projects %s and tags %j, offers a project file: %s",
    async (enabled, tags, offered) => {
      state.projectsEnabled = enabled;
      state.conversation = { tags };
      const user = userEvent.setup();
      render(<ChatAddFileButton handleFileIconClick={vi.fn()} />);

      await user.click(screen.getByTestId("chat-plus-button"));
      const item = screen.queryByTestId("add-project-file-button");
      expect(!!item).toBe(offered);
      if (!item) return;

      await user.click(item);
      expect(screen.getByTestId("project-file-picker")).toHaveTextContent(
        "aa01",
      );
      expect(screen.queryByTestId("tools-context-menu")).toBeNull();
    },
  );
});
