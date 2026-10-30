import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ChatAddFileButton } from "#/components/features/chat/chat-add-file-button";
import { useCloseOnEscape } from "#/hooks/use-close-on-escape";
import { I18nKey } from "#/i18n/declaration";

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: undefined }),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: undefined }),
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
});
