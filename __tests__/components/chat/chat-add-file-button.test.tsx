import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ChatAddFileButton } from "#/components/features/chat/chat-add-file-button";
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

describe("ChatAddFileButton", () => {
  it("uses the translated aria-label for the plus menu trigger", () => {
    render(<ChatAddFileButton handleFileIconClick={vi.fn()} />);

    const button = screen.getByTestId("chat-plus-button");
    expect(button).toHaveAttribute("aria-label", I18nKey.CHAT_INTERFACE$PLUS_MENU);
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

  it("does not open the menu or invoke the file handler when disabled", async () => {
    const user = userEvent.setup();
    const handleFileIconClick = vi.fn();

    render(
      <ChatAddFileButton
        handleFileIconClick={handleFileIconClick}
        disabled
      />,
    );

    const button = screen.getByTestId("chat-plus-button");
    expect(button).toBeDisabled();

    await user.click(button);
    expect(screen.queryByTestId("tools-context-menu")).not.toBeInTheDocument();
    expect(handleFileIconClick).not.toHaveBeenCalled();
  });
});
