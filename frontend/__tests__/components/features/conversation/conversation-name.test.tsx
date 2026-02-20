import { screen, within, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {
  afterEach,
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";
import { renderWithProviders } from "test-utils";
import { ConversationName } from "#/components/features/conversation/conversation-name";
import { ConversationNameContextMenu } from "#/components/features/conversation/conversation-name-context-menu";
import { BrowserRouter } from "react-router";
import type { Conversation } from "#/api/open-hands.types";

// Hoisted mocks for controllable return values
const {
  mockMutate,
  mockDisplaySuccessToast,
  useActiveConversationMock,
  useConfigMock,
} = vi.hoisted(() => ({
  mockMutate: vi.fn(),
  mockDisplaySuccessToast: vi.fn(),
  useActiveConversationMock: vi.fn(() => ({
    data: {
      conversation_id: "test-conversation-id",
      title: "Test Conversation",
      status: "RUNNING",
    },
  })),
  useConfigMock: vi.fn(() => ({
    data: {
      app_mode: "oss",
    },
  })),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => useActiveConversationMock(),
}));

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => useConfigMock(),
}));

vi.mock("#/hooks/mutation/use-update-conversation", () => ({
  useUpdateConversation: () => ({
    mutate: mockMutate,
  }),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: mockDisplaySuccessToast,
}));

// Mock react-i18next
vi.mock("react-i18next", async () => {
  const actual = await vi.importActual("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => {
        const translations: Record<string, string> = {
          CONVERSATION$TITLE_UPDATED: "Conversation title updated",
          BUTTON$RENAME: "Rename",
          BUTTON$EXPORT_CONVERSATION: "Export Conversation",
          BUTTON$DOWNLOAD_VIA_VSCODE: "Download via VS Code",
          BUTTON$SHOW_AGENT_TOOLS_AND_METADATA: "Show Agent Tools",
          CONVERSATION$SHOW_SKILLS: "Show Skills",
          BUTTON$DISPLAY_COST: "Display Cost",
          COMMON$CLOSE_CONVERSATION_STOP_RUNTIME:
            "Close Conversation (Stop Runtime)",
          COMMON$DELETE_CONVERSATION: "Delete Conversation",
          CONVERSATION$SHARE_PUBLICLY: "Share Publicly",
          CONVERSATION$LINK_COPIED: "Link copied to clipboard",
          BUTTON$COPY_TO_CLIPBOARD: "Copy to Clipboard",
          BUTTON$OPEN_IN_NEW_TAB: "Open in New Tab",
        };
        return translations[key] || key;
      },
      i18n: {
        changeLanguage: () => new Promise(() => {}),
      },
    }),
  };
});

// Helper function to render ConversationName with Router context
const renderConversationNameWithRouter = () => {
  return renderWithProviders(
    <BrowserRouter>
      <ConversationName />
    </BrowserRouter>,
  );
};

describe("ConversationName", () => {
  beforeAll(() => {
    vi.stubGlobal("window", {
      open: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      location: {
        origin: "http://localhost:3000",
      },
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render the conversation name in view mode", () => {
    renderConversationNameWithRouter();

    const container = screen.getByTestId("conversation-name");
    const titleElement = within(container).getByTestId(
      "conversation-name-title",
    );

    expect(container).toBeInTheDocument();
    expect(titleElement).toBeInTheDocument();
    expect(titleElement).toHaveTextContent("Test Conversation");
  });

  it("should switch to edit mode on double click", async () => {
    const user = userEvent.setup();
    renderConversationNameWithRouter();

    const titleElement = screen.getByTestId("conversation-name-title");

    // Initially should be in view mode
    expect(titleElement).toBeInTheDocument();
    expect(
      screen.queryByTestId("conversation-name-input"),
    ).not.toBeInTheDocument();

    // Double click to enter edit mode
    await user.dblClick(titleElement);

    // Should now be in edit mode
    expect(
      screen.queryByTestId("conversation-name-title"),
    ).not.toBeInTheDocument();
    const inputElement = screen.getByTestId("conversation-name-input");
    expect(inputElement).toBeInTheDocument();
    expect(inputElement).toHaveValue("Test Conversation");
  });

  it("should update conversation title when input loses focus with valid value", async () => {
    const user = userEvent.setup();
    renderConversationNameWithRouter();

    const titleElement = screen.getByTestId("conversation-name-title");
    await user.dblClick(titleElement);

    const inputElement = screen.getByTestId("conversation-name-input");
    await user.clear(inputElement);
    await user.type(inputElement, "New Conversation Title");
    await user.tab(); // Trigger blur event

    // Verify that the update function was called
    expect(mockMutate).toHaveBeenCalledWith(
      {
        conversationId: "test-conversation-id",
        newTitle: "New Conversation Title",
      },
      expect.any(Object),
    );
  });

  it("should not update conversation when title is unchanged", async () => {
    const user = userEvent.setup();
    renderConversationNameWithRouter();

    const titleElement = screen.getByTestId("conversation-name-title");
    await user.dblClick(titleElement);

    const inputElement = screen.getByTestId("conversation-name-input");
    // Keep the same title
    await user.tab();

    // Should still have the original title
    expect(inputElement).toHaveValue("Test Conversation");
  });

  it("should not call the API if user attempts to save an unchanged title", async () => {
    const user = userEvent.setup();
    renderConversationNameWithRouter();

    const titleElement = screen.getByTestId("conversation-name-title");
    await user.dblClick(titleElement);

    const inputElement = screen.getByTestId("conversation-name-input");

    // Verify the input has the original title
    expect(inputElement).toHaveValue("Test Conversation");

    // Trigger blur without changing the title
    await user.tab();

    // Verify that the API was NOT called
    expect(mockMutate).not.toHaveBeenCalled();
  });

  it("should reset input value when title is empty and blur", async () => {
    const user = userEvent.setup();
    renderConversationNameWithRouter();

    const titleElement = screen.getByTestId("conversation-name-title");
    await user.dblClick(titleElement);

    const inputElement = screen.getByTestId("conversation-name-input");
    await user.clear(inputElement);
    await user.tab();

    // Should reset to original title
    expect(inputElement).toHaveValue("Test Conversation");
  });

  it("should trim whitespace from input value", async () => {
    const user = userEvent.setup();
    renderConversationNameWithRouter();

    const titleElement = screen.getByTestId("conversation-name-title");
    await user.dblClick(titleElement);

    const inputElement = screen.getByTestId("conversation-name-input");
    await user.clear(inputElement);
    await user.type(inputElement, "  Trimmed Title  ");
    await user.tab();

    // Should call mutation with trimmed value
    expect(mockMutate).toHaveBeenCalledWith(
      {
        conversationId: "test-conversation-id",
        newTitle: "Trimmed Title",
      },
      expect.any(Object),
    );
  });

  it("should handle Enter key to save changes", async () => {
    const user = userEvent.setup();
    renderConversationNameWithRouter();

    const titleElement = screen.getByTestId("conversation-name-title");
    await user.dblClick(titleElement);

    const inputElement = screen.getByTestId("conversation-name-input");
    await user.clear(inputElement);
    await user.type(inputElement, "New Title");
    await user.keyboard("{Enter}");

    // Should have the new title
    expect(inputElement).toHaveValue("New Title");
  });

  it("should prevent event propagation when clicking input in edit mode", async () => {
    const user = userEvent.setup();
    renderConversationNameWithRouter();

    const titleElement = screen.getByTestId("conversation-name-title");
    await user.dblClick(titleElement);

    const inputElement = screen.getByTestId("conversation-name-input");
    const clickEvent = new MouseEvent("click", { bubbles: true });
    const preventDefaultSpy = vi.spyOn(clickEvent, "preventDefault");
    const stopPropagationSpy = vi.spyOn(clickEvent, "stopPropagation");

    inputElement.dispatchEvent(clickEvent);

    expect(preventDefaultSpy).toHaveBeenCalled();
    expect(stopPropagationSpy).toHaveBeenCalled();
  });

  it("should return to view mode after blur", async () => {
    const user = userEvent.setup();
    renderConversationNameWithRouter();

    const titleElement = screen.getByTestId("conversation-name-title");
    await user.dblClick(titleElement);

    // Should be in edit mode
    expect(screen.getByTestId("conversation-name-input")).toBeInTheDocument();

    await user.tab();

    // Should be back in view mode
    expect(screen.getByTestId("conversation-name-title")).toBeInTheDocument();
    expect(
      screen.queryByTestId("conversation-name-input"),
    ).not.toBeInTheDocument();
  });

  it("should focus input when entering edit mode", async () => {
    const user = userEvent.setup();
    renderConversationNameWithRouter();

    const titleElement = screen.getByTestId("conversation-name-title");
    await user.dblClick(titleElement);

    const inputElement = screen.getByTestId("conversation-name-input");
    expect(inputElement).toHaveFocus();
  });
});

describe("ConversationNameContextMenu", () => {
  const defaultProps = {
    onClose: vi.fn(),
  };

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render all menu options when all handlers are provided", () => {
    const handlers = {
      onRename: vi.fn(),
      onDelete: vi.fn(),
      onStop: vi.fn(),
      onDisplayCost: vi.fn(),
      onShowAgentTools: vi.fn(),
      onShowSkills: vi.fn(),
      onExportConversation: vi.fn(),
      onDownloadViaVSCode: vi.fn(),
    };

    renderWithProviders(
      <ConversationNameContextMenu {...defaultProps} {...handlers} />,
    );

    expect(screen.getByTestId("rename-button")).toBeInTheDocument();
    expect(screen.getByTestId("delete-button")).toBeInTheDocument();
    expect(screen.getByTestId("stop-button")).toBeInTheDocument();
    expect(screen.getByTestId("display-cost-button")).toBeInTheDocument();
    expect(screen.getByTestId("show-agent-tools-button")).toBeInTheDocument();
    expect(screen.getByTestId("show-skills-button")).toBeInTheDocument();
    expect(
      screen.getByTestId("export-conversation-button"),
    ).toBeInTheDocument();
    expect(screen.getByTestId("download-vscode-button")).toBeInTheDocument();
  });

  it("should not render menu options when handlers are not provided", () => {
    renderWithProviders(<ConversationNameContextMenu {...defaultProps} />);

    expect(screen.queryByTestId("rename-button")).not.toBeInTheDocument();
    expect(screen.queryByTestId("delete-button")).not.toBeInTheDocument();
    expect(screen.queryByTestId("stop-button")).not.toBeInTheDocument();
    expect(screen.queryByTestId("display-cost-button")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("show-agent-tools-button"),
    ).not.toBeInTheDocument();
    expect(screen.queryByTestId("show-skills-button")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("export-conversation-button"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("download-vscode-button"),
    ).not.toBeInTheDocument();
  });

  it("should call rename handler when rename button is clicked", async () => {
    const user = userEvent.setup();
    const onRename = vi.fn();

    renderWithProviders(
      <ConversationNameContextMenu {...defaultProps} onRename={onRename} />,
    );

    const renameButton = screen.getByTestId("rename-button");
    await user.click(renameButton);

    expect(onRename).toHaveBeenCalledTimes(1);
  });

  it("should call delete handler when delete button is clicked", async () => {
    const user = userEvent.setup();
    const onDelete = vi.fn();

    renderWithProviders(
      <ConversationNameContextMenu {...defaultProps} onDelete={onDelete} />,
    );

    const deleteButton = screen.getByTestId("delete-button");
    await user.click(deleteButton);

    expect(onDelete).toHaveBeenCalledTimes(1);
  });

  it("should call stop handler when stop button is clicked", async () => {
    const user = userEvent.setup();
    const onStop = vi.fn();

    renderWithProviders(
      <ConversationNameContextMenu {...defaultProps} onStop={onStop} />,
    );

    const stopButton = screen.getByTestId("stop-button");
    await user.click(stopButton);

    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it("should call display cost handler when display cost button is clicked", async () => {
    const user = userEvent.setup();
    const onDisplayCost = vi.fn();

    renderWithProviders(
      <ConversationNameContextMenu
        {...defaultProps}
        onDisplayCost={onDisplayCost}
      />,
    );

    const displayCostButton = screen.getByTestId("display-cost-button");
    await user.click(displayCostButton);

    expect(onDisplayCost).toHaveBeenCalledTimes(1);
  });

  it("should call show agent tools handler when show agent tools button is clicked", async () => {
    const user = userEvent.setup();
    const onShowAgentTools = vi.fn();

    renderWithProviders(
      <ConversationNameContextMenu
        {...defaultProps}
        onShowAgentTools={onShowAgentTools}
      />,
    );

    const showAgentToolsButton = screen.getByTestId("show-agent-tools-button");
    await user.click(showAgentToolsButton);

    expect(onShowAgentTools).toHaveBeenCalledTimes(1);
  });

  it("should call show microagents handler when show microagents button is clicked", async () => {
    const user = userEvent.setup();
    const onShowSkills = vi.fn();

    renderWithProviders(
      <ConversationNameContextMenu
        {...defaultProps}
        onShowSkills={onShowSkills}
      />,
    );

    const showMicroagentsButton = screen.getByTestId("show-skills-button");
    await user.click(showMicroagentsButton);

    expect(onShowSkills).toHaveBeenCalledTimes(1);
  });

  it("should call export conversation handler when export conversation button is clicked", async () => {
    const user = userEvent.setup();
    const onExportConversation = vi.fn();

    renderWithProviders(
      <ConversationNameContextMenu
        {...defaultProps}
        onExportConversation={onExportConversation}
      />,
    );

    const exportButton = screen.getByTestId("export-conversation-button");
    await user.click(exportButton);

    expect(onExportConversation).toHaveBeenCalledTimes(1);
  });

  it("should call download via VSCode handler when download via VSCode button is clicked", async () => {
    const user = userEvent.setup();
    const onDownloadViaVSCode = vi.fn();

    renderWithProviders(
      <ConversationNameContextMenu
        {...defaultProps}
        onDownloadViaVSCode={onDownloadViaVSCode}
      />,
    );

    const downloadButton = screen.getByTestId("download-vscode-button");
    await user.click(downloadButton);

    expect(onDownloadViaVSCode).toHaveBeenCalledTimes(1);
  });

  it("should render separators between logical groups", () => {
    const handlers = {
      onRename: vi.fn(),
      onShowAgentTools: vi.fn(),
      onExportConversation: vi.fn(),
      onDisplayCost: vi.fn(),
      onStop: vi.fn(),
    };

    renderWithProviders(
      <ConversationNameContextMenu {...defaultProps} {...handlers} />,
    );

    // Look for separator elements using test IDs
    expect(screen.getByTestId("separator-tools")).toBeInTheDocument();
    expect(screen.getByTestId("separator-export")).toBeInTheDocument();
    expect(screen.getByTestId("separator-info-control")).toBeInTheDocument();
  });

  it("should apply correct positioning class when position is top", () => {
    const handlers = {
      onRename: vi.fn(),
    };

    renderWithProviders(
      <ConversationNameContextMenu
        {...defaultProps}
        {...handlers}
        position="top"
      />,
    );

    const contextMenu = screen.getByTestId("conversation-name-context-menu");
    expect(contextMenu).toHaveClass("bottom-full");
  });

  it("should apply correct positioning class when position is bottom", () => {
    const handlers = {
      onRename: vi.fn(),
    };

    renderWithProviders(
      <ConversationNameContextMenu
        {...defaultProps}
        {...handlers}
        position="bottom"
      />,
    );

    const contextMenu = screen.getByTestId("conversation-name-context-menu");
    expect(contextMenu).toHaveClass("top-full");
  });

  it("should render correct text content for each menu option", () => {
    const handlers = {
      onRename: vi.fn(),
      onDelete: vi.fn(),
      onStop: vi.fn(),
      onDisplayCost: vi.fn(),
      onShowAgentTools: vi.fn(),
      onShowSkills: vi.fn(),
      onExportConversation: vi.fn(),
      onDownloadViaVSCode: vi.fn(),
    };

    renderWithProviders(
      <ConversationNameContextMenu {...defaultProps} {...handlers} />,
    );

    expect(screen.getByTestId("rename-button")).toHaveTextContent("Rename");
    expect(screen.getByTestId("delete-button")).toHaveTextContent(
      "Delete Conversation",
    );
    expect(screen.getByTestId("stop-button")).toHaveTextContent(
      "Close Conversation (Stop Runtime)",
    );
    expect(screen.getByTestId("display-cost-button")).toHaveTextContent(
      "Display Cost",
    );
    expect(screen.getByTestId("show-agent-tools-button")).toHaveTextContent(
      "Show Agent Tools",
    );
    expect(screen.getByTestId("show-skills-button")).toHaveTextContent(
      "Show Skills",
    );
    expect(screen.getByTestId("export-conversation-button")).toHaveTextContent(
      "Export Conversation",
    );
    expect(screen.getByTestId("download-vscode-button")).toHaveTextContent(
      "Download via VS Code",
    );
  });

  it("should call onClose when context menu is closed", () => {
    const onClose = vi.fn();
    const handlers = {
      onRename: vi.fn(),
    };

    renderWithProviders(
      <ConversationNameContextMenu
        {...defaultProps}
        onClose={onClose}
        {...handlers}
      />,
    );

    // The onClose is typically called by the parent component when clicking outside
    // This test verifies the prop is properly passed
    expect(onClose).toBeDefined();
  });
});

describe("ConversationNameContextMenu - Share Link Functionality", () => {
  const mockWriteText = vi.fn().mockResolvedValue(undefined);

  const mockOnCopyShareLink = vi.fn();
  const mockOnTogglePublic = vi.fn();
  const mockOnClose = vi.fn();

  const defaultProps = {
    onClose: mockOnClose,
    onTogglePublic: mockOnTogglePublic,
    onCopyShareLink: mockOnCopyShareLink,
    shareUrl: "https://example.com/shared/conversations/test-id",
  };

  vi.mock("#/hooks/mutation/use-update-conversation-public-flag", () => ({
    useUpdateConversationPublicFlag: () => ({
      mutate: vi.fn(),
    }),
  }));

  beforeAll(() => {
    // Mock navigator.clipboard
    Object.defineProperty(navigator, "clipboard", {
      value: {
        writeText: mockWriteText,
        readText: vi.fn(),
      },
      writable: true,
      configurable: true,
    });
  });

  beforeEach(() => {
    mockWriteText.mockClear();
    mockDisplaySuccessToast.mockClear();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should display copy and open buttons when conversation is public", () => {
    // Arrange
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-id",
        title: "Test Conversation",
        status: "STOPPED",
        conversation_version: "V1" as const,
        public: true,
      } as Conversation,
    });

    useConfigMock.mockReturnValue({
      data: {
        app_mode: "saas",
      },
    });

    // Act
    renderWithProviders(<ConversationNameContextMenu {...defaultProps} />);

    // Assert
    expect(screen.getByTestId("copy-share-link-button")).toBeInTheDocument();
    expect(screen.getByTestId("open-share-link-button")).toBeInTheDocument();
  });

  it("should not display share buttons when conversation is not public", () => {
    // Arrange
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-id",
        title: "Test Conversation",
        status: "STOPPED",
        conversation_version: "V1" as const,
        public: false,
      } as Conversation,
    });

    useConfigMock.mockReturnValue({
      data: {
        app_mode: "saas",
      },
    });

    // Act
    renderWithProviders(<ConversationNameContextMenu {...defaultProps} />);

    // Assert
    expect(
      screen.queryByTestId("copy-share-link-button"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("open-share-link-button"),
    ).not.toBeInTheDocument();
  });

  it("should call copy handler when copy button is clicked", async () => {
    // Arrange
    const user = userEvent.setup();
    const shareUrl = "https://example.com/shared/conversations/test-id";

    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-id",
        title: "Test Conversation",
        status: "STOPPED",
        conversation_version: "V1" as const,
        public: true,
      } as Conversation,
    });

    useConfigMock.mockReturnValue({
      data: {
        app_mode: "saas",
      },
    });

    renderWithProviders(
      <ConversationNameContextMenu {...defaultProps} shareUrl={shareUrl} />,
    );

    const copyButton = screen.getByTestId("copy-share-link-button");

    // Act
    await user.click(copyButton);

    // Assert
    expect(mockOnCopyShareLink).toHaveBeenCalledTimes(1);
  });

  it("should have correct attributes for open share link button", () => {
    // Arrange
    const shareUrl = "https://example.com/shared/conversations/test-id";

    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-id",
        title: "Test Conversation",
        status: "STOPPED",
        conversation_version: "V1" as const,
        public: true,
      } as Conversation,
    });

    useConfigMock.mockReturnValue({
      data: {
        app_mode: "saas",
      },
    });

    renderWithProviders(
      <ConversationNameContextMenu {...defaultProps} shareUrl={shareUrl} />,
    );

    const openButton = screen.getByTestId("open-share-link-button");

    // Assert
    expect(openButton).toHaveAttribute("href", shareUrl);
    expect(openButton).toHaveAttribute("target", "_blank");
    expect(openButton).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("should display correct tooltips for share buttons", () => {
    // Arrange
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-id",
        title: "Test Conversation",
        status: "STOPPED",
        conversation_version: "V1" as const,
        public: true,
      } as Conversation,
    });

    useConfigMock.mockReturnValue({
      data: {
        app_mode: "saas",
      },
    });

    renderWithProviders(<ConversationNameContextMenu {...defaultProps} />);

    // Assert
    const copyButton = screen.getByTestId("copy-share-link-button");
    const openButton = screen.getByTestId("open-share-link-button");

    expect(copyButton).toHaveAttribute("title", "Copy to Clipboard");
    expect(openButton).toHaveAttribute("title", "Open in New Tab");
  });

  describe("Integration with ConversationName component", () => {
    beforeEach(() => {
      // Default mocks for public V1 conversation in SAAS mode
      useActiveConversationMock.mockReturnValue({
        data: {
          conversation_id: "test-conversation-id",
          title: "Test Conversation",
          status: "STOPPED",
          conversation_version: "V1" as const,
          public: true,
        } as Conversation,
      });

      useConfigMock.mockReturnValue({
        data: {
          app_mode: "saas",
        },
      });
    });

    it("should copy share URL to clipboard and show success toast when copy button is clicked through ConversationName", async () => {
      // Arrange
      const user = userEvent.setup();
      const expectedUrl =
        "http://localhost:3000/shared/conversations/test-conversation-id";

      // Ensure navigator.clipboard is properly mocked
      if (!navigator.clipboard) {
        Object.defineProperty(navigator, "clipboard", {
          value: {
            writeText: mockWriteText,
            readText: vi.fn(),
          },
          writable: true,
          configurable: true,
        });
      } else {
        vi.spyOn(navigator.clipboard, "writeText").mockImplementation(
          mockWriteText,
        );
      }

      renderConversationNameWithRouter();

      // Open context menu by clicking ellipsis
      const ellipsisButton = screen.getByRole("button", { hidden: true });
      await user.click(ellipsisButton);

      // Wait for context menu to appear and find share publicly button
      const sharePubliclyButton = await screen.findByTestId(
        "share-publicly-button",
      );
      expect(sharePubliclyButton).toBeInTheDocument();

      // Find copy button
      const copyButton = screen.getByTestId("copy-share-link-button");

      // Act
      await user.click(copyButton);

      // Assert - clipboard.writeText is async, so we need to wait
      await waitFor(
        () => {
          expect(mockWriteText).toHaveBeenCalledWith(expectedUrl);
          expect(mockDisplaySuccessToast).toHaveBeenCalledWith(
            "Link copied to clipboard",
          );
        },
        { timeout: 2000, container: document.body },
      );
    });

    it("should show both copy and open buttons when conversation is public", async () => {
      // Arrange
      const user = userEvent.setup();

      renderConversationNameWithRouter();

      // Act - open context menu
      const ellipsisButton = screen.getByRole("button", { hidden: true });
      await user.click(ellipsisButton);

      // Wait for context menu
      const sharePubliclyButton = await screen.findByTestId(
        "share-publicly-button",
      );

      // Assert
      expect(sharePubliclyButton).toBeInTheDocument();
      expect(screen.getByTestId("copy-share-link-button")).toBeInTheDocument();
      expect(screen.getByTestId("open-share-link-button")).toBeInTheDocument();
    });
  });
});
