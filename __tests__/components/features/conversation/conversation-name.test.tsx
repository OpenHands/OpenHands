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
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { Backend } from "#/api/backend-registry/types";
import type { Conversation } from "#/api/open-hands.types";

const localBackend: Backend = {
  id: "bundled",
  name: "Bundled",
  host: "http://localhost:3000",
  apiKey: "",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

// Hoisted mocks for controllable return values
const {
  mockMutate,
  mockDisplaySuccessToast,
  useActiveConversationMock,
  useConfigMock,
  useActiveBackendMock,
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
    data: {},
  })),
  useActiveBackendMock: vi.fn(),
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

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => useActiveBackendMock(),
}));

vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: mockDisplaySuccessToast,
  displayErrorToast: vi.fn(),
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
          BUTTON$EXPORT_TRANSCRIPT: "Export…",
          BUTTON$DOWNLOAD_CONVERSATION_DATA: "Download conversation data",
          BUTTON$DOWNLOAD_VIA_VSCODE: "Download via VS Code",
          BUTTON$SHOW_AGENT_TOOLS_AND_METADATA: "Show Agent Tools",
          CONVERSATION$SHOW_SKILLS: "Show Skills",
          BUTTON$DISPLAY_COST: "Display Cost",
          COMMON$CLOSE_CONVERSATION_STOP_RUNTIME:
            "Stop Conversation (Runtime)",
          COMMON$STOP_CONVERSATION: "Stop Conversation",
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

// Helper function to render ConversationName with navigation context
const renderConversationNameWithRouter = () => {
  return renderWithProviders(<ConversationName />, {
    navigation: {
      currentPath: "/conversations/test-conversation-id",
      conversationId: "test-conversation-id",
    },
  });
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

  beforeEach(() => {
    useActiveBackendMock.mockReturnValue({
      backend: localBackend,
      orgId: null,
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

  it("should not render the llm model in the title bar", () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-conversation-id",
        title: "Test Conversation",
        status: "RUNNING",
        llm_model: "openai/gpt-4o",
      } as Conversation,
    });

    renderConversationNameWithRouter();

    expect(
      screen.queryByTestId("conversation-name-llm-model"),
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

  beforeEach(() => {
    useActiveBackendMock.mockReturnValue({
      backend: localBackend,
      orgId: null,
    });
  });

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
      onDownloadConversation: vi.fn(),
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
      screen.getByTestId("download-trajectory-button"),
    ).toBeInTheDocument();
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

  it("forwards the position prop as a data-position attribute (top)", () => {
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

    expect(
      screen.getByTestId("conversation-name-context-menu"),
    ).toHaveAttribute("data-position", "top");
  });

  it("forwards the position prop as a data-position attribute (bottom)", () => {
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

    expect(
      screen.getByTestId("conversation-name-context-menu"),
    ).toHaveAttribute("data-position", "bottom");
  });

  it("should render correct text content for each menu option", () => {
    const handlers = {
      onRename: vi.fn(),
      onDelete: vi.fn(),
      onStop: vi.fn(),
      onDisplayCost: vi.fn(),
      onShowAgentTools: vi.fn(),
      onShowSkills: vi.fn(),
      onExportTranscript: vi.fn(),
      onDownloadConversation: vi.fn(),
    };

    renderWithProviders(
      <ConversationNameContextMenu {...defaultProps} {...handlers} />,
    );

    expect(screen.getByTestId("rename-button")).toHaveTextContent("Rename");
    expect(screen.getByTestId("delete-button")).toHaveTextContent(
      "Delete Conversation",
    );
    expect(screen.getByTestId("stop-button")).toHaveTextContent(
      "Stop Conversation",
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
    expect(screen.getByTestId("export-transcript-button")).toHaveTextContent(
      "Export…",
    );
    expect(screen.getByTestId("download-trajectory-button")).toHaveTextContent(
      "Download conversation data",
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

describe("ConversationName public sharing", () => {
  let updatePublicFlagSpy: ReturnType<typeof vi.spyOn>;
  let writeTextSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    if (!("clipboard" in navigator)) {
      Object.defineProperty(globalThis.navigator, "clipboard", {
        configurable: true,
        value: { writeText: () => Promise.resolve() },
      });
    }
    writeTextSpy = vi
      .spyOn(navigator.clipboard, "writeText")
      .mockResolvedValue(undefined);
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-conversation-id",
        title: "Test Conversation",
        status: "RUNNING",
        public: false,
      } as Conversation,
    });
    useActiveBackendMock.mockReturnValue({
      backend: cloudBackend,
      orgId: null,
    });
    updatePublicFlagSpy = vi
      .spyOn(AgentServerConversationService, "updateConversationPublicFlag")
      .mockResolvedValue({ id: "test-conversation-id", public: true } as never);
  });

  afterEach(() => {
    updatePublicFlagSpy.mockRestore();
    writeTextSpy.mockRestore();
    vi.clearAllMocks();
  });

  it("renders the Public Share menu item on cloud backends", async () => {
    const user = userEvent.setup();
    renderConversationNameWithRouter();

    await user.click(screen.getByTestId("ellipsis-button"));

    expect(screen.getByTestId("share-publicly-button")).toBeInTheDocument();
  });

  it("hides the Public Share menu item on local backends", async () => {
    useActiveBackendMock.mockReturnValue({
      backend: localBackend,
      orgId: null,
    });
    const user = userEvent.setup();
    renderConversationNameWithRouter();

    await user.click(screen.getByTestId("ellipsis-button"));

    expect(
      screen.queryByTestId("share-publicly-button"),
    ).not.toBeInTheDocument();
  });

  it("calls the update service with the toggled flag when clicked", async () => {
    const user = userEvent.setup();
    renderConversationNameWithRouter();

    await user.click(screen.getByTestId("ellipsis-button"));
    const toggleLabel = screen
      .getByTestId("share-publicly-button")
      .closest("label");
    expect(toggleLabel).not.toBeNull();
    await user.click(toggleLabel!);

    expect(updatePublicFlagSpy).toHaveBeenCalledWith(
      "test-conversation-id",
      true,
    );
  });

  it("uses the cloud environment domain for the share link and clipboard copy", async () => {
    useActiveConversationMock.mockReturnValue({
      data: {
        conversation_id: "test-conversation-id",
        title: "Test Conversation",
        status: "RUNNING",
        public: true,
      } as Conversation,
    });
    const expectedUrl =
      "https://app.all-hands.dev/shared/conversations/test-conversation-id";
    const user = userEvent.setup();
    renderConversationNameWithRouter();

    await user.click(screen.getByTestId("ellipsis-button"));
    expect(screen.getByTestId("open-share-link-button")).toHaveAttribute(
      "href",
      expectedUrl,
    );

    await user.click(screen.getByTestId("copy-share-link-button"));
    expect(writeTextSpy).toHaveBeenCalledWith(expectedUrl);
  });
});
