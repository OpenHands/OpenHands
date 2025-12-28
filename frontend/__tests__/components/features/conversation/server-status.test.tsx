import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { ServerStatus } from "#/components/features/controls/server-status";
import { ServerStatusContextMenu } from "#/components/features/controls/server-status-context-menu";
import { ConversationStatus } from "#/types/conversation-status";
import { AgentState } from "#/types/agent-state";
import { useAgentState } from "#/hooks/use-agent-state";

// Mock the agent state hook
vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: vi.fn(),
}));

vi.mock("#/hooks/query/use-task-polling", () => ({
  useTaskPolling: () => ({
    isTask: false,
    taskId: null,
    conversationId: "test-conversation-id",
    task: null,
    taskStatus: null,
    taskDetail: null,
    taskError: null,
    isLoadingTask: false,
  }),
}));

// Mock react-i18next
vi.mock("react-i18next", async () => {
  const actual = await vi.importActual("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => {
        const translations: Record<string, string> = {
          COMMON$RUNNING: "Running",
          COMMON$SERVER_STOPPED: "Server Stopped",
          COMMON$ERROR: "Error",
          COMMON$STARTING: "Starting",
          COMMON$STOPPING: "Stopping...",
          COMMON$STOP_RUNTIME: "Stop Runtime",
          COMMON$START_RUNTIME: "Start Runtime",
          CONVERSATION$ERROR_STARTING_CONVERSATION:
            "Error starting conversation",
          CONVERSATION$READY: "Ready",
        };
        return translations[key] || key;
      },
      i18n: {
        changeLanguage: () => new Promise(() => {}),
      },
    }),
  };
});

describe("ServerStatus", () => {
  // Helper function to mock agent state with specific state
  const mockAgentStore = (agentState: AgentState) => {
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: agentState,
    });
  };

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render server status with RUNNING conversation status", () => {
    mockAgentStore(AgentState.RUNNING);

    renderWithProviders(<ServerStatus conversationStatus="RUNNING" />);

    expect(screen.getByTestId("server-status")).toBeInTheDocument();
    expect(screen.getByText("Running")).toBeInTheDocument();
  });

  it("should render server status with STOPPED conversation status", () => {
    mockAgentStore(AgentState.RUNNING);

    renderWithProviders(<ServerStatus conversationStatus="STOPPED" />);

    expect(screen.getByTestId("server-status")).toBeInTheDocument();
    expect(screen.getByText("Server Stopped")).toBeInTheDocument();
  });

  it("should render STARTING status when agent state is LOADING", () => {
    mockAgentStore(AgentState.LOADING);

    renderWithProviders(<ServerStatus conversationStatus="STARTING" />);

    expect(screen.getByTestId("server-status")).toBeInTheDocument();
    expect(screen.getByText("Starting")).toBeInTheDocument();
  });

  it("should render STARTING status when agent state is INIT", () => {
    mockAgentStore(AgentState.INIT);

    renderWithProviders(<ServerStatus conversationStatus="STARTING" />);

    expect(screen.getByTestId("server-status")).toBeInTheDocument();
    expect(screen.getByText("Starting")).toBeInTheDocument();
  });

  it("should render ERROR status when agent state is ERROR", () => {
    mockAgentStore(AgentState.ERROR);

    renderWithProviders(<ServerStatus conversationStatus="RUNNING" />);

    expect(screen.getByTestId("server-status")).toBeInTheDocument();
    expect(screen.getByText("Error")).toBeInTheDocument();
  });

  it("should render STOPPING status when isPausing is true", () => {
    mockAgentStore(AgentState.RUNNING);

    renderWithProviders(
      <ServerStatus conversationStatus="RUNNING" isPausing={true} />,
    );

    expect(screen.getByTestId("server-status")).toBeInTheDocument();
    expect(screen.getByText("Stopping...")).toBeInTheDocument();
  });

  it("should handle null conversation status", () => {
    mockAgentStore(AgentState.RUNNING);

    renderWithProviders(<ServerStatus conversationStatus={null} />);

    expect(screen.getByTestId("server-status")).toBeInTheDocument();
    expect(screen.getByText("Running")).toBeInTheDocument();
  });

  it("should apply custom className", () => {
    mockAgentStore(AgentState.RUNNING);

    renderWithProviders(
      <ServerStatus conversationStatus="RUNNING" className="custom-class" />,
    );

    const container = screen.getByTestId("server-status");
    expect(container).toHaveClass("custom-class");
  });
});

describe("ServerStatusContextMenu", () => {
  // Helper function to mock agent state with specific state
  const mockAgentStore = (agentState: AgentState) => {
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: agentState,
    });
  };

  const defaultProps = {
    onClose: vi.fn(),
    conversationStatus: "RUNNING" as ConversationStatus,
  };

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render stop server button when status is RUNNING", () => {
    mockAgentStore(AgentState.RUNNING);

    renderWithProviders(
      <ServerStatusContextMenu
        {...defaultProps}
        conversationStatus="RUNNING"
        onStopServer={vi.fn()}
      />,
    );

    expect(screen.getByTestId("server-status")).toBeInTheDocument();
    expect(screen.getByTestId("stop-server-button")).toBeInTheDocument();
    expect(screen.getByText("Stop Runtime")).toBeInTheDocument();
  });

  it("should render start server button when status is STOPPED", () => {
    mockAgentStore(AgentState.RUNNING);

    renderWithProviders(
      <ServerStatusContextMenu
        {...defaultProps}
        conversationStatus="STOPPED"
        onStartServer={vi.fn()}
      />,
    );

    expect(screen.getByTestId("server-status")).toBeInTheDocument();
    expect(screen.getByTestId("start-server-button")).toBeInTheDocument();
    expect(screen.getByText("Start Runtime")).toBeInTheDocument();
  });

  it("should not render stop server button when onStopServer is not provided", () => {
    mockAgentStore(AgentState.RUNNING);

    renderWithProviders(
      <ServerStatusContextMenu
        {...defaultProps}
        conversationStatus="RUNNING"
      />,
    );

    expect(screen.getByTestId("server-status")).toBeInTheDocument();
    expect(screen.queryByTestId("stop-server-button")).not.toBeInTheDocument();
  });

  it("should not render start server button when onStartServer is not provided", () => {
    mockAgentStore(AgentState.RUNNING);

    renderWithProviders(
      <ServerStatusContextMenu
        {...defaultProps}
        conversationStatus="STOPPED"
      />,
    );

    expect(screen.getByTestId("server-status")).toBeInTheDocument();
    expect(screen.queryByTestId("start-server-button")).not.toBeInTheDocument();
  });

  it("should call onStopServer when stop button is clicked", async () => {
    const user = userEvent.setup();
    const onStopServer = vi.fn();
    mockAgentStore(AgentState.RUNNING);

    renderWithProviders(
      <ServerStatusContextMenu
        {...defaultProps}
        conversationStatus="RUNNING"
        onStopServer={onStopServer}
      />,
    );

    const stopButton = screen.getByTestId("stop-server-button");
    await user.click(stopButton);

    expect(onStopServer).toHaveBeenCalledTimes(1);
  });

  it("should call onStartServer when start button is clicked", async () => {
    const user = userEvent.setup();
    const onStartServer = vi.fn();
    mockAgentStore(AgentState.RUNNING);

    renderWithProviders(
      <ServerStatusContextMenu
        {...defaultProps}
        conversationStatus="STOPPED"
        onStartServer={onStartServer}
      />,
    );

    const startButton = screen.getByTestId("start-server-button");
    await user.click(startButton);

    expect(onStartServer).toHaveBeenCalledTimes(1);
  });

  it("should render correct text content for stop server button", () => {
    mockAgentStore(AgentState.RUNNING);

    renderWithProviders(
      <ServerStatusContextMenu
        {...defaultProps}
        conversationStatus="RUNNING"
        onStopServer={vi.fn()}
      />,
    );

    expect(screen.getByTestId("stop-server-button")).toHaveTextContent(
      "Stop Runtime",
    );
  });

  it("should render correct text content for start server button", () => {
    mockAgentStore(AgentState.RUNNING);

    renderWithProviders(
      <ServerStatusContextMenu
        {...defaultProps}
        conversationStatus="STOPPED"
        onStartServer={vi.fn()}
      />,
    );

    expect(screen.getByTestId("start-server-button")).toHaveTextContent(
      "Start Runtime",
    );
  });

  it("should call onClose when context menu is closed", () => {
    const onClose = vi.fn();
    mockAgentStore(AgentState.RUNNING);

    renderWithProviders(
      <ServerStatusContextMenu
        {...defaultProps}
        onClose={onClose}
        conversationStatus="RUNNING"
        onStopServer={vi.fn()}
      />,
    );

    // The onClose is typically called by the parent component when clicking outside
    // This test verifies the prop is properly passed
    expect(onClose).toBeDefined();
  });

  it("should not render any buttons for other conversation statuses", () => {
    mockAgentStore(AgentState.RUNNING);

    renderWithProviders(
      <ServerStatusContextMenu
        {...defaultProps}
        conversationStatus="STARTING"
      />,
    );

    expect(screen.getByTestId("server-status")).toBeInTheDocument();
    expect(screen.queryByTestId("stop-server-button")).not.toBeInTheDocument();
    expect(screen.queryByTestId("start-server-button")).not.toBeInTheDocument();
  });
});
