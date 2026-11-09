import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ConversationNameWithStatus } from "#/components/features/conversation/conversation-name-with-status";
import {
  OH_STATUS_ERROR_COLOR,
  OH_STATUS_SUCCESS_COLOR,
} from "#/constants/status-colors";
import { AgentState } from "#/types/agent-state";
import { useConversationLaunchStore } from "#/stores/conversation-launch-store";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

const mocks = vi.hoisted(() => ({
  conversationId: "conversation-1" as string | undefined,
  conversation: {
    execution_status: "running",
  } as
    | {
        execution_status?: ExecutionStatus | null;
        sandbox_status?: string | null;
      }
    | undefined,
  curAgentState: "running" as AgentState,
  isTask: false,
  taskStatus: null as string | null,
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: mocks.conversationId }),
  useOptionalConversationId: () => ({ conversationId: mocks.conversationId }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: mocks.conversation }),
}));

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({ curAgentState: mocks.curAgentState }),
}));

vi.mock("#/hooks/query/use-task-polling", () => ({
  useTaskPolling: () => ({
    isTask: mocks.isTask,
    taskStatus: mocks.taskStatus,
    taskDetail: null,
  }),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

vi.mock("#/icons/debug-stackframe-dot.svg?react", () => ({
  default: ({ color }: { color: string }) => (
    <span data-testid="conversation-status-dot" data-color={color} />
  ),
}));

vi.mock("#/components/features/conversation/conversation-name", () => ({
  ConversationName: () => <div data-testid="conversation-name" />,
}));

vi.mock("#/components/features/conversation/right-panel-toggle", () => ({
  RightPanelToggle: ({ className }: { className: string }) => (
    <div data-testid="right-panel-toggle" data-class-name={className} />
  ),
}));

function renderSubject(onParentClick = vi.fn()) {
  const view = render(
    <div onClick={onParentClick}>
      <ConversationNameWithStatus />
    </div>,
  );
  return { ...view, onParentClick };
}

describe("conversation status dot", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useConversationLaunchStore.setState({ launches: {} });
    mocks.conversationId = "conversation-1";
    mocks.conversation = { execution_status: ExecutionStatus.RUNNING };
    mocks.curAgentState = AgentState.RUNNING;
    mocks.isTask = false;
    mocks.taskStatus = null;
  });

  it("shows status only: clicking it opens no menu", () => {
    renderSubject();

    const status = screen.getByTestId("conversation-status");
    fireEvent.click(status);

    expect(status).toHaveAttribute("aria-label", "COMMON$RUNNING");
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    expect(screen.queryByText("COMMON$STOP_SANDBOX")).not.toBeInTheDocument();
    expect(screen.getByTestId("conversation-name")).toBeInTheDocument();
  });

  it("calls a paused agent stopped, even while its sandbox keeps running", () => {
    mocks.conversation = {
      execution_status: ExecutionStatus.PAUSED,
      sandbox_status: "RUNNING",
    };
    renderSubject();

    expect(screen.getByTestId("conversation-status")).toHaveAttribute(
      "aria-label",
      "AGENT_STATUS$AGENT_STOPPED",
    );
  });

  it("shows a failed launch instead of the previous conversation status", () => {
    const launchId = useConversationLaunchStore.getState().begin({
      backendId: "local",
      orgId: null,
      message: "hello",
      files: [],
    });
    useConversationLaunchStore
      .getState()
      .update(launchId, { error: "Create failed" });
    mocks.conversationId = launchId;
    mocks.conversation = undefined;
    renderSubject();

    expect(screen.getByTestId("conversation-status-dot")).toHaveAttribute(
      "data-color",
      OH_STATUS_ERROR_COLOR,
    );
    expect(screen.getByTestId("conversation-status")).toHaveAttribute(
      "aria-label",
      "COMMON$ERROR",
    );
  });

  it("shows the running color for an active conversation", () => {
    renderSubject();

    expect(screen.getByTestId("conversation-status-dot")).toHaveAttribute(
      "data-color",
      OH_STATUS_SUCCESS_COLOR,
    );
  });

  it.each([AgentState.LOADING, AgentState.INIT])(
    "shows a starting color for the %s agent state without a conversation",
    (curAgentState) => {
      mocks.conversation = undefined;
      mocks.curAgentState = curAgentState;
      renderSubject();

      expect(screen.getByTestId("conversation-status-dot")).toHaveAttribute(
        "data-color",
        "#FFD600",
      );
    },
  );

  it.each([
    {
      name: "task error",
      executionStatus: ExecutionStatus.RUNNING,
      agentState: AgentState.RUNNING,
      isTask: true,
      taskStatus: "ERROR",
      expectedColor: OH_STATUS_ERROR_COLOR,
    },
    {
      name: "execution error",
      executionStatus: ExecutionStatus.ERROR,
      agentState: AgentState.RUNNING,
      isTask: false,
      taskStatus: null,
      expectedColor: "#ffffff",
    },
    {
      name: "agent error",
      executionStatus: null,
      agentState: AgentState.ERROR,
      isTask: false,
      taskStatus: null,
      expectedColor: OH_STATUS_ERROR_COLOR,
    },
  ])(
    "shows the expected color for $name",
    ({ agentState, executionStatus, expectedColor, isTask, taskStatus }) => {
      mocks.conversation = { execution_status: executionStatus };
      mocks.curAgentState = agentState;
      mocks.isTask = isTask;
      mocks.taskStatus = taskStatus;
      renderSubject();

      expect(screen.getByTestId("conversation-status-dot")).toHaveAttribute(
        "data-color",
        expectedColor,
      );
    },
  );
});
