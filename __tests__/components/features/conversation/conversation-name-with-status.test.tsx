import { fireEvent, render, screen } from "@testing-library/react";
import type { MouseEventHandler } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ConversationNameWithStatus } from "#/components/features/conversation/conversation-name-with-status";
import {
  OH_STATUS_ERROR_COLOR,
  OH_STATUS_SUCCESS_COLOR,
} from "#/constants/status-colors";
import { AgentState } from "#/types/agent-state";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

const mocks = vi.hoisted(() => ({
  conversationId: "conversation-1" as string | undefined,
  conversation: {
    execution_status: "running",
  } as { execution_status?: ExecutionStatus | null } | undefined,
  curAgentState: "running" as AgentState,
  isTask: false,
  taskStatus: null as string | null,
  pauseConversation: vi.fn(),
  resumeConversation: vi.fn(),
  providers: [{ name: "provider-1" }],
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
  }),
}));

vi.mock("#/hooks/mutation/use-unified-stop-conversation", () => ({
  useUnifiedPauseConversation: () => ({
    mutate: mocks.pauseConversation,
  }),
}));

vi.mock("#/hooks/mutation/use-unified-start-conversation", () => ({
  useUnifiedResumeConversation: () => ({
    mutate: mocks.resumeConversation,
  }),
}));

vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => ({ providers: mocks.providers }),
}));

vi.mock("#/icons/debug-stackframe-dot.svg?react", () => ({
  default: ({ color }: { color: string }) => (
    <span data-testid="conversation-status-dot" data-color={color} />
  ),
}));

vi.mock("#/components/features/controls/server-status-context-menu", () => ({
  ServerStatusContextMenu: ({
    executionStatus,
    isPausing,
    onClose,
    onStartServer,
    onStopServer,
    position,
  }: {
    executionStatus: ExecutionStatus | null;
    isPausing: boolean;
    onClose: () => void;
    onStartServer?: MouseEventHandler<HTMLButtonElement>;
    onStopServer?: MouseEventHandler<HTMLButtonElement>;
    position: "top" | "bottom";
  }) => (
    <div
      data-testid="server-status-context-menu"
      data-execution-status={executionStatus ?? "null"}
      data-is-pausing={String(isPausing)}
      data-position={position}
    >
      <button type="button" data-testid="close-status-menu" onClick={onClose}>
        close
      </button>
      {onStopServer && (
        <button
          type="button"
          data-testid="stop-server-button"
          onClick={onStopServer}
        >
          stop
        </button>
      )}
      {onStartServer && (
        <button
          type="button"
          data-testid="start-server-button"
          onClick={onStartServer}
        >
          start
        </button>
      )}
    </div>
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

/**
 * The status controls live behind a trigger button; the menu (and its
 * start/stop actions) only mount once it is opened.
 */
function openStatusMenu() {
  fireEvent.click(screen.getByTestId("server-status-menu-trigger"));
}

describe("conversation name status controls", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.conversationId = "conversation-1";
    mocks.conversation = { execution_status: ExecutionStatus.RUNNING };
    mocks.curAgentState = AgentState.RUNNING;
    mocks.isTask = false;
    mocks.taskStatus = null;
    mocks.providers = [{ name: "provider-1" }];
  });

  it("pauses an active conversation without triggering its parent", () => {
    const { onParentClick } = renderSubject();

    expect(screen.getByTestId("conversation-name")).toBeInTheDocument();
    // The toolbar wrapper carries the trailing margin around the toggles.
    expect(screen.getByTestId("right-panel-toggle").parentElement).toHaveClass(
      "mr-2",
    );
    expect(screen.getByTestId("conversation-status-dot")).toHaveAttribute(
      "data-color",
      OH_STATUS_SUCCESS_COLOR,
    );

    openStatusMenu();

    expect(screen.getByTestId("server-status-context-menu")).toMatchObject({
      dataset: {
        executionStatus: ExecutionStatus.RUNNING,
        isPausing: "false",
        position: "bottom",
      },
    });
    expect(screen.queryByTestId("start-server-button")).not.toBeInTheDocument();

    expect(fireEvent.click(screen.getByTestId("stop-server-button"))).toBe(
      false,
    );

    expect(mocks.pauseConversation).toHaveBeenCalledWith({
      conversationId: "conversation-1",
    });
    expect(mocks.resumeConversation).not.toHaveBeenCalled();
    expect(onParentClick).not.toHaveBeenCalled();
  });

  it("resumes a paused conversation with the selected providers", () => {
    mocks.conversation = { execution_status: ExecutionStatus.PAUSED };
    mocks.providers = [{ name: "provider-a" }, { name: "provider-b" }];
    const { onParentClick } = renderSubject();

    openStatusMenu();

    expect(screen.queryByTestId("stop-server-button")).not.toBeInTheDocument();
    expect(fireEvent.click(screen.getByTestId("start-server-button"))).toBe(
      false,
    );

    expect(mocks.resumeConversation).toHaveBeenCalledWith({
      conversationId: "conversation-1",
      providers: mocks.providers,
    });
    expect(mocks.pauseConversation).not.toHaveBeenCalled();
    expect(onParentClick).not.toHaveBeenCalled();
  });

  it("dismisses the status menu when the child menu requests close", () => {
    renderSubject();

    openStatusMenu();
    expect(
      screen.getByTestId("server-status-context-menu"),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByTestId("close-status-menu"));

    expect(
      screen.queryByTestId("server-status-context-menu"),
    ).not.toBeInTheDocument();
    expect(mocks.pauseConversation).not.toHaveBeenCalled();
    expect(mocks.resumeConversation).not.toHaveBeenCalled();
  });

  it("opens on mouse hover and closes when the pointer leaves", () => {
    renderSubject();
    const container = screen.getByTestId(
      "server-status-menu-trigger",
    ).parentElement as HTMLElement;

    expect(
      screen.queryByTestId("server-status-context-menu"),
    ).not.toBeInTheDocument();

    fireEvent.pointerEnter(container, { pointerType: "mouse" });
    expect(
      screen.getByTestId("server-status-context-menu"),
    ).toBeInTheDocument();

    // A non-mouse pointer leaving must not dismiss a hover-opened menu.
    fireEvent.pointerLeave(container, { pointerType: "touch" });
    expect(
      screen.getByTestId("server-status-context-menu"),
    ).toBeInTheDocument();

    fireEvent.pointerLeave(container, { pointerType: "mouse" });
    expect(
      screen.queryByTestId("server-status-context-menu"),
    ).not.toBeInTheDocument();
  });

  it("ignores hover from non-mouse pointers", () => {
    renderSubject();
    const container = screen.getByTestId(
      "server-status-menu-trigger",
    ).parentElement as HTMLElement;

    fireEvent.pointerEnter(container, { pointerType: "touch" });

    expect(
      screen.queryByTestId("server-status-context-menu"),
    ).not.toBeInTheDocument();
  });

  it("toggles the menu closed and reflects aria-expanded when clicked twice", () => {
    renderSubject();
    const trigger = screen.getByTestId("server-status-menu-trigger");
    expect(trigger).toHaveClass("rounded-md", "hover:bg-white/10");
    expect(trigger).toHaveAttribute("aria-expanded", "false");

    fireEvent.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(
      screen.getByTestId("server-status-context-menu"),
    ).toBeInTheDocument();

    fireEvent.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(
      screen.queryByTestId("server-status-context-menu"),
    ).not.toBeInTheDocument();
  });

  it.each([
    [ExecutionStatus.RUNNING, "stop-server-button"],
    [ExecutionStatus.PAUSED, "start-server-button"],
  ] as const)(
    "does not invoke a runtime action for %s without a conversation id",
    (executionStatus, actionTestId) => {
      mocks.conversationId = undefined;
      mocks.conversation = { execution_status: executionStatus };
      const { onParentClick } = renderSubject();

      openStatusMenu();

      expect(fireEvent.click(screen.getByTestId(actionTestId))).toBe(false);
      expect(mocks.pauseConversation).not.toHaveBeenCalled();
      expect(mocks.resumeConversation).not.toHaveBeenCalled();
      expect(onParentClick).not.toHaveBeenCalled();
    },
  );

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

      openStatusMenu();

      expect(screen.getByTestId("server-status-context-menu")).toHaveAttribute(
        "data-execution-status",
        "null",
      );
      expect(
        screen.queryByTestId("stop-server-button"),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByTestId("start-server-button"),
      ).not.toBeInTheDocument();
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
