import type { ReactNode, SVGProps } from "react";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ChangeAgentButton } from "#/components/features/chat/change-agent-button";
import { AgentState } from "#/types/agent-state";
import { useConversationStore } from "#/stores/conversation-store";
import { renderWithProviders } from "../../../../test-utils";

const mocks = vi.hoisted(() => ({
  webSocketStatus: "OPEN",
  agentState: "awaiting_user_input",
  conversation: {
    id: "parent-conversation-123",
    sub_conversation_ids: [] as string[],
  } as
    | {
        id: string;
        sub_conversation_ids: string[];
      }
    | undefined,
  taskStatus: undefined as string | undefined,
  subConversationId: undefined as string | undefined,
  isCreatingConversation: false,
  handlePlanClick: vi.fn(
    (event?: { preventDefault?: () => void; stopPropagation?: () => void }) => {
      event?.preventDefault?.();
      event?.stopPropagation?.();
    },
  ),
  invalidateQueries: vi.fn(),
}));

vi.mock("#/components/shared/buttons/styled-tooltip", () => ({
  StyledTooltip: ({
    content,
    children,
  }: {
    content: ReactNode;
    children: ReactNode;
  }) => (
    <>
      {children}
      <span data-testid="styled-tooltip-content">{content}</span>
    </>
  ),
}));

vi.mock("#/icons/lesson-plan.svg?react", () => ({
  default: (props: SVGProps<SVGSVGElement>) => (
    <svg data-testid="plan-agent-icon" {...props} />
  ),
}));

vi.mock("#/icons/code-pill", () => ({
  CodePillIcon: (props: SVGProps<SVGSVGElement>) => (
    <svg data-testid="code-agent-icon" {...props} />
  ),
}));

vi.mock("#/hooks/use-unified-websocket-status", () => ({
  useUnifiedWebSocketStatus: () => mocks.webSocketStatus,
}));

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({ curAgentState: mocks.agentState }),
}));

vi.mock("@tanstack/react-query", async () => {
  const actual = await vi.importActual("@tanstack/react-query");
  return {
    ...actual,
    useQueryClient: () => ({
      invalidateQueries: mocks.invalidateQueries,
    }),
  };
});

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: mocks.conversation,
    isFetched: true,
    refetch: vi.fn(),
  }),
}));

vi.mock("#/hooks/query/use-sub-conversation-task-polling", () => ({
  useSubConversationTaskPolling: (
    _taskId: string | null,
    parentConversationId: string | null,
  ) => {
    const hasExpectedParent =
      parentConversationId === (mocks.conversation?.id ?? null);

    return {
      taskStatus: hasExpectedParent ? mocks.taskStatus : undefined,
      subConversationId: hasExpectedParent
        ? mocks.subConversationId
        : undefined,
    };
  },
}));

vi.mock("#/hooks/use-handle-plan-click", () => ({
  useHandlePlanClick: () => ({
    handlePlanClick: mocks.handlePlanClick,
    isCreatingConversation: mocks.isCreatingConversation,
  }),
}));

type RenderOverrides = {
  conversationMode?: "code" | "plan";
  subConversationTaskId?: string | null;
  conversationId?: string | null;
  activeConversationId?: string | null;
  webSocketStatus?: "CONNECTING" | "OPEN" | "CLOSED" | "CLOSING";
  agentState?: AgentState;
  isCreatingConversation?: boolean;
  taskStatus?: string;
  subConversationId?: string;
};

function renderButton(overrides: RenderOverrides = {}) {
  mocks.invalidateQueries.mockReset();
  mocks.handlePlanClick.mockReset();
  mocks.webSocketStatus = overrides.webSocketStatus ?? "OPEN";
  mocks.agentState = overrides.agentState ?? AgentState.AWAITING_USER_INPUT;
  mocks.conversation =
    overrides.activeConversationId === null
      ? undefined
      : {
          id: overrides.activeConversationId ?? "parent-conversation-123",
          sub_conversation_ids: [],
        };
  mocks.taskStatus = overrides.taskStatus;
  mocks.subConversationId = overrides.subConversationId;
  mocks.isCreatingConversation = overrides.isCreatingConversation ?? false;

  useConversationStore.setState({
    conversationMode: overrides.conversationMode ?? "code",
    subConversationTaskId: overrides.subConversationTaskId ?? null,
  });

  mocks.handlePlanClick.mockImplementation((event) => {
    event?.preventDefault?.();
    event?.stopPropagation?.();
    useConversationStore.getState().setConversationMode("plan");
  });

  return renderWithProviders(<ChangeAgentButton />, {
    navigation: {
      conversationId:
        overrides.conversationId === undefined
          ? "route-conversation-123"
          : overrides.conversationId,
    },
  });
}

function getModeButton() {
  return screen.getByRole("button", {
    name: /COMMON\$(CODE|PLAN)/,
  });
}

describe("ChangeAgentButton cache refresh", () => {
  it("refreshes the parent once per ready task", async () => {
    const { rerender } = renderButton({
      subConversationTaskId: "task-456",
      taskStatus: "READY",
      subConversationId: "sub-conversation-789",
    });

    await waitFor(() => {
      expect(mocks.invalidateQueries).toHaveBeenCalledTimes(1);
    });
    expect(mocks.invalidateQueries).toHaveBeenLastCalledWith({
      queryKey: ["user", "conversation", "parent-conversation-123"],
    });

    mocks.subConversationId = "sub-conversation-790";
    rerender(<ChangeAgentButton />);
    expect(mocks.invalidateQueries).toHaveBeenCalledTimes(1);

    act(() => {
      useConversationStore.setState({ subConversationTaskId: "task-457" });
    });

    await waitFor(() => {
      expect(mocks.invalidateQueries).toHaveBeenCalledTimes(2);
    });
  });

  it.each([
    {
      condition: "the task is still working",
      overrides: {
        subConversationTaskId: "task-456",
        taskStatus: "WORKING",
        subConversationId: "sub-conversation-789",
      },
    },
    {
      condition: "the task has no sub-conversation yet",
      overrides: {
        subConversationTaskId: "task-456",
        taskStatus: "READY",
      },
    },
    {
      condition: "the parent conversation is unavailable",
      overrides: {
        subConversationTaskId: "task-456",
        taskStatus: "READY",
        subConversationId: "sub-conversation-789",
        activeConversationId: null,
      },
    },
    {
      condition: "there is no task to refresh",
      overrides: {
        taskStatus: "READY",
        subConversationId: "sub-conversation-789",
      },
    },
  ])("does not refresh when $condition", ({ overrides }) => {
    renderButton(overrides);

    expect(mocks.invalidateQueries).not.toHaveBeenCalled();
  });
});

describe("ChangeAgentButton mode selection", () => {
  it("shows the current mode and toggles its menu", async () => {
    const user = userEvent.setup();
    renderButton();

    const button = getModeButton();
    expect(button).toBeEnabled();
    expect(button).toHaveTextContent("COMMON$CODE");
    expect(button).toHaveClass(
      "flex",
      "items-center",
      "rounded-[100px]",
      "border",
      "border-transparent",
      "text-[var(--oh-muted)]",
      "cursor-pointer",
      "hover:text-white",
      "hover:bg-white/10",
    );
    expect(screen.getByTestId("code-agent-icon")).toBeInTheDocument();
    expect(screen.queryByTestId("plan-agent-icon")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("styled-tooltip-content"),
    ).not.toBeInTheDocument();

    await user.click(button);
    expect(screen.getByTestId("change-agent-context-menu")).toBeInTheDocument();

    await user.click(button);
    expect(
      screen.queryByTestId("change-agent-context-menu"),
    ).not.toBeInTheDocument();
  });

  it("selects plan and code modes from the menu", async () => {
    const user = userEvent.setup();
    renderButton();

    await user.click(getModeButton());
    await user.click(screen.getByTestId("plan-option"));

    expect(getModeButton()).toHaveTextContent("COMMON$PLAN");
    expect(getModeButton()).toHaveClass(
      "border",
      "border-[#597FF4]",
      "bg-[#4A67BD]",
      "cursor-pointer",
      "text-white",
      "hover:bg-[#597FF4]",
    );
    expect(getModeButton()).not.toHaveClass("border-transparent");
    expect(screen.getByTestId("plan-agent-icon")).toBeInTheDocument();
    expect(screen.queryByTestId("code-agent-icon")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("change-agent-context-menu"),
    ).not.toBeInTheDocument();

    await user.click(getModeButton());
    await user.click(screen.getByTestId("code-option"));

    expect(getModeButton()).toHaveTextContent("COMMON$CODE");
    expect(screen.getByTestId("code-agent-icon")).toBeInTheDocument();
    expect(
      screen.queryByTestId("change-agent-context-menu"),
    ).not.toBeInTheDocument();
  });

  it("cycles modes with Shift+Tab without leaking the shortcut", async () => {
    renderButton();

    fireEvent.keyDown(document, { key: "Tab", shiftKey: false });
    fireEvent.keyDown(document, { key: "x", shiftKey: true });
    expect(getModeButton()).toHaveTextContent("COMMON$CODE");
    expect(mocks.handlePlanClick).not.toHaveBeenCalled();

    const propagated = vi.fn();
    window.addEventListener("keydown", propagated);
    const planEvent = new KeyboardEvent("keydown", {
      key: "Tab",
      shiftKey: true,
      bubbles: true,
      cancelable: true,
    });
    let dispatched = true;
    act(() => {
      dispatched = document.dispatchEvent(planEvent);
    });
    window.removeEventListener("keydown", propagated);

    expect(dispatched).toBe(false);
    expect(planEvent.defaultPrevented).toBe(true);
    expect(propagated).not.toHaveBeenCalled();
    expect(mocks.handlePlanClick).toHaveBeenCalledWith(planEvent);
    await waitFor(() => {
      expect(getModeButton()).toHaveTextContent("COMMON$PLAN");
    });

    const codeEvent = new KeyboardEvent("keydown", {
      key: "Tab",
      shiftKey: true,
      bubbles: true,
      cancelable: true,
    });
    act(() => {
      document.dispatchEvent(codeEvent);
    });

    expect(codeEvent.defaultPrevented).toBe(true);
    expect(getModeButton()).toHaveTextContent("COMMON$CODE");
    expect(mocks.handlePlanClick).toHaveBeenCalledTimes(1);
  });
});

describe("ChangeAgentButton availability", () => {
  it.each([
    {
      condition: "no conversation has started",
      overrides: { conversationId: null },
    },
    {
      condition: "the agent is running",
      overrides: { agentState: AgentState.RUNNING },
    },
    {
      condition: "a plan conversation is being created",
      overrides: {
        conversationMode: "plan" as const,
        isCreatingConversation: true,
      },
    },
    {
      condition: "the websocket is disconnected",
      overrides: { webSocketStatus: "CLOSED" as const },
    },
  ])("disables mode changes when $condition", async ({ overrides }) => {
    const user = userEvent.setup();
    renderButton(overrides);

    const button = getModeButton();
    expect(button).toBeDisabled();
    expect(button).toHaveClass("opacity-50", "cursor-not-allowed");
    [
      "cursor-pointer",
      "hover:text-white",
      "hover:bg-white/10",
      "text-white",
      "hover:bg-[#597FF4]",
    ].forEach((interactiveClass) => {
      expect(button).not.toHaveClass(interactiveClass);
    });

    await user.click(button);
    fireEvent.keyDown(document, { key: "Tab", shiftKey: true });

    expect(
      screen.queryByTestId("change-agent-context-menu"),
    ).not.toBeInTheDocument();
    expect(mocks.handlePlanClick).not.toHaveBeenCalled();
  });

  it("keeps disabled Plan styling non-interactive", () => {
    renderButton({
      conversationMode: "plan",
      isCreatingConversation: true,
    });

    expect(getModeButton()).toHaveClass(
      "border",
      "border-[#597FF4]",
      "bg-[#4A67BD]",
      "opacity-50",
      "cursor-not-allowed",
    );
    expect(getModeButton()).not.toHaveClass("border-transparent");
  });

  it.each([
    {
      transition: "the agent starts running",
      applyTransition: () => {
        mocks.agentState = AgentState.RUNNING;
      },
    },
    {
      transition: "the websocket disconnects",
      applyTransition: () => {
        mocks.webSocketStatus = "CLOSED";
      },
    },
  ])("closes an open menu when $transition", async ({ applyTransition }) => {
    const user = userEvent.setup();
    const { rerender } = renderButton();

    await user.click(getModeButton());
    expect(screen.getByTestId("change-agent-context-menu")).toBeInTheDocument();

    applyTransition();
    rerender(<ChangeAgentButton />);

    await waitFor(() => {
      expect(
        screen.queryByTestId("change-agent-context-menu"),
      ).not.toBeInTheDocument();
    });
    expect(getModeButton()).toBeDisabled();
  });

  it("explains why mode selection is unavailable before a conversation starts", () => {
    renderButton({ conversationId: null });

    expect(getModeButton()).toBeDisabled();
    expect(screen.getByTestId("styled-tooltip-content")).toHaveTextContent(
      "CHANGE_AGENT$SWITCH_AFTER_CONVERSATION",
    );
  });
});
