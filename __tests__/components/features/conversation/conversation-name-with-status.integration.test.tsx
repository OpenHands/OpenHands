import { fireEvent, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { ConversationNameWithStatus } from "#/components/features/conversation/conversation-name-with-status";
import { AgentState } from "#/types/agent-state";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { useAgentState } from "#/hooks/use-agent-state";
import type { SandboxStatus } from "#/api/conversation-service/agent-server-conversation-service.types";
import { OH_STATUS_ERROR_COLOR } from "#/constants/status-colors";

const mocks = vi.hoisted(() => ({
  conversation: {} as {
    execution_status: ExecutionStatus | null;
    sandbox_status?: SandboxStatus | null;
  },
  pauseConversation: vi.fn(),
  resumeConversation: vi.fn(),
}));

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: vi.fn(),
}));

vi.mock("#/hooks/query/use-task-polling", () => ({
  useTaskPolling: () => ({
    isTask: false,
    taskStatus: null,
  }),
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      id: "test-conversation-id",
      title: "Test conversation",
      ...mocks.conversation,
    },
  }),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "test-conversation-id" }),
  useOptionalConversationId: () => ({
    conversationId: "test-conversation-id",
  }),
}));

vi.mock("#/hooks/mutation/use-unified-stop-conversation", () => ({
  useUnifiedPauseConversation: () => ({ mutate: mocks.pauseConversation }),
}));

vi.mock("#/hooks/mutation/use-unified-start-conversation", () => ({
  useUnifiedResumeConversation: () => ({ mutate: mocks.resumeConversation }),
}));

vi.mock("#/hooks/use-user-providers", () => ({
  useUserProviders: () => ({ providers: [] }),
}));

vi.mock("#/components/features/conversation/conversation-name", () => ({
  ConversationName: () => <div data-testid="conversation-name" />,
}));

vi.mock("#/components/features/conversation/right-panel-toggle", () => ({
  RightPanelToggle: () => null,
}));

vi.mock("react-i18next", async () => {
  const actual = await vi.importActual("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => key,
      i18n: { changeLanguage: () => Promise.resolve() },
    }),
  };
});

describe("ConversationNameWithStatus", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.conversation = { execution_status: ExecutionStatus.RUNNING };
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.RUNNING,
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("opens the server status menu on click and closes on outside click", async () => {
    const user = userEvent.setup();

    renderWithProviders(
      <div>
        <div data-testid="outside" />
        <ConversationNameWithStatus />
      </div>,
    );

    expect(
      screen.queryByTestId("server-status-context-menu"),
    ).not.toBeInTheDocument();

    // fireEvent click avoids a preceding mouse pointerenter (hover-open), which
    // models the touch/toggle path this assertion covers.
    fireEvent.click(screen.getByTestId("server-status-menu-trigger"));

    expect(
      screen.getByTestId("server-status-context-menu"),
    ).toBeInTheDocument();

    await user.click(screen.getByTestId("outside"));

    expect(
      screen.queryByTestId("server-status-context-menu"),
    ).not.toBeInTheDocument();
  });

  it("opens on mouse pointerenter and dismisses on a following mouse click", () => {
    renderWithProviders(<ConversationNameWithStatus />);

    const trigger = screen.getByTestId("server-status-menu-trigger");
    const hoverTarget = trigger.parentElement;
    expect(hoverTarget).not.toBeNull();

    // pointerover is the event React delegates for onPointerEnter synthesis,
    // so it reaches the handler reliably across jsdom setups.
    fireEvent.pointerOver(hoverTarget!, { pointerType: "mouse" });
    expect(
      screen.getByTestId("server-status-context-menu"),
    ).toBeInTheDocument();

    fireEvent.click(trigger);
    expect(
      screen.queryByTestId("server-status-context-menu"),
    ).not.toBeInTheDocument();
  });

  it("keeps click-toggle open after touch/compatibility hover events on fine-hover hardware", () => {
    // Device-primary pointer reports fine hover, but the interaction is touch.
    vi.stubGlobal(
      "matchMedia",
      vi.fn().mockReturnValue({
        matches: true,
        media: "(hover: hover) and (pointer: fine)",
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
      }),
    );

    renderWithProviders(<ConversationNameWithStatus />);

    const trigger = screen.getByTestId("server-status-menu-trigger");
    const hoverTarget = trigger.parentElement;
    expect(hoverTarget).not.toBeNull();

    // Compatibility mouseenter must not open-then-close against the tap.
    fireEvent.mouseEnter(hoverTarget!);
    fireEvent.pointerOver(hoverTarget!, { pointerType: "touch" });
    fireEvent.click(trigger);

    expect(
      screen.getByTestId("server-status-context-menu"),
    ).toBeInTheDocument();
  });

  describe("runtime availability", () => {
    function openMenu() {
      fireEvent.click(screen.getByTestId("server-status-menu-trigger"));
      return screen.getByTestId("server-status-context-menu");
    }

    it("shows a missing runtime as archived and offers no runtime action", () => {
      mocks.conversation = {
        execution_status: ExecutionStatus.IDLE,
        sandbox_status: "MISSING",
      };
      vi.mocked(useAgentState).mockReturnValue({
        curAgentState: AgentState.AWAITING_USER_INPUT,
      });
      renderWithProviders(<ConversationNameWithStatus />);

      const trigger = screen.getByTestId("server-status-menu-trigger");
      expect(
        within(trigger).getByTestId("conversation-status-archived"),
      ).toBeInTheDocument();

      const menu = openMenu();
      expect(within(menu).getByTestId("server-status")).toHaveTextContent(
        "COMMON$ARCHIVED",
      );
      expect(menu).not.toHaveTextContent("COMMON$RUNNING");
      expect(
        within(menu).queryByTestId("stop-server-button"),
      ).not.toBeInTheDocument();
      expect(
        within(menu).queryByTestId("start-server-button"),
      ).not.toBeInTheDocument();
      expect(mocks.pauseConversation).not.toHaveBeenCalled();
      expect(mocks.resumeConversation).not.toHaveBeenCalled();
    });

    it("keeps the red error presentation for an errored sandbox", () => {
      mocks.conversation = {
        execution_status: null,
        sandbox_status: "ERROR",
      };
      vi.mocked(useAgentState).mockReturnValue({
        curAgentState: AgentState.ERROR,
      });
      renderWithProviders(<ConversationNameWithStatus />);

      const trigger = screen.getByTestId("server-status-menu-trigger");
      expect(
        within(trigger).queryByTestId("conversation-status-archived"),
      ).not.toBeInTheDocument();
      expect(trigger.querySelector("svg")).toHaveAttribute(
        "color",
        OH_STATUS_ERROR_COLOR,
      );

      const menu = openMenu();
      expect(within(menu).getByTestId("server-status")).toHaveTextContent(
        "COMMON$ERROR",
      );
      expect(menu).not.toHaveTextContent("COMMON$ARCHIVED");
    });

    it.each([
      {
        name: "running",
        executionStatus: ExecutionStatus.RUNNING,
        sandboxStatus: "RUNNING" as const,
        agentState: AgentState.RUNNING,
        label: "COMMON$RUNNING",
        action: "stop-server-button",
        absentAction: "start-server-button",
      },
      {
        name: "paused",
        executionStatus: ExecutionStatus.PAUSED,
        sandboxStatus: "PAUSED" as const,
        agentState: AgentState.PAUSED,
        label: "COMMON$SERVER_STOPPED",
        action: "start-server-button",
        absentAction: "stop-server-button",
      },
    ])(
      "keeps the $name runtime controls when the runtime is available",
      ({
        executionStatus,
        sandboxStatus,
        agentState,
        label,
        action,
        absentAction,
      }) => {
        mocks.conversation = {
          execution_status: executionStatus,
          sandbox_status: sandboxStatus,
        };
        vi.mocked(useAgentState).mockReturnValue({ curAgentState: agentState });
        renderWithProviders(<ConversationNameWithStatus />);

        expect(
          screen.queryByTestId("conversation-status-archived"),
        ).not.toBeInTheDocument();

        const menu = openMenu();
        expect(within(menu).getByTestId("server-status")).toHaveTextContent(
          label,
        );
        expect(within(menu).getByTestId(action)).toBeInTheDocument();
        expect(
          within(menu).queryByTestId(absentAction),
        ).not.toBeInTheDocument();
      },
    );
  });
});
