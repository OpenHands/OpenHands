import { fireEvent, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import { ConversationNameWithStatus } from "#/components/features/conversation/conversation-name-with-status";
import { AgentState } from "#/types/agent-state";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";
import { useAgentState } from "#/hooks/use-agent-state";

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
      execution_status: ExecutionStatus.RUNNING,
    },
  }),
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "test-conversation-id" }),
}));

vi.mock("#/hooks/mutation/use-unified-stop-conversation", () => ({
  useUnifiedPauseConversation: () => ({ mutate: vi.fn() }),
}));

vi.mock("#/hooks/mutation/use-unified-start-conversation", () => ({
  useUnifiedResumeConversation: () => ({ mutate: vi.fn() }),
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
        <div data-testid="outside">Outside</div>
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

  it("toggles the server status menu closed when the trigger is clicked again", () => {
    renderWithProviders(<ConversationNameWithStatus />);

    const trigger = screen.getByTestId("server-status-menu-trigger");
    fireEvent.click(trigger);
    expect(
      screen.getByTestId("server-status-context-menu"),
    ).toBeInTheDocument();

    fireEvent.click(trigger);
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
});
