import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
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
  useOptionalConversationId: () => ({
    conversationId: "test-conversation-id",
  }),
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

  it("labels the dot with the status and opens no menu when clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ConversationNameWithStatus />);

    const status = screen.getByTestId("conversation-status");
    await user.click(status);

    expect(status).toHaveAccessibleName("COMMON$RUNNING");
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
  });
});
