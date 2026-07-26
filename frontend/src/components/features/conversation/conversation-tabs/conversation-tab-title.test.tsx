import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ConversationTabTitle } from "./conversation-tab-title";
import { useConversationStore } from "#/stores/conversation-store";
import { AgentState } from "#/types/agent-state";

const mockState = vi.hoisted(() => ({
  refetch: vi.fn(),
  isFetching: false,
  handleBuildPlanClick: vi.fn(),
  curAgentState: "STOPPED",
}));

vi.mock("#/hooks/query/use-unified-get-git-changes", () => ({
  useUnifiedGetGitChanges: () => ({
    refetch: mockState.refetch,
    isFetching: mockState.isFetching,
  }),
}));

vi.mock("#/hooks/use-handle-build-plan-click", () => ({
  useHandleBuildPlanClick: () => ({
    handleBuildPlanClick: mockState.handleBuildPlanClick,
  }),
}));

vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: () => ({
    curAgentState: mockState.curAgentState,
  }),
}));

describe("ConversationTabTitle", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockState.isFetching = false;
    mockState.curAgentState = AgentState.STOPPED;

    useConversationStore.setState({
      planContent: null,
      conversationMode: "plan",
    });

    Object.assign(navigator, {
      clipboard: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
    });
  });

  it("shows a copy button for planner tabs when plan content exists and copies raw markdown", async () => {
    useConversationStore.setState({
      planContent: "# Plan\n\n- Step 1\n- Step 2",
    });

    render(<ConversationTabTitle title="Planner" conversationKey="planner" />);

    const copyButton = screen.getByTestId("copy-to-clipboard");
    fireEvent.click(copyButton);

    await waitFor(() => {
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
        "# Plan\n\n- Step 1\n- Step 2",
      );
    });

    expect(copyButton).toHaveAttribute("aria-label", "BUTTON$COPIED");
  });

  it("hides the planner copy button and disables build when there is no plan content", () => {
    render(<ConversationTabTitle title="Planner" conversationKey="planner" />);

    expect(screen.queryByTestId("copy-to-clipboard")).not.toBeInTheDocument();
    expect(screen.getByTestId("planner-tab-build-button")).toBeDisabled();
  });

  it("keeps the build button disabled while the agent is running", () => {
    mockState.curAgentState = AgentState.RUNNING;
    useConversationStore.setState({
      planContent: "# Plan\n\n- Step 1",
    });

    render(<ConversationTabTitle title="Planner" conversationKey="planner" />);

    expect(screen.getByTestId("planner-tab-build-button")).toBeDisabled();
  });
});
