import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { GoalStatusContent } from "#/components/features/chat/goal-status-content";
import { useGoalStore } from "#/stores/goal-store";
import type { GoalStatus } from "#/types/agent-server/core/events/conversation-state-event";

const stopGoal = vi.fn();
const resumeGoal = vi.fn();
const pauseConversation = vi.fn();

vi.mock("#/hooks/mutation/conversation-mutation-utils", () => ({
  stopGoal: (...a: unknown[]) => stopGoal(...a),
  resumeGoal: (...a: unknown[]) => resumeGoal(...a),
  pauseConversation: (...a: unknown[]) => pauseConversation(...a),
}));
vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "conv-1" }),
}));
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string) => key }),
}));

const status = (overrides: Partial<GoalStatus> = {}): GoalStatus =>
  ({
    active: false,
    status: "interrupted",
    iteration: 1,
    max_iterations: 10,
    objective: "do the thing",
    verdict: null,
    ...overrides,
  }) as GoalStatus;

describe("GoalStatusContent loop controls", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    stopGoal.mockResolvedValue(undefined);
    resumeGoal.mockResolvedValue(undefined);
    pauseConversation.mockResolvedValue(undefined);
    useGoalStore.setState({ statusByConversation: {} });
  });

  it("Stop cancels the loop and interrupts the agent on an active goal", async () => {
    render(
      <GoalStatusContent
        status={status({ active: true, status: "running" })}
      />,
    );
    await userEvent.click(screen.getByTestId("goal-stop"));
    await waitFor(() => expect(stopGoal).toHaveBeenCalledWith("conv-1"));
    await waitFor(() =>
      expect(pauseConversation).toHaveBeenCalledWith("conv-1"),
    );
  });

  it("Resume continues an interrupted goal", async () => {
    render(<GoalStatusContent status={status({ status: "interrupted" })} />);
    await userEvent.click(screen.getByTestId("goal-resume"));
    await waitFor(() => expect(resumeGoal).toHaveBeenCalledWith("conv-1"));
  });

  it("hides Resume while another goal is already active", () => {
    useGoalStore.setState({
      statusByConversation: {
        "conv-1": status({ active: true, status: "running" }),
      },
    });
    render(<GoalStatusContent status={status({ status: "interrupted" })} />);
    expect(screen.queryByTestId("goal-resume")).toBeNull();
  });

  it("shows no controls for a completed goal", () => {
    render(<GoalStatusContent status={status({ status: "complete" })} />);
    expect(screen.queryByTestId("goal-stop")).toBeNull();
    expect(screen.queryByTestId("goal-resume")).toBeNull();
  });
});
