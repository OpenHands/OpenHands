import { act, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { DeepPlanPanel } from "#/components/features/chat/deep-plan-panel";
import { useConversationStore } from "#/stores/conversation-store";
import { renderWithProviders } from "../../../../test-utils";

describe("DeepPlanPanel", () => {
  beforeEach(() => {
    useConversationStore.setState({
      deepPlan: { activePhase: null, confirmed: [], documents: {} },
    });
  });

  it("offers to open the chain when no phase is active", async () => {
    renderWithProviders(<DeepPlanPanel />);

    await userEvent.click(screen.getByRole("button"));

    expect(useConversationStore.getState().deepPlan.activePhase).toBe(
      "analysis",
    );
  });

  it("locks a later phase until every earlier phase is confirmed", () => {
    act(() => useConversationStore.getState().startDeepPlan());

    renderWithProviders(<DeepPlanPanel />);

    expect(screen.getByTestId("deep-plan-phase-analysis")).not.toBeDisabled();
    expect(screen.getByTestId("deep-plan-phase-database")).toBeDisabled();
    expect(screen.getByTestId("deep-plan-phase-database")).toHaveAttribute(
      "data-state",
      "locked",
    );
  });

  it("refuses the checkpoint and names the dangling reference", async () => {
    const store = useConversationStore.getState();
    act(() => {
      store.startDeepPlan();
      store.setDeepPlanDocument("requirements", "## 3.1 Authentication\n");
      store.confirmDeepPlanPhase("analysis");
      store.confirmDeepPlanPhase("requirements");
      store.setDeepPlanDocument("database", "## 2.1 Users [Req 9.9]\n");
    });

    renderWithProviders(<DeepPlanPanel />);

    await userEvent.click(screen.getByTestId("deep-plan-confirm"));

    expect(screen.getByTestId("deep-plan-error")).toHaveTextContent(
      "[Req 9.9]",
    );
    expect(useConversationStore.getState().deepPlan.activePhase).toBe(
      "database",
    );
  });
});
