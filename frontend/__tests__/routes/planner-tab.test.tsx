import { screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import PlannerTab from "#/routes/planner-tab";
import { renderWithProviders } from "../../test-utils";
import { useConversationStore } from "#/stores/conversation-store";

// Mock the handle plan click hook
vi.mock("#/hooks/use-handle-plan-click", () => ({
  useHandlePlanClick: () => ({
    handlePlanClick: vi.fn(),
  }),
}));

describe("PlannerTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset store state to defaults
    useConversationStore.setState({
      planContent: null,
      conversationMode: "code",
    });
  });

  describe("Create a plan button", () => {
    it("should be enabled when conversation mode is 'code'", () => {
      // Arrange
      useConversationStore.setState({
        planContent: null,
        conversationMode: "code",
      });

      // Act
      renderWithProviders(<PlannerTab />);

      // Assert
      const button = screen.getByRole("button");
      expect(button).not.toBeDisabled();
    });

    it("should be disabled when conversation mode is 'plan'", () => {
      // Arrange
      useConversationStore.setState({
        planContent: null,
        conversationMode: "plan",
      });

      // Act
      renderWithProviders(<PlannerTab />);

      // Assert
      const button = screen.getByRole("button");
      expect(button).toBeDisabled();
    });
  });
});
