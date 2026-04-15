import { screen, act } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
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
  const originalRAF = global.requestAnimationFrame;

  beforeEach(() => {
    vi.clearAllMocks();
    // Make requestAnimationFrame execute synchronously for testing
    global.requestAnimationFrame = (cb: FrameRequestCallback) => {
      cb(0);
      return 0;
    };
    // Reset store state to defaults
    useConversationStore.setState({
      planContent: null,
      conversationMode: "code",
    });
  });

  afterEach(() => {
    global.requestAnimationFrame = originalRAF;
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

  describe("Auto-scroll behavior", () => {
    it("should scroll to bottom when plan content is updated", () => {
      // Arrange
      const scrollTopSetter = vi.fn();
      const mockScrollHeight = 500;

      // Mock scroll properties on HTMLElement prototype
      const originalScrollHeightDescriptor = Object.getOwnPropertyDescriptor(
        HTMLElement.prototype,
        "scrollHeight",
      );
      const originalScrollTopDescriptor = Object.getOwnPropertyDescriptor(
        HTMLElement.prototype,
        "scrollTop",
      );

      Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
        get: () => mockScrollHeight,
        configurable: true,
      });
      Object.defineProperty(HTMLElement.prototype, "scrollTop", {
        get: () => 0,
        set: scrollTopSetter,
        configurable: true,
      });

      try {
        // Render with initial plan content
        useConversationStore.setState({
          planContent: "# Initial Plan",
          conversationMode: "plan",
        });

        renderWithProviders(<PlannerTab />);

        // Clear calls from initial render
        scrollTopSetter.mockClear();

        // Act - Update plan content which should trigger auto-scroll
        act(() => {
          useConversationStore.setState({
            planContent: "# Updated Plan\n\nMore content added here.",
          });
        });

        // Assert - scrollTop should be set to scrollHeight
        expect(scrollTopSetter).toHaveBeenCalledWith(mockScrollHeight);
      } finally {
        // Restore original descriptors
        if (originalScrollHeightDescriptor) {
          Object.defineProperty(
            HTMLElement.prototype,
            "scrollHeight",
            originalScrollHeightDescriptor,
          );
        }
        if (originalScrollTopDescriptor) {
          Object.defineProperty(
            HTMLElement.prototype,
            "scrollTop",
            originalScrollTopDescriptor,
          );
        }
      }
    });
  });
});
