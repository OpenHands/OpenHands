import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderWithProviders } from "test-utils";
import { SkillsModal } from "#/components/features/conversation-panel/skills-modal";
import SkillsService from "#/api/skills-service";

describe("SkillsModal", () => {
  const mockOnClose = vi.fn();

  const defaultProps = {
    onClose: mockOnClose,
  };

  const mockSkills = [
    {
      name: "Test Skill 1",
      type: "repo" as const,
      source: null,
      triggers: ["test", "example"],
      content: "This is test content for skill 1",
    },
    {
      name: "Test Skill 2",
      type: "knowledge" as const,
      source: null,
      triggers: ["help", "support"],
      content: "This is test content for skill 2",
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();

    vi.spyOn(SkillsService, "getSkills").mockResolvedValue(mockSkills);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Refresh Button Rendering", () => {
    it("should render the refresh button as an icon-only control with accessible label", async () => {
      renderWithProviders(<SkillsModal {...defaultProps} />);

      const refreshButton = await screen.findByTestId("refresh-skills");
      expect(refreshButton).toBeInTheDocument();
      expect(refreshButton).toHaveAttribute(
        "aria-label",
        "BUTTON$REFRESH",
      );
      expect(refreshButton).not.toHaveTextContent("BUTTON$REFRESH");
    });
  });

  describe("Close Button", () => {
    it("should render the close button and call onClose when clicked", async () => {
      const user = userEvent.setup();
      renderWithProviders(<SkillsModal {...defaultProps} />);

      const closeButton = await screen.findByTestId("close-skills-modal");
      expect(closeButton).toBeInTheDocument();

      await user.click(closeButton);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });
  });

  describe("Refresh Button Functionality", () => {
    it("should call refetch when refresh button is clicked", async () => {
      const user = userEvent.setup();
      const refreshSpy = vi.spyOn(SkillsService, "getSkills");

      renderWithProviders(<SkillsModal {...defaultProps} />);

      const refreshButton = await screen.findByTestId("refresh-skills");

      refreshSpy.mockClear();

      await user.click(refreshButton);

      expect(refreshSpy).toHaveBeenCalled();
    });
  });

  describe("Skills Display", () => {
    it("displays the skills catalog when opened with no active conversation (home page)", async () => {
      // Arrange: the catalog fetch succeeds; no conversation exists in this
      // render, so no runtime ever starts (the original infinite-spinner bug)
      vi.spyOn(SkillsService, "getSkills").mockResolvedValue(mockSkills);

      // Act
      renderWithProviders(<SkillsModal {...defaultProps} />);

      // Assert: the list renders instead of waiting for a runtime
      expect(await screen.findByText("Test Skill 1")).toBeInTheDocument();
      expect(screen.getByText("Test Skill 2")).toBeInTheDocument();
    });

    it("surfaces a fetch error message when the skills catalog cannot be loaded", async () => {
      // Arrange: the catalog fetch fails (e.g. agent server unreachable)
      vi.spyOn(SkillsService, "getSkills").mockRejectedValue(
        new Error("network error"),
      );

      // Act
      renderWithProviders(<SkillsModal {...defaultProps} />);

      // Assert: a clear failure message is shown instead of an endless spinner
      expect(await screen.findByText("COMMON$FETCH_ERROR")).toBeInTheDocument();
    });
  });
});
