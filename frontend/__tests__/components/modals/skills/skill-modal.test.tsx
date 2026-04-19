import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderWithProviders } from "test-utils";
import { SkillsModal } from "#/components/features/conversation-panel/skills-modal";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { AgentState } from "#/types/agent-state";
import { useAgentState } from "#/hooks/use-agent-state";

// Mock the agent state hook
vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: vi.fn(),
}));

// Mock the conversation ID hook
vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "test-conversation-id" }),
}));

// Mock useActiveConversation to provide execution_status
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({
    data: {
      execution_status: "IDLE",
    },
  }),
}));

describe("SkillsModal", () => {
  const mockOnClose = vi.fn();
  const conversationId = "test-conversation-id";

  const defaultProps = {
    onClose: mockOnClose,
    conversationId,
  };

  const mockSkills = [
    {
      name: "Test Skill 1",
      type: "repo" as const,
      triggers: ["test", "example"],
      content: "This is test content for skill 1",
    },
    {
      name: "Test Skill 2",
      type: "knowledge" as const,
      triggers: ["help", "support"],
      content: "This is test content for skill 2",
    },
  ];

  beforeEach(() => {
    // Reset all mocks before each test
    vi.clearAllMocks();

    // Setup default mock for getSkills (V1)
    vi.spyOn(V1ConversationService, "getSkills").mockResolvedValue({
      skills: mockSkills,
    });

    // Mock the agent state to return a ready state
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.AWAITING_USER_INPUT,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Refresh Button Rendering", () => {
    it("should render the refresh button with correct text and test ID", async () => {
      renderWithProviders(<SkillsModal {...defaultProps} />);

      // Wait for the component to load and render the refresh button
      const refreshButton = await screen.findByTestId("refresh-skills");
      expect(refreshButton).toBeInTheDocument();
      expect(refreshButton).toHaveTextContent("BUTTON$REFRESH");
    });
  });

  describe("Refresh Button Functionality", () => {
    it("should call refetch when refresh button is clicked", async () => {
      const user = userEvent.setup();
      const refreshSpy = vi.spyOn(V1ConversationService, "getSkills");

      renderWithProviders(<SkillsModal {...defaultProps} />);

      // Wait for the component to load and render the refresh button
      const refreshButton = await screen.findByTestId("refresh-skills");

      // Clear previous calls to only track the click
      refreshSpy.mockClear();

      await user.click(refreshButton);

      // Verify the refresh triggered a new API call
      expect(refreshSpy).toHaveBeenCalled();
    });
  });

  describe("Skills Display", () => {
    it("should display skills correctly", async () => {
      vi.spyOn(V1ConversationService, "getSkills").mockResolvedValue({
        skills: mockSkills,
      });

      renderWithProviders(<SkillsModal {...defaultProps} />);

      // Wait for skills to be loaded
      await screen.findByText("Test Skill 1");
      expect(screen.getByText("Test Skill 1")).toBeInTheDocument();
      expect(screen.getByText("Test Skill 2")).toBeInTheDocument();
    });
  });
});

// Note: Tests for V0 API and v1_enabled settings were removed as the component
// now uses V1 API exclusively via useConversationSkills hook
