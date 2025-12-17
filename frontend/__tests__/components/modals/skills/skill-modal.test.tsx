import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { renderWithProviders } from "test-utils";
import { SkillsModal } from "#/components/features/conversation-panel/skills-modal";
import ConversationService from "#/api/conversation-service/conversation-service.api";
import V1ConversationService from "#/api/conversation-service/v1-conversation-service.api";
import { AgentState } from "#/types/agent-state";
import { useAgentState } from "#/hooks/use-agent-state";
import SettingsService from "#/api/settings-service/settings-service.api";

// Mock the agent state hook
vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: vi.fn(),
}));

// Mock the conversation ID hook
vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "test-conversation-id" }),
}));

describe("SkillsModal - Refresh Button", () => {
  const mockOnClose = vi.fn();
  const conversationId = "test-conversation-id";

  const defaultProps = {
    onClose: mockOnClose,
    conversationId,
  };

  const mockSkills = [
    {
      name: "Test Agent 1",
      type: "repo" as const,
      triggers: ["test", "example"],
      content: "This is test content for agent 1",
    },
    {
      name: "Test Agent 2",
      type: "knowledge" as const,
      triggers: ["help", "support"],
      content: "This is test content for agent 2",
    },
  ];

  beforeEach(() => {
    // Reset all mocks before each test
    vi.clearAllMocks();

    // Setup default mock for getMicroagents (V0)
    vi.spyOn(ConversationService, "getMicroagents").mockResolvedValue({
      microagents: mockSkills,
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
      const refreshSpy = vi.spyOn(ConversationService, "getMicroagents");

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
});

describe("useConversationSkills - V1 API Integration", () => {
  const conversationId = "test-conversation-id";

  const mockMicroagents = [
    {
      name: "V0 Test Agent",
      type: "repo" as const,
      triggers: ["v0"],
      content: "V0 skill content",
    },
  ];

  const mockSkills = [
    {
      name: "V1 Test Skill",
      type: "knowledge" as const,
      triggers: ["v1", "skill"],
      content: "V1 skill content",
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();

    // Mock agent state
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.AWAITING_USER_INPUT,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("V0 API Usage (v1_enabled: false)", () => {
    it("should call v0 ConversationService.getMicroagents when v1_enabled is false", async () => {
      // Arrange
      const getMicroagentsSpy = vi
        .spyOn(ConversationService, "getMicroagents")
        .mockResolvedValue({ microagents: mockMicroagents });

      vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
        v1_enabled: false,
        llm_model: "test-model",
        llm_base_url: "",
        agent: "test-agent",
        language: "en",
        llm_api_key: null,
        llm_api_key_set: false,
        search_api_key_set: false,
        confirmation_mode: false,
        security_analyzer: null,
        remote_runtime_resource_factor: null,
        provider_tokens_set: {},
        enable_default_condenser: false,
        condenser_max_size: null,
        enable_sound_notifications: false,
        enable_proactive_conversation_starters: false,
        enable_solvability_analysis: false,
        user_consents_to_analytics: null,
        max_budget_per_task: null,
      });

      // Act
      renderWithProviders(<SkillsModal onClose={vi.fn()} />);

      // Assert
      await screen.findByText("V0 Test Agent");
      expect(getMicroagentsSpy).toHaveBeenCalledWith(conversationId);
      expect(getMicroagentsSpy).toHaveBeenCalledTimes(1);
    });

    it("should display v0 skills correctly", async () => {
      // Arrange
      vi.spyOn(ConversationService, "getMicroagents").mockResolvedValue({
        microagents: mockMicroagents,
      });

      vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
        v1_enabled: false,
        llm_model: "test-model",
        llm_base_url: "",
        agent: "test-agent",
        language: "en",
        llm_api_key: null,
        llm_api_key_set: false,
        search_api_key_set: false,
        confirmation_mode: false,
        security_analyzer: null,
        remote_runtime_resource_factor: null,
        provider_tokens_set: {},
        enable_default_condenser: false,
        condenser_max_size: null,
        enable_sound_notifications: false,
        enable_proactive_conversation_starters: false,
        enable_solvability_analysis: false,
        user_consents_to_analytics: null,
        max_budget_per_task: null,
      });

      // Act
      renderWithProviders(<SkillsModal onClose={vi.fn()} />);

      // Assert
      const agentName = await screen.findByText("V0 Test Agent");
      expect(agentName).toBeInTheDocument();
    });
  });

  describe("V1 API Usage (v1_enabled: true)", () => {
    it("should call v1 V1ConversationService.getSkills when v1_enabled is true", async () => {
      // Arrange
      const getSkillsSpy = vi
        .spyOn(V1ConversationService, "getSkills")
        .mockResolvedValue({ skills: mockSkills });

      vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
        v1_enabled: true,
        llm_model: "test-model",
        llm_base_url: "",
        agent: "test-agent",
        language: "en",
        llm_api_key: null,
        llm_api_key_set: false,
        search_api_key_set: false,
        confirmation_mode: false,
        security_analyzer: null,
        remote_runtime_resource_factor: null,
        provider_tokens_set: {},
        enable_default_condenser: false,
        condenser_max_size: null,
        enable_sound_notifications: false,
        enable_proactive_conversation_starters: false,
        enable_solvability_analysis: false,
        user_consents_to_analytics: null,
        max_budget_per_task: null,
      });

      // Act
      renderWithProviders(<SkillsModal onClose={vi.fn()} />);

      // Assert
      await screen.findByText("V1 Test Skill");
      expect(getSkillsSpy).toHaveBeenCalledWith(conversationId);
      expect(getSkillsSpy).toHaveBeenCalledTimes(1);
    });

    it("should display v1 skills correctly", async () => {
      // Arrange
      vi.spyOn(V1ConversationService, "getSkills").mockResolvedValue({
        skills: mockSkills,
      });

      vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
        v1_enabled: true,
        llm_model: "test-model",
        llm_base_url: "",
        agent: "test-agent",
        language: "en",
        llm_api_key: null,
        llm_api_key_set: false,
        search_api_key_set: false,
        confirmation_mode: false,
        security_analyzer: null,
        remote_runtime_resource_factor: null,
        provider_tokens_set: {},
        enable_default_condenser: false,
        condenser_max_size: null,
        enable_sound_notifications: false,
        enable_proactive_conversation_starters: false,
        enable_solvability_analysis: false,
        user_consents_to_analytics: null,
        max_budget_per_task: null,
      });

      // Act
      renderWithProviders(<SkillsModal onClose={vi.fn()} />);

      // Assert
      const skillName = await screen.findByText("V1 Test Skill");
      expect(skillName).toBeInTheDocument();
    });

    it("should use v1 API when v1_enabled is true", async () => {
      // Arrange
      vi.spyOn(SettingsService, "getSettings").mockResolvedValue({
        v1_enabled: true,
        llm_model: "test-model",
        llm_base_url: "",
        agent: "test-agent",
        language: "en",
        llm_api_key: null,
        llm_api_key_set: false,
        search_api_key_set: false,
        confirmation_mode: false,
        security_analyzer: null,
        remote_runtime_resource_factor: null,
        provider_tokens_set: {},
        enable_default_condenser: false,
        condenser_max_size: null,
        enable_sound_notifications: false,
        enable_proactive_conversation_starters: false,
        enable_solvability_analysis: false,
        user_consents_to_analytics: null,
        max_budget_per_task: null,
      });

      const getSkillsSpy = vi
        .spyOn(V1ConversationService, "getSkills")
        .mockResolvedValue({
          skills: mockSkills,
        });

      // Act
      renderWithProviders(<SkillsModal onClose={vi.fn()} />);

      // Assert
      await screen.findByText("V1 Test Skill");
      // Verify v1 API was called
      expect(getSkillsSpy).toHaveBeenCalledWith(conversationId);
    });
  });

  describe("API Switching on Settings Change", () => {
    it("should refetch using different API when v1_enabled setting changes", async () => {
      // Arrange
      const getMicroagentsSpy = vi
        .spyOn(ConversationService, "getMicroagents")
        .mockResolvedValue({ microagents: mockMicroagents });
      const getSkillsSpy = vi
        .spyOn(V1ConversationService, "getSkills")
        .mockResolvedValue({ skills: mockSkills });

      const settingsSpy = vi
        .spyOn(SettingsService, "getSettings")
        .mockResolvedValue({
          v1_enabled: false,
          llm_model: "test-model",
          llm_base_url: "",
          agent: "test-agent",
          language: "en",
          llm_api_key: null,
          llm_api_key_set: false,
          search_api_key_set: false,
          confirmation_mode: false,
          security_analyzer: null,
          remote_runtime_resource_factor: null,
          provider_tokens_set: {},
          enable_default_condenser: false,
          condenser_max_size: null,
          enable_sound_notifications: false,
          enable_proactive_conversation_starters: false,
          enable_solvability_analysis: false,
          user_consents_to_analytics: null,
          max_budget_per_task: null,
        });

      // Act - Initial render with v1_enabled: false
      const { rerender } = renderWithProviders(
        <SkillsModal onClose={vi.fn()} />,
      );

      // Assert - v0 API called initially
      await screen.findByText("V0 Test Agent");
      expect(getMicroagentsSpy).toHaveBeenCalledWith(conversationId);

      // Arrange - Change settings to v1_enabled: true
      settingsSpy.mockResolvedValue({
        v1_enabled: true,
        llm_model: "test-model",
        llm_base_url: "",
        agent: "test-agent",
        language: "en",
        llm_api_key: null,
        llm_api_key_set: false,
        search_api_key_set: false,
        confirmation_mode: false,
        security_analyzer: null,
        remote_runtime_resource_factor: null,
        provider_tokens_set: {},
        enable_default_condenser: false,
        condenser_max_size: null,
        enable_sound_notifications: false,
        enable_proactive_conversation_starters: false,
        enable_solvability_analysis: false,
        user_consents_to_analytics: null,
        max_budget_per_task: null,
      });

      // Act - Force re-render
      rerender(<SkillsModal onClose={vi.fn()} />);

      // Assert - v1 API should be called after settings change
      await screen.findByText("V1 Test Skill");
      expect(getSkillsSpy).toHaveBeenCalledWith(conversationId);
    });
  });
});
