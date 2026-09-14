import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { screen } from "@testing-library/react";
import { EventMessage } from "#/components/conversation-events/chat/event-message";
import { useConversationStore } from "#/stores/conversation-store";
import { useAgentState, usePlanningAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";
import {
  renderWithProviders,
  createPlanningObservationEvent,
} from "test-utils";

// Mock useConfig
vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({
    data: {},
  }),
}));

// Mock useAgentState (main) and usePlanningAgentState (planner) independently
// so each test controls exactly one without the other's default bleeding in.
vi.mock("#/hooks/use-agent-state", () => ({
  useAgentState: vi.fn(),
  usePlanningAgentState: vi.fn(),
}));

// Mock useConversationId (EventMessage -> useAgentState -> useActiveConversation -> useConversationId)
vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({ conversationId: "test-conversation-id" }),
  useConversationId: () => ({ conversationId: "test-conversation-id" }),
}));

// Mock PlanPreview component to verify it's rendered
vi.mock("#/components/features/chat/plan-preview", () => ({
  PlanPreview: ({
    planContent,
    isStreaming,
  }: {
    planContent?: string | null;
    isStreaming?: boolean;
  }) => (
    <div data-testid="plan-preview" data-is-streaming={isStreaming}>
      Plan Preview: {planContent || "null"}
    </div>
  ),
}));

describe("EventMessage - PlanPreview rendering", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useConversationStore.setState({ planContent: null });
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.INIT,
    });
    // A resolved planner id by default, idle, so the isStreaming-specific
    // tests below only need to vary isLastMessage / planning agent state.
    vi.mocked(usePlanningAgentState).mockReturnValue({
      localPlanningConversationId: "planner-1",
      curPlanningAgentState: AgentState.INIT,
      isPlanningAgentRunning: false,
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render PlanPreview when PlanningFileEditorObservation event ID is in planPreviewEventIds", () => {
    const event = createPlanningObservationEvent("plan-obs-1");
    const planPreviewEventIds = new Set(["plan-obs-1"]);
    const planContent = "This is the plan content";

    useConversationStore.setState({ planContent });

    renderWithProviders(
      <EventMessage
        event={event}
        messages={[]}
        isLastMessage={false}
        isInLast10Actions={false}
        planPreviewEventIds={planPreviewEventIds}
      />,
    );

    expect(screen.getByTestId("plan-preview")).toBeInTheDocument();
    expect(
      screen.getByText(`Plan Preview: ${planContent}`),
    ).toBeInTheDocument();
  });

  it("should return null when PlanningFileEditorObservation event ID is NOT in planPreviewEventIds", () => {
    const event = createPlanningObservationEvent("plan-obs-1");
    const planPreviewEventIds = new Set(["plan-obs-2"]); // Different ID

    const { container } = renderWithProviders(
      <EventMessage
        event={event}
        messages={[]}
        isLastMessage={false}
        isInLast10Actions={false}
        planPreviewEventIds={planPreviewEventIds}
      />,
    );

    expect(screen.queryByTestId("plan-preview")).not.toBeInTheDocument();
    expect(container.firstChild).toBeNull();
  });

  it("should return null when planPreviewEventIds is undefined", () => {
    const event = createPlanningObservationEvent("plan-obs-1");

    const { container } = renderWithProviders(
      <EventMessage
        event={event}
        messages={[]}
        isLastMessage={false}
        isInLast10Actions={false}
        planPreviewEventIds={undefined}
      />,
    );

    expect(screen.queryByTestId("plan-preview")).not.toBeInTheDocument();
    expect(container.firstChild).toBeNull();
  });

  it("should use planContent from conversation store", () => {
    const event = createPlanningObservationEvent("plan-obs-1");
    const planPreviewEventIds = new Set(["plan-obs-1"]);
    const planContent = "Store plan content";

    useConversationStore.setState({ planContent });

    renderWithProviders(
      <EventMessage
        event={event}
        messages={[]}
        isLastMessage={false}
        isInLast10Actions={false}
        planPreviewEventIds={planPreviewEventIds}
      />,
    );

    expect(
      screen.getByText(`Plan Preview: ${planContent}`),
    ).toBeInTheDocument();
  });

  it("should handle null planContent from store", () => {
    const event = createPlanningObservationEvent("plan-obs-1");
    const planPreviewEventIds = new Set(["plan-obs-1"]);

    useConversationStore.setState({ planContent: null });

    renderWithProviders(
      <EventMessage
        event={event}
        messages={[]}
        isLastMessage={false}
        isInLast10Actions={false}
        planPreviewEventIds={planPreviewEventIds}
      />,
    );

    expect(screen.getByTestId("plan-preview")).toBeInTheDocument();
    expect(screen.getByText("Plan Preview: null")).toBeInTheDocument();
  });

  it("should handle empty planPreviewEventIds set", () => {
    const event = createPlanningObservationEvent("plan-obs-1");
    const planPreviewEventIds = new Set<string>();

    const { container } = renderWithProviders(
      <EventMessage
        event={event}
        messages={[]}
        isLastMessage={false}
        isInLast10Actions={false}
        planPreviewEventIds={planPreviewEventIds}
      />,
    );

    expect(screen.queryByTestId("plan-preview")).not.toBeInTheDocument();
    expect(container.firstChild).toBeNull();
  });

  describe("isStreaming prop", () => {
    it("should pass isStreaming=true when isLastMessage is true and planner is RUNNING", () => {
      const event = createPlanningObservationEvent("plan-obs-1");
      const planPreviewEventIds = new Set(["plan-obs-1"]);
      const planContent = "Streaming plan content";

      useConversationStore.setState({ planContent });
      vi.mocked(usePlanningAgentState).mockReturnValue({
        localPlanningConversationId: "planner-1",
        curPlanningAgentState: AgentState.RUNNING,
        isPlanningAgentRunning: true,
      });

      renderWithProviders(
        <EventMessage
          event={event}
          messages={[]}
          isLastMessage={true}
          isInLast10Actions={false}
          planPreviewEventIds={planPreviewEventIds}
        />,
      );

      const planPreview = screen.getByTestId("plan-preview");
      expect(planPreview).toBeInTheDocument();
      expect(planPreview).toHaveAttribute("data-is-streaming", "true");
    });

    it("should pass isStreaming=false when isLastMessage is false even if planner is RUNNING", () => {
      const event = createPlanningObservationEvent("plan-obs-1");
      const planPreviewEventIds = new Set(["plan-obs-1"]);
      const planContent = "Plan content";

      useConversationStore.setState({ planContent });
      vi.mocked(usePlanningAgentState).mockReturnValue({
        localPlanningConversationId: "planner-1",
        curPlanningAgentState: AgentState.RUNNING,
        isPlanningAgentRunning: true,
      });

      renderWithProviders(
        <EventMessage
          event={event}
          messages={[]}
          isLastMessage={false}
          isInLast10Actions={false}
          planPreviewEventIds={planPreviewEventIds}
        />,
      );

      const planPreview = screen.getByTestId("plan-preview");
      expect(planPreview).toBeInTheDocument();
      expect(planPreview).toHaveAttribute("data-is-streaming", "false");
    });

    it("should pass isStreaming=false when planner state is not RUNNING even if isLastMessage is true", () => {
      const event = createPlanningObservationEvent("plan-obs-1");
      const planPreviewEventIds = new Set(["plan-obs-1"]);
      const planContent = "Completed plan content";

      useConversationStore.setState({ planContent });
      vi.mocked(usePlanningAgentState).mockReturnValue({
        localPlanningConversationId: "planner-1",
        curPlanningAgentState: AgentState.AWAITING_USER_INPUT,
        isPlanningAgentRunning: false,
      });

      renderWithProviders(
        <EventMessage
          event={event}
          messages={[]}
          isLastMessage={true}
          isInLast10Actions={false}
          planPreviewEventIds={planPreviewEventIds}
        />,
      );

      const planPreview = screen.getByTestId("plan-preview");
      expect(planPreview).toBeInTheDocument();
      expect(planPreview).toHaveAttribute("data-is-streaming", "false");
    });

    it("should pass isStreaming=false when localPlanningConversationId hasn't resolved yet, even if isLastMessage and planner state are RUNNING", () => {
      // Regression: useAgentState(undefined) falls back to the route
      // conversation's own state, so without this guard the code agent's
      // activity could be mistaken for the planner's before the planner id
      // resolves (e.g. right after "Create a Plan" is clicked).
      const event = createPlanningObservationEvent("plan-obs-1");
      const planPreviewEventIds = new Set(["plan-obs-1"]);
      const planContent = "Plan content";

      useConversationStore.setState({ planContent });
      vi.mocked(usePlanningAgentState).mockReturnValue({
        localPlanningConversationId: null,
        curPlanningAgentState: AgentState.RUNNING,
        isPlanningAgentRunning: false,
      });

      renderWithProviders(
        <EventMessage
          event={event}
          messages={[]}
          isLastMessage
          isInLast10Actions={false}
          planPreviewEventIds={planPreviewEventIds}
        />,
      );

      const planPreview = screen.getByTestId("plan-preview");
      expect(planPreview).toBeInTheDocument();
      expect(planPreview).toHaveAttribute("data-is-streaming", "false");
    });

    it("should pass isStreaming=false when planner state is FINISHED", () => {
      const event = createPlanningObservationEvent("plan-obs-1");
      const planPreviewEventIds = new Set(["plan-obs-1"]);
      const planContent = "Finished plan content";

      useConversationStore.setState({ planContent });
      vi.mocked(usePlanningAgentState).mockReturnValue({
        localPlanningConversationId: "planner-1",
        curPlanningAgentState: AgentState.FINISHED,
        isPlanningAgentRunning: false,
      });

      renderWithProviders(
        <EventMessage
          event={event}
          messages={[]}
          isLastMessage={true}
          isInLast10Actions={false}
          planPreviewEventIds={planPreviewEventIds}
        />,
      );

      const planPreview = screen.getByTestId("plan-preview");
      expect(planPreview).toBeInTheDocument();
      expect(planPreview).toHaveAttribute("data-is-streaming", "false");
    });
  });
});
