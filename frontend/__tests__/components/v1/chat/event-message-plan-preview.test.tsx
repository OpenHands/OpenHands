import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { screen, render } from "@testing-library/react";
import { EventMessage } from "#/components/v1/chat/event-message";
import { useConversationStore } from "#/stores/conversation-store";
import {
  renderWithProviders,
  createPlanningObservationEvent,
} from "test-utils";

// Mock the feature flag
vi.mock("#/utils/feature-flags", () => ({
  USE_PLANNING_AGENT: vi.fn(() => true),
}));

// Mock useConfig
vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({
    data: { APP_MODE: "saas" },
  }),
}));

// Mock PlanPreview component to verify it's rendered
vi.mock("#/components/features/chat/plan-preview", () => ({
  PlanPreview: ({ planContent }: { planContent?: string | null }) => (
    <div data-testid="plan-preview">Plan Preview: {planContent || "null"}</div>
  ),
}));

describe("EventMessage - PlanPreview rendering", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Reset conversation store
    useConversationStore.setState({
      planContent: null,
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
});
