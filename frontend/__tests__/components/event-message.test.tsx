import { afterEach, describe, expect, it, vi } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { EventMessage } from "#/components/features/chat/event-message";

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({
    data: { app_mode: "saas" },
  }),
}));

vi.mock("#/hooks/query/use-feedback-exists", () => ({
  useFeedbackExists: (eventId: number | undefined) => ({
    data: { exists: false },
    isLoading: false,
  }),
}));

describe("EventMessage", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render LikertScale for finish action when it's the last message", () => {
    const finishEvent = {
      id: 123,
      source: "agent" as const,
      action: "finish" as const,
      args: {
        final_thought: "Task completed successfully",
        outputs: {},
        thought: "Task completed successfully",
      },
      message: "Task completed successfully",
      timestamp: new Date().toISOString(),
    };

    renderWithProviders(
      <EventMessage
        event={finishEvent}
        hasObservationPair={false}
        isAwaitingUserConfirmation={false}
        isLastMessage={true}
        isInLast10Actions={true}
      />
    );

    expect(screen.getByLabelText("Rate 1 stars")).toBeInTheDocument();
    expect(screen.getByLabelText("Rate 5 stars")).toBeInTheDocument();
  });

  it("should render LikertScale for assistant message when it's the last message", () => {
    const assistantMessageEvent = {
      id: 456,
      source: "agent" as const,
      action: "message" as const,
      args: {
        thought: "I need more information to proceed.",
        image_urls: null,
        file_urls: [],
        wait_for_response: true,
      },
      message: "I need more information to proceed.",
      timestamp: new Date().toISOString(),
    };

    renderWithProviders(
      <EventMessage
        event={assistantMessageEvent}
        hasObservationPair={false}
        isAwaitingUserConfirmation={false}
        isLastMessage={true}
        isInLast10Actions={true}
      />
    );

    expect(screen.getByLabelText("Rate 1 stars")).toBeInTheDocument();
    expect(screen.getByLabelText("Rate 5 stars")).toBeInTheDocument();
  });

  it("should render LikertScale for error observation when it's the last message", () => {
    const errorEvent = {
      id: 789,
      source: "user" as const,
      observation: "error" as const,
      content: "An error occurred",
      extras: {
        error_id: "test-error-123",
      },
      message: "An error occurred",
      timestamp: new Date().toISOString(),
      cause: 123,
    };

    renderWithProviders(
      <EventMessage
        event={errorEvent}
        hasObservationPair={false}
        isAwaitingUserConfirmation={false}
        isLastMessage={true}
        isInLast10Actions={true}
      />
    );

    expect(screen.getByLabelText("Rate 1 stars")).toBeInTheDocument();
    expect(screen.getByLabelText("Rate 5 stars")).toBeInTheDocument();
  });

  it("should NOT render LikertScale when not the last message", () => {
    const finishEvent = {
      id: 101,
      source: "agent" as const,
      action: "finish" as const,
      args: {
        final_thought: "Task completed successfully",
        outputs: {},
        thought: "Task completed successfully",
      },
      message: "Task completed successfully",
      timestamp: new Date().toISOString(),
    };

    renderWithProviders(
      <EventMessage
        event={finishEvent}
        hasObservationPair={false}
        isAwaitingUserConfirmation={false}
        isLastMessage={false}
        isInLast10Actions={false}
      />
    );

    expect(screen.queryByLabelText("Rate 1 stars")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Rate 5 stars")).not.toBeInTheDocument();
  });

  it("should render LikertScale for error observation when in last 10 actions but not last message", () => {
    const errorEvent = {
      id: 999,
      source: "user" as const,
      observation: "error" as const,
      content: "An error occurred",
      extras: {
        error_id: "test-error-456",
      },
      message: "An error occurred",
      timestamp: new Date().toISOString(),
      cause: 123,
    };

    renderWithProviders(
      <EventMessage
        event={errorEvent}
        hasObservationPair={false}
        isAwaitingUserConfirmation={false}
        isLastMessage={false}
        isInLast10Actions={true}
      />
    );

    expect(screen.getByLabelText("Rate 1 stars")).toBeInTheDocument();
    expect(screen.getByLabelText("Rate 5 stars")).toBeInTheDocument();
  });

  it("should NOT render LikertScale for error observation when not in last 10 actions", () => {
    const errorEvent = {
      id: 888,
      source: "user" as const,
      observation: "error" as const,
      content: "An error occurred",
      extras: {
        error_id: "test-error-789",
      },
      message: "An error occurred",
      timestamp: new Date().toISOString(),
      cause: 123,
    };

    renderWithProviders(
      <EventMessage
        event={errorEvent}
        hasObservationPair={false}
        isAwaitingUserConfirmation={false}
        isLastMessage={false}
        isInLast10Actions={false}
      />
    );

    expect(screen.queryByLabelText("Rate 1 stars")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Rate 5 stars")).not.toBeInTheDocument();
  });
});
