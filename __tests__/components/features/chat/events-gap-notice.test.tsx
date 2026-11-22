import { act, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { EventsGapNotice } from "#/components/features/chat/events-gap-notice";
import { useEventStore } from "#/stores/use-event-store";

describe("EventsGapNotice", () => {
  beforeEach(() => {
    act(() => useEventStore.getState().clearEvents());
  });

  it("renders nothing while no gap has been skipped", () => {
    render(<EventsGapNotice />);

    expect(screen.queryByTestId("events-gap-notice")).not.toBeInTheDocument();
  });

  it("announces the skipped interval once the cursor records a gap", () => {
    render(<EventsGapNotice />);

    act(() => {
      useEventStore.getState().recordSkippedEventRange(10, 14);
    });

    expect(screen.getByTestId("events-gap-notice")).toBeInTheDocument();
  });

  it("disappears when the conversation's events are cleared", () => {
    render(<EventsGapNotice />);

    act(() => {
      useEventStore.getState().recordSkippedEventRange(10, 14);
      useEventStore.getState().clearEventsForConversation("conv-2");
    });

    expect(screen.queryByTestId("events-gap-notice")).not.toBeInTheDocument();
  });
});
