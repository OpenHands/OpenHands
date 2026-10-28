import React from "react";
import { render, screen, waitFor, act } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { ScrollProvider } from "#/context/scroll-context";
import { useJumpToEvent } from "#/hooks/chat/use-jump-to-event";

const mockToastError = vi.fn();
vi.mock("#/utils/custom-toast-handlers", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  displayErrorToast: (msg: string | null) => mockToastError(msg),
}));

const mockLoadOlder = vi.fn();
vi.mock("#/hooks/use-load-older-events", () => ({
  useLoadOlderEvents: () => ({
    isLoading: false,
    hasMore: true,
    loadOlder: mockLoadOlder,
  }),
}));

function JumpHarness({ eventId }: { eventId: string }) {
  const jump = useJumpToEvent();
  return (
    <button
      type="button"
      data-testid="jump"
      onClick={() => void jump(eventId)}
    />
  );
}

function renderHarness(eventId: string) {
  return render(
    <ScrollProvider>
      <JumpHarness eventId={eventId} />
    </ScrollProvider>,
  );
}

function anchorInto(container: HTMLElement, eventId: string) {
  const anchor = document.createElement("div");
  anchor.setAttribute("data-event-id", eventId);
  container.appendChild(anchor);
  return anchor;
}

describe("useJumpToEvent", () => {
  beforeEach(() => {
    mockLoadOlder.mockReset().mockResolvedValue(undefined);
    mockToastError.mockReset();
    Element.prototype.scrollIntoView = vi.fn();
  });

  afterEach(() => {
    document.querySelectorAll("[data-event-id]").forEach((n) => n.remove());
    vi.restoreAllMocks();
  });

  it("scrolls the found anchor into center and marks it highlighted", async () => {
    const { container } = renderHarness("evt-1");
    const anchor = anchorInto(container, "evt-1");

    await act(async () => {
      screen.getByTestId("jump").click();
    });

    await waitFor(() => {
      expect(Element.prototype.scrollIntoView).toHaveBeenCalledWith(
        expect.objectContaining({ block: "center" }),
      );
    });
    expect(anchor).toHaveAttribute("data-event-jump-highlight");
  });

  it("loads older pages until the anchor appears, then jumps", async () => {
    const { container } = renderHarness("evt-deep");
    // The anchor only materializes after the second page load.
    mockLoadOlder.mockImplementation(async () => {
      if (mockLoadOlder.mock.calls.length >= 2) {
        anchorInto(container, "evt-deep");
      }
    });

    await act(async () => {
      screen.getByTestId("jump").click();
    });

    await waitFor(() => {
      expect(mockLoadOlder).toHaveBeenCalledTimes(2);
    });
    expect(Element.prototype.scrollIntoView).toHaveBeenCalled();
  });

  it("stops after the 10-page guard and surfaces the not-loaded hint", async () => {
    renderHarness("evt-missing");

    await act(async () => {
      screen.getByTestId("jump").click();
    });

    await waitFor(() => {
      expect(mockLoadOlder).toHaveBeenCalledTimes(10);
    });
    // The "not in loaded history" hint is surfaced via the error toast.
    expect(mockToastError).toHaveBeenCalledWith(
      "CHAT_INTERFACE$REPLY_NOT_LOADED",
    );
  });
});
