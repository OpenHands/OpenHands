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
const mockHasMore = vi.fn(() => true);
const lastConversationId: { id?: string | null } = {};
vi.mock("#/hooks/use-load-older-events", () => ({
  // Capture the conversation id the hook wires in: load-until-found is a
  // no-op without it (loadOlder early-returns on a falsy id).
  useLoadOlderEvents: (conversationId?: string | null) => {
    lastConversationId.id = conversationId;
    return {
      isLoading: false,
      hasMore: mockHasMore(),
      loadOlder: mockLoadOlder,
    };
  },
}));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({
    conversationId: "conv-1",
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
    mockHasMore.mockClear().mockReturnValue(true);
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

  it("stops early once history is exhausted and surfaces the not-loaded hint", async () => {
    mockHasMore.mockReturnValue(false);
    renderHarness("evt-missing");

    await act(async () => {
      screen.getByTestId("jump").click();
    });

    // Exhausted history: no point burning page loads — one find pass, no
    // loadOlder, straight to the hint.
    expect(mockLoadOlder).not.toHaveBeenCalled();
    expect(mockToastError).toHaveBeenCalledWith(
      "CHAT_INTERFACE$REPLY_NOT_LOADED",
    );
  });

  it("pages with a real conversation id and stops at the 10-page guard", async () => {
    renderHarness("evt-missing");

    await act(async () => {
      screen.getByTestId("jump").click();
    });

    // Regression: an undefined conversation id makes every loadOlder a
    // guaranteed no-op, so the guard exhausts without fetching anything.
    expect(lastConversationId.id).toBe("conv-1");
    expect(mockLoadOlder).toHaveBeenCalledTimes(10);
    expect(mockToastError).toHaveBeenCalledWith(
      "CHAT_INTERFACE$REPLY_NOT_LOADED",
    );
  });
});
