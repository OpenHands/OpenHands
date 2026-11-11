import { describe, it, expect, vi, beforeEach } from "vitest";
import type { ReactElement } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import toast from "react-hot-toast";
import {
  displaySuccessToast,
  displayErrorToast,
  displaySuccessToastWithLink,
} from "#/utils/custom-toast-handlers";

const { toastMock } = vi.hoisted(() => ({
  toastMock: Object.assign(vi.fn(), {
    success: vi.fn(),
    error: vi.fn(),
    loading: vi.fn(),
    dismiss: vi.fn(),
  }),
}));

// Mock react-hot-toast
vi.mock("react-hot-toast", () => ({
  default: toastMock,
}));

describe("custom-toast-handlers", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("displaySuccessToast", () => {
    it("should call toast.success with calculated duration for short message", () => {
      const shortMessage = "Settings saved";
      displaySuccessToast(shortMessage);

      expect(toast.success).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({
          duration: 5000, // Should use minimum duration of 5000ms
          position: "top-right",
          style: expect.objectContaining({
            borderRadius: "var(--oh-radius)",
            maxWidth: "400px",
            wordBreak: "break-word",
          }),
        }),
      );
    });

    it("should call toast.success with longer duration for long message", () => {
      const longMessage =
        "Settings saved. For old conversations, you will need to stop and restart the conversation to see the changes.";
      displaySuccessToast(longMessage);

      expect(toast.success).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({
          duration: expect.any(Number),
          position: "top-right",
          style: expect.objectContaining({
            borderRadius: "var(--oh-radius)",
            maxWidth: "400px",
            wordBreak: "break-word",
          }),
        }),
      );

      // Get the actual duration that was passed
      const callArgs = (
        toast.success as unknown as { mock: { calls: unknown[][] } }
      ).mock.calls[0][1] as { duration: number };
      const actualDuration = callArgs.duration;

      // For a long message, duration should be more than the minimum 5000ms
      expect(actualDuration).toBeGreaterThan(5000);
      // But should not exceed the maximum 10000ms
      expect(actualDuration).toBeLessThanOrEqual(10000);
    });
  });

  describe("displayErrorToast", () => {
    it("should call toast with calculated duration for short message", () => {
      const shortMessage = "Error occurred";
      displayErrorToast(shortMessage);

      expect(toastMock).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({
          duration: 4000, // Should use minimum duration of 4000ms for errors
          position: "top-right",
          icon: null,
          style: expect.objectContaining({
            borderRadius: "var(--oh-radius)",
            maxWidth: "400px",
            wordBreak: "break-word",
          }),
        }),
      );
    });

    it("should call toast with longer duration for long error message", () => {
      const longMessage =
        "A very long error message that should take more time to read and understand what went wrong with the operation.";
      displayErrorToast(longMessage);

      expect(toastMock).toHaveBeenCalledWith(
        expect.anything(),
        expect.objectContaining({
          duration: expect.any(Number),
          position: "top-right",
          icon: null,
          style: expect.objectContaining({
            borderRadius: "var(--oh-radius)",
            maxWidth: "400px",
            wordBreak: "break-word",
          }),
        }),
      );

      // Get the actual duration that was passed
      const callArgs = toastMock.mock.calls[0][1] as { duration: number };
      const actualDuration = callArgs.duration;

      // For a long message, duration should be more than the minimum 4000ms
      expect(actualDuration).toBeGreaterThan(4000);
      // But should not exceed the maximum 10000ms
      expect(actualDuration).toBeLessThanOrEqual(10000);
    });
  });

  describe("displaySuccessToastWithLink", () => {
    const toastNode = () =>
      (toastMock.success as unknown as { mock: { calls: unknown[][] } }).mock
        .calls[0][0] as ReactElement;

    it("renders a plain anchor the browser follows when no navigation owner is given", () => {
      displaySuccessToastWithLink("Branched", "Open", "/conversations/fork-1");

      render(toastNode());

      const link = screen.getByRole("link", { name: "Open" });
      expect(link).toHaveAttribute("href", "/conversations/fork-1");
      expect(screen.queryByRole("button", { name: "Open" })).toBeNull();
    });

    it("routes the link through the navigation owner when one is given", async () => {
      // An embedded Canvas must not follow a plain href: that navigates the
      // host document out of the app. The owner's navigate keeps it embedded.
      const onOpen = vi.fn();
      displaySuccessToastWithLink(
        "Branched",
        "Open",
        "/conversations/fork-1",
        onOpen,
      );

      render(toastNode());

      expect(screen.queryByRole("link", { name: "Open" })).toBeNull();
      await userEvent.click(screen.getByRole("button", { name: "Open" }));
      expect(onOpen).toHaveBeenCalledTimes(1);
    });
  });
});
