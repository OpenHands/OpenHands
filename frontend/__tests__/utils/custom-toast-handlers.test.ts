import { describe, it, expect, vi, beforeEach } from "vitest";
import toast from "react-hot-toast";
import {
  displaySuccessToast,
  displayErrorToast,
} from "#/utils/custom-toast-handlers";

// Mock react-hot-toast
vi.mock("react-hot-toast", () => ({
  default: {
    success: vi.fn(),
    error: vi.fn(),
  },
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
        shortMessage,
        expect.objectContaining({
          duration: 5000, // Should use minimum duration of 5000ms
          position: "top-right",
          style: expect.any(Object),
        }),
      );
    });

    it("should call toast.success with longer duration for long message", () => {
      const longMessage =
        "Settings saved. For old conversations, you will need to stop and restart the conversation to see the changes.";
      displaySuccessToast(longMessage);

      expect(toast.success).toHaveBeenCalledWith(
        longMessage,
        expect.objectContaining({
          duration: expect.any(Number),
          position: "top-right",
          style: expect.any(Object),
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
    it("should call toast.error with calculated duration for short message", () => {
      const shortMessage = "Error occurred";
      displayErrorToast(shortMessage);

      expect(toast.error).toHaveBeenCalledWith(
        shortMessage,
        expect.objectContaining({
          duration: 4000, // Should use minimum duration of 4000ms for errors
          position: "top-right",
          style: expect.any(Object),
        }),
      );
    });

    it("should call toast.error with longer duration for long error message", () => {
      const longMessage =
        "A very long error message that should take more time to read and understand what went wrong with the operation.";
      displayErrorToast(longMessage);

      expect(toast.error).toHaveBeenCalledWith(
        longMessage,
        expect.objectContaining({
          duration: expect.any(Number),
          position: "top-right",
          style: expect.any(Object),
        }),
      );

      // Get the actual duration that was passed
      const callArgs = (
        toast.error as unknown as { mock: { calls: unknown[][] } }
      ).mock.calls[0][1] as { duration: number };
      const actualDuration = callArgs.duration;

      // For a long message, duration should be more than the minimum 4000ms
      expect(actualDuration).toBeGreaterThan(4000);
      // But should not exceed the maximum 10000ms
      expect(actualDuration).toBeLessThanOrEqual(10000);
    });
  });
});
