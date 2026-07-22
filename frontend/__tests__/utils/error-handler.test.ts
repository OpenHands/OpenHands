import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  trackError,
  showErrorToast,
  showChatError,
  classifyBudgetOrCreditError,
} from "#/utils/error-handler";
import * as Actions from "#/services/actions";
import * as CustomToast from "#/utils/custom-toast-handlers";

vi.mock("#/services/actions", () => ({
  handleStatusMessage: vi.fn(),
}));

describe("Error Handler", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe("trackError", () => {
    it("should be a no-op (PostHog capture removed)", () => {
      // trackError no longer does anything — error tracking is server-side
      expect(() =>
        trackError({ message: "Test error", source: "test" }),
      ).not.toThrow();
    });

    it("should accept ErrorDetails without throwing", () => {
      expect(() =>
        trackError({
          message: "Test error",
          source: "test",
          metadata: { extra: "info" },
        }),
      ).not.toThrow();
    });
  });

  describe("showErrorToast", () => {
    const errorToastSpy = vi.spyOn(CustomToast, "displayErrorToast");

    it("should show toast with the error message", () => {
      showErrorToast({ message: "Toast error", source: "toast-test" });

      expect(errorToastSpy).toHaveBeenCalledWith("Toast error");
    });

    it("should show toast even without source or metadata", () => {
      showErrorToast({ message: "Simple error" });

      expect(errorToastSpy).toHaveBeenCalledWith("Simple error");
    });
  });

  describe("showChatError", () => {
    it("should show chat error message via handleStatusMessage", () => {
      showChatError({
        message: "Chat error",
        source: "chat-test",
        msgId: "123",
      });

      expect(Actions.handleStatusMessage).toHaveBeenCalledWith({
        type: "error",
        message: "Chat error",
        id: "123",
        status_update: true,
      });
    });

    it("should show chat error without msgId", () => {
      showChatError({
        message: "Chat error no id",
        source: "chat-test",
      });

      expect(Actions.handleStatusMessage).toHaveBeenCalledWith({
        type: "error",
        message: "Chat error no id",
        id: undefined,
        status_update: true,
      });
    });
  });

  describe("classifyBudgetOrCreditError", () => {
    it("classifies OpenHands budget and credit limit errors", () => {
      expect(
        classifyBudgetOrCreditError(
          "Budget has been exceeded! Current cost: 18.51, Max budget: 18.24",
        ),
      ).toBe("budget");
      expect(
        classifyBudgetOrCreditError("OpenHands Credits are exhausted"),
      ).toBe("credit");
      expect(classifyBudgetOrCreditError("Credit limit reached")).toBe("credit");
    });

    it("ignores provider-side credit messages", () => {
      expect(
        classifyBudgetOrCreditError(
          "OpenrouterException - This model requires provider credits. Check your OpenRouter account.",
        ),
      ).toBeNull();
      expect(
        classifyBudgetOrCreditError(
          "Provider returned insufficient credits for minimax/minimax-m2.5:free",
        ),
      ).toBeNull();
    });
  });
});
