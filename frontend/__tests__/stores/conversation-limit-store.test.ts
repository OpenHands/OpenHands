import { describe, expect, it, beforeEach } from "vitest";
import { useConversationLimitStore } from "#/stores/conversation-limit-store";

describe("useConversationLimitStore", () => {
  beforeEach(() => {
    // Reset store to initial state before each test
    useConversationLimitStore.setState({
      isOpen: false,
      limit: null,
    });
  });

  describe("initial state", () => {
    it("has isOpen set to false", () => {
      const { isOpen } = useConversationLimitStore.getState();
      expect(isOpen).toBe(false);
    });

    it("has limit set to null", () => {
      const { limit } = useConversationLimitStore.getState();
      expect(limit).toBeNull();
    });
  });

  describe("showLimitModal", () => {
    it("sets isOpen to true", () => {
      const { showLimitModal } = useConversationLimitStore.getState();
      showLimitModal(3);

      const { isOpen } = useConversationLimitStore.getState();
      expect(isOpen).toBe(true);
    });

    it("stores the limit value", () => {
      const { showLimitModal } = useConversationLimitStore.getState();
      showLimitModal(5);

      const { limit } = useConversationLimitStore.getState();
      expect(limit).toBe(5);
    });

    it("stores different limit values correctly", () => {
      const { showLimitModal } = useConversationLimitStore.getState();

      showLimitModal(1);
      expect(useConversationLimitStore.getState().limit).toBe(1);

      showLimitModal(10);
      expect(useConversationLimitStore.getState().limit).toBe(10);

      showLimitModal(100);
      expect(useConversationLimitStore.getState().limit).toBe(100);
    });

    it("updates state when called multiple times", () => {
      const { showLimitModal } = useConversationLimitStore.getState();

      showLimitModal(3);
      expect(useConversationLimitStore.getState().isOpen).toBe(true);
      expect(useConversationLimitStore.getState().limit).toBe(3);

      showLimitModal(5);
      expect(useConversationLimitStore.getState().isOpen).toBe(true);
      expect(useConversationLimitStore.getState().limit).toBe(5);
    });
  });

  describe("closeLimitModal", () => {
    it("sets isOpen to false", () => {
      const { showLimitModal, closeLimitModal } =
        useConversationLimitStore.getState();

      // First open the modal
      showLimitModal(3);
      expect(useConversationLimitStore.getState().isOpen).toBe(true);

      // Then close it
      closeLimitModal();
      expect(useConversationLimitStore.getState().isOpen).toBe(false);
    });

    it("resets limit to null", () => {
      const { showLimitModal, closeLimitModal } =
        useConversationLimitStore.getState();

      // First open the modal with a limit
      showLimitModal(5);
      expect(useConversationLimitStore.getState().limit).toBe(5);

      // Then close it
      closeLimitModal();
      expect(useConversationLimitStore.getState().limit).toBeNull();
    });

    it("resets to initial state", () => {
      const { showLimitModal, closeLimitModal } =
        useConversationLimitStore.getState();

      // Open modal
      showLimitModal(10);

      // Close modal
      closeLimitModal();

      const state = useConversationLimitStore.getState();
      expect(state.isOpen).toBe(false);
      expect(state.limit).toBeNull();
    });

    it("can be called when modal is already closed", () => {
      const { closeLimitModal } = useConversationLimitStore.getState();

      // Should not throw when called on already closed modal
      expect(() => closeLimitModal()).not.toThrow();

      const state = useConversationLimitStore.getState();
      expect(state.isOpen).toBe(false);
      expect(state.limit).toBeNull();
    });
  });

  describe("workflow", () => {
    it("supports open -> close -> open cycle", () => {
      const { showLimitModal, closeLimitModal } =
        useConversationLimitStore.getState();

      // Open
      showLimitModal(3);
      expect(useConversationLimitStore.getState().isOpen).toBe(true);
      expect(useConversationLimitStore.getState().limit).toBe(3);

      // Close
      closeLimitModal();
      expect(useConversationLimitStore.getState().isOpen).toBe(false);
      expect(useConversationLimitStore.getState().limit).toBeNull();

      // Open again with different limit
      showLimitModal(5);
      expect(useConversationLimitStore.getState().isOpen).toBe(true);
      expect(useConversationLimitStore.getState().limit).toBe(5);
    });
  });
});
