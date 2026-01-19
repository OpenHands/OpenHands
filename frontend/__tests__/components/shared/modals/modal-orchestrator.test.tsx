import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import { ModalRoot } from "#/components/shared/modals/modal-orchestrator";
import { useModalStore } from "#/stores/modal-store";

describe("ModalRoot", () => {
  // Create portal target before each test
  beforeEach(() => {
    const portalRoot = document.createElement("div");
    portalRoot.setAttribute("id", "modal-portal-exit");
    document.body.appendChild(portalRoot);

    // Reset modal store
    useModalStore.setState({ modalStack: [] });
  });

  afterEach(() => {
    const portalRoot = document.getElementById("modal-portal-exit");
    if (portalRoot) {
      document.body.removeChild(portalRoot);
    }
  });

  it("should render nothing when no modals are open", () => {
    render(<ModalRoot />);

    expect(screen.queryByTestId(/modal-backdrop/)).not.toBeInTheDocument();
  });

  it("should render modal when openModal is called", () => {
    render(<ModalRoot />);

    // Open a modal
    act(() => {
      useModalStore.getState().openModal("confirm-delete", {
        conversationTitle: "Test Conversation",
        onConfirm: () => {},
      });
    });

    expect(
      screen.getByTestId("modal-backdrop-confirm-delete"),
    ).toBeInTheDocument();
  });

  it("should update modalStack when closeModal is called", () => {
    render(<ModalRoot />);

    // Open a modal
    act(() => {
      useModalStore.getState().openModal("confirm-delete", {
        conversationTitle: "Test",
        onConfirm: () => {},
      });
    });

    expect(useModalStore.getState().modalStack).toHaveLength(1);

    // Close modal via store
    act(() => {
      useModalStore.getState().closeModal();
    });

    expect(useModalStore.getState().modalStack).toHaveLength(0);
  });

  it("should prevent duplicate modals of same type by default", () => {
    render(<ModalRoot />);

    // Open modal
    act(() => {
      useModalStore.getState().openModal("confirm-delete", {
        conversationTitle: "First",
        onConfirm: () => {},
      });
    });

    // Try to open same modal type again
    act(() => {
      useModalStore.getState().openModal("confirm-delete", {
        conversationTitle: "Second",
        onConfirm: () => {},
      });
    });

    // Should only have one modal in stack
    expect(useModalStore.getState().modalStack).toHaveLength(1);
  });

  it("should allow stacking different modal types", () => {
    // Open two different modals directly via store
    act(() => {
      useModalStore.getState().openModal("confirm-delete", {
        conversationTitle: "Delete",
        onConfirm: () => {},
      });
    });

    act(() => {
      useModalStore.getState().openModal("confirm-stop", {
        onConfirm: () => {},
      });
    });

    // Both should be in stack
    expect(useModalStore.getState().modalStack).toHaveLength(2);
    expect(useModalStore.getState().modalStack[0].type).toBe("confirm-delete");
    expect(useModalStore.getState().modalStack[1].type).toBe("confirm-stop");
  });

  it("should closeModalByType correctly", () => {
    // Open two different modals
    act(() => {
      useModalStore.getState().openModal("confirm-delete", {
        conversationTitle: "Delete",
        onConfirm: () => {},
      });
    });

    act(() => {
      useModalStore.getState().openModal("confirm-stop", {
        onConfirm: () => {},
      });
    });

    expect(useModalStore.getState().modalStack).toHaveLength(2);

    // Close specific modal by type
    act(() => {
      useModalStore.getState().closeModalByType("confirm-delete");
    });

    expect(useModalStore.getState().modalStack).toHaveLength(1);
    expect(useModalStore.getState().modalStack[0].type).toBe("confirm-stop");
  });

  it("should closeAllModals correctly", () => {
    // Open multiple modals
    act(() => {
      useModalStore.getState().openModal("confirm-delete", {
        conversationTitle: "Delete",
        onConfirm: () => {},
      });
    });

    act(() => {
      useModalStore.getState().openModal("confirm-stop", {
        onConfirm: () => {},
      });
    });

    expect(useModalStore.getState().modalStack).toHaveLength(2);

    // Close all modals
    act(() => {
      useModalStore.getState().closeAllModals();
    });

    expect(useModalStore.getState().modalStack).toHaveLength(0);
  });

  it("should replaceModal correctly", () => {
    // Open a modal
    act(() => {
      useModalStore.getState().openModal("confirm-delete", {
        conversationTitle: "Original",
        onConfirm: () => {},
      });
    });

    // Replace with new modal
    act(() => {
      useModalStore.getState().replaceModal("confirm-stop", {
        onConfirm: () => {},
      });
    });

    expect(useModalStore.getState().modalStack).toHaveLength(1);
    expect(useModalStore.getState().modalStack[0].type).toBe("confirm-stop");
  });

  it("should have correct store selectors", () => {
    expect(useModalStore.getState().isOpen()).toBe(false);
    expect(useModalStore.getState().topModal()).toBeUndefined();

    act(() => {
      useModalStore.getState().openModal("confirm-delete", {
        conversationTitle: "Test",
        onConfirm: () => {},
      });
    });

    expect(useModalStore.getState().isOpen()).toBe(true);
    expect(useModalStore.getState().topModal()?.type).toBe("confirm-delete");
  });
});
