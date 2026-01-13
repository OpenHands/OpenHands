import { render, fireEvent } from "@testing-library/react";
import { vi, describe, it, expect, beforeEach, afterEach } from "vitest";
import { ModalRoot } from "#/components/shared/modals/modal-orchestrator";
import { useModalStore } from "#/stores/modal-store";

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

beforeEach(() => {
  const portalRoot = document.createElement("div");
  portalRoot.id = "modal-portal-exit";
  document.body.appendChild(portalRoot);
});

afterEach(() => {
  const portalRoot = document.getElementById("modal-portal-exit");
  if (portalRoot) {
    document.body.removeChild(portalRoot);
  }
  useModalStore.getState().closeAllModals();
});

describe("ModalRoot", () => {
  describe("ESC key handling", () => {
    it("should close the topmost modal when ESC is pressed", () => {
      useModalStore.getState().openModal("confirmation", {
        text: "Test confirmation",
        onConfirm: vi.fn(),
      });

      render(<ModalRoot />);

      expect(useModalStore.getState().modalStack.length).toBe(1);

      fireEvent.keyDown(window, { key: "Escape" });

      expect(useModalStore.getState().modalStack.length).toBe(0);
    });

    it("should not close the modal when closeOnEscape is false", () => {
      useModalStore.getState().openModal("confirmation", {
        text: "Test confirmation",
        onConfirm: vi.fn(),
        closeOnEscape: false,
      });

      render(<ModalRoot />);

      expect(useModalStore.getState().modalStack.length).toBe(1);

      fireEvent.keyDown(window, { key: "Escape" });

      expect(useModalStore.getState().modalStack.length).toBe(1);
    });

    it("should only close the topmost modal in a stack", () => {
      const store = useModalStore.getState();

      store.openModal("confirmation", {
        text: "First modal",
        onConfirm: vi.fn(),
      });
      store.openModal(
        "confirmation",
        {
          text: "Second modal",
          onConfirm: vi.fn(),
        },
        { allowDuplicate: true },
      );

      render(<ModalRoot />);

      expect(useModalStore.getState().modalStack.length).toBe(2);
      fireEvent.keyDown(window, { key: "Escape" });
      expect(useModalStore.getState().modalStack.length).toBe(1);
    });
  });

  describe("Backdrop click handling", () => {
    it("should close modal when backdrop is clicked", () => {
      useModalStore.getState().openModal("confirmation", {
        text: "Test confirmation",
        onConfirm: vi.fn(),
      });

      render(<ModalRoot />);

      const backdrop = document.querySelector(".bg-black.opacity-60");
      expect(backdrop).toBeTruthy();

      if (backdrop) {
        fireEvent.click(backdrop);
      }

      expect(useModalStore.getState().modalStack.length).toBe(0);
    });

    it("should not close modal when closeOnBackdrop is false", () => {
      useModalStore.getState().openModal("confirmation", {
        text: "Test confirmation",
        onConfirm: vi.fn(),
        closeOnBackdrop: false,
      });

      render(<ModalRoot />);

      const backdrop = document.querySelector(".bg-black.opacity-60");
      if (backdrop) {
        fireEvent.click(backdrop);
      }

      expect(useModalStore.getState().modalStack.length).toBe(1);
    });
  });

  describe("Modal rendering", () => {
    it("should render nothing when no modals are open", () => {
      const { container } = render(<ModalRoot />);
      expect(container.firstChild).toBeNull();
    });

    it("should render modal when opened", () => {
      useModalStore.getState().openModal("confirmation", {
        text: "Test confirmation text",
        onConfirm: vi.fn(),
      });

      render(<ModalRoot />);

      const portalRoot = document.getElementById("modal-portal-exit");
      expect(portalRoot?.children.length).toBeGreaterThan(0);
    });
  });
});
