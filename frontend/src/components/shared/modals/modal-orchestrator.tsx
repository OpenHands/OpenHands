import React, { useEffect } from "react";
import { createPortal } from "react-dom";
import { useModalStore } from "#/stores/modal-store";
import type { ModalInstance } from "#/stores/modal-store";
import { MODAL_BASE_Z_INDEX, MODAL_Z_INDEX_GAP } from "#/utils/constants";
import { MODAL_REGISTRY } from "#/components/shared/modals/modal-registry";

function renderModal(
  modal: ModalInstance,
  onClose: () => void,
): React.ReactNode {
  const renderer = MODAL_REGISTRY[modal.type];
  if (!renderer) return null;
  return renderer(modal.props as never, onClose);
}

/**
 * ModalRoot - Central modal renderer
 *
 * Renders all modals from the modal store stack using React Portal.
 * Handles:
 * - ESC key to close topmost modal
 * - Backdrop click to close
 * - Proper z-index stacking for nested modals
 */
export function ModalRoot() {
  const modalStack = useModalStore((state) => state.modalStack);
  const closeModal = useModalStore((state) => state.closeModal);
  const topModal = useModalStore((state) => state.topModal);

  // Handle ESC key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      const top = topModal();
      if (e.key === "Escape" && top && top.props.closeOnEscape !== false) {
        closeModal();
      }
    };

    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [topModal, closeModal]);

  // Don't render if no modals
  if (modalStack.length === 0) return null;

  const portalRoot = document.getElementById("modal-portal-exit");
  if (!portalRoot) {
    // eslint-disable-next-line no-console
    console.warn(
      'ModalRoot: #modal-portal-exit not found. Add <div id="modal-portal-exit" /> to your HTML.',
    );
    return null;
  }

  return createPortal(
    <>
      {modalStack.map((modal, index) => {
        const isTopModal = index === modalStack.length - 1;
        const zIndex = MODAL_BASE_Z_INDEX + index * MODAL_Z_INDEX_GAP;
        const allowBackdropClose = modal.props.closeOnBackdrop !== false;

        const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
          if (
            e.target === e.currentTarget &&
            isTopModal &&
            allowBackdropClose
          ) {
            closeModal();
          }
        };

        return (
          <div
            key={`${modal.type}-${index}`}
            className="fixed inset-0 flex items-center justify-center"
            style={{ zIndex }}
            data-testid={`modal-backdrop-${modal.type}`}
          >
            <div
              className="fixed inset-0 bg-black opacity-60"
              onClick={handleBackdropClick}
              aria-hidden="true"
            />
            <div className="relative">{renderModal(modal, closeModal)}</div>
          </div>
        );
      })}
    </>,
    portalRoot,
  );
}
