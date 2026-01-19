import React, { useEffect } from "react";
import { createPortal } from "react-dom";
import { useModalStore } from "#/stores/modal-store";
import type { ModalInstance, ModalConfigMap } from "#/stores/modal-store";
import { ConfirmDeleteModal } from "#/components/features/conversation-panel/confirm-delete-modal";
import { ConfirmStopModal } from "#/components/features/conversation-panel/confirm-stop-modal";
import { FeedbackModal } from "#/components/features/feedback/feedback-modal";

/**
 * Renders the appropriate modal component based on modal type.
 * Add new modal cases here when migrating modals to the centralized system.
 */
function renderModal(
  modal: ModalInstance,
  onClose: () => void,
): React.ReactNode {
  switch (modal.type) {
    case "confirm-delete": {
      const props = modal.props as ModalConfigMap["confirm-delete"];
      return (
        <ConfirmDeleteModal
          conversationTitle={props.conversationTitle}
          onConfirm={() => {
            props.onConfirm();
            onClose();
          }}
          onCancel={onClose}
        />
      );
    }

    case "confirm-stop": {
      const props = modal.props as ModalConfigMap["confirm-stop"];
      return (
        <ConfirmStopModal
          onConfirm={() => {
            props.onConfirm();
            onClose();
          }}
          onCancel={onClose}
        />
      );
    }

    case "feedback": {
      const props = modal.props as ModalConfigMap["feedback"];
      return <FeedbackModal polarity={props.polarity} onClose={onClose} />;
    }

    default:
      return null;
  }
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
        // Higher base z-index (1000) to stay above toasts, command palette, etc.
        const zIndex = 1000 + index * 10;
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
