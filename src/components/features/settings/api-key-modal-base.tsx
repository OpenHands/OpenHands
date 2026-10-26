import React, { ReactNode, useEffect, useRef } from "react";
import {
  BaseModalDescription,
  BaseModalTitle,
} from "#/components/shared/modals/confirmation-modals/base-modal";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import {
  MODAL_MAX_WIDTH_VIEWPORT,
  type ModalWidth,
  modalWidthClassName,
} from "#/components/shared/modals/modal-body";
import { cn } from "#/utils/utils";

interface ApiKeyModalBaseProps {
  isOpen: boolean;
  title: string;
  /** Optional text shown under the title. */
  description?: ReactNode;
  width?: ModalWidth;
  children: ReactNode;
  footer: ReactNode;
  /** Called when the modal should close (e.g., Escape key or backdrop click) */
  onClose?: () => void;
  /** Ref to an element that should receive initial focus when modal opens */
  initialFocusRef?: React.RefObject<HTMLElement | null>;
  /** Show an X close button in the top-right corner (needs `onClose`). */
  showCloseButton?: boolean;
  /** Disables that close button, e.g. while a save is in flight. */
  isCloseDisabled?: boolean;
}

export function ApiKeyModalBase({
  isOpen,
  title,
  description,
  width = "md",
  children,
  footer,
  onClose,
  initialFocusRef,
  showCloseButton = false,
  isCloseDisabled = false,
}: ApiKeyModalBaseProps) {
  const modalRef = useRef<HTMLDivElement>(null);

  // Focus management: set initial focus and trap focus within modal
  useEffect(() => {
    if (!isOpen) return undefined;

    // Set initial focus
    const focusTarget = initialFocusRef?.current ?? modalRef.current;
    if (focusTarget) {
      // Small delay to ensure modal is fully rendered
      const timeoutId = setTimeout(() => {
        focusTarget.focus();
      }, 0);
      return () => clearTimeout(timeoutId);
    }
    return undefined;
  }, [isOpen, initialFocusRef]);

  // Focus trap: keep focus within modal
  useEffect(() => {
    if (!isOpen || !modalRef.current) return undefined;

    const modal = modalRef.current;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;

      const focusableElements = modal.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );

      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];

      if (e.shiftKey && document.activeElement === firstElement) {
        e.preventDefault();
        lastElement?.focus();
      } else if (!e.shiftKey && document.activeElement === lastElement) {
        e.preventDefault();
        firstElement?.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <ModalBackdrop onClose={onClose}>
      <div
        ref={modalRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
        aria-describedby={description ? "modal-description" : undefined}
        className={cn(
          "relative bg-base-secondary p-6 rounded-xl flex flex-col gap-4 border border-border",
          modalWidthClassName(width),
          MODAL_MAX_WIDTH_VIEWPORT,
        )}
      >
        {showCloseButton && onClose ? (
          <ModalCloseButton
            onClose={onClose}
            disabled={isCloseDisabled}
            testId="modal-close-button"
          />
        ) : null}
        <div className="flex flex-col gap-1">
          <BaseModalTitle
            id="modal-title"
            title={title}
            className={cn(showCloseButton && "pr-6")}
          />
          {description ? (
            <span id="modal-description">
              <BaseModalDescription description={description} />
            </span>
          ) : null}
        </div>
        {children}
        <div className="w-full flex justify-end gap-2 mt-2">{footer}</div>
      </div>
    </ModalBackdrop>
  );
}
