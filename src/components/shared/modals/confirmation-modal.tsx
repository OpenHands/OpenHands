import { useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ModalBackdrop } from "./modal-backdrop";

interface ConfirmationModalProps {
  text: string;
  onConfirm: () => void;
  onCancel: () => void;
  confirmText?: string;
  /**
   * Disables both action buttons while an asynchronous confirm
   * mutation is in flight. Defaults to false to preserve existing
   * call sites that don't track mutation state.
   */
  isConfirming?: boolean;
}

export function ConfirmationModal({
  text,
  onConfirm,
  onCancel,
  confirmText,
  isConfirming = false,
}: ConfirmationModalProps) {
  const { t } = useTranslation("openhands");
  const modalRef = useRef<HTMLDivElement>(null);
  const previouslyFocusedElementRef = useRef<HTMLElement | null>(null);

  // Focus restoration: capture activeElement on mount, restore on unmount if it's still attached
  useEffect(() => {
    previouslyFocusedElementRef.current =
      document.activeElement as HTMLElement | null;

    return () => {
      const el = previouslyFocusedElementRef.current;
      if (el && typeof el.focus === "function" && document.contains(el)) {
        el.focus();
      }
    };
  }, []);

  // Initial focus: focus the first enabled focusable element inside the modal (e.g. cancel-button),
  // or fall back to the modal container.
  useEffect(() => {
    const modal = modalRef.current;
    if (!modal) return undefined;

    const timeoutId = setTimeout(() => {
      const focusable = modal.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );
      if (focusable.length > 0) {
        focusable[0]?.focus();
      } else {
        modal.focus();
      }
    }, 0);

    return () => clearTimeout(timeoutId);
  }, []);

  // Focus trap: keep focus within modal on Tab / Shift+Tab
  useEffect(() => {
    const modal = modalRef.current;
    if (!modal) return undefined;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;

      const focusable = Array.from(
        modal.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((el) => !el.hasAttribute("disabled") && el.tabIndex !== -1);

      if (focusable.length === 0) {
        e.preventDefault();
        modal.focus();
        return;
      }

      const firstElement = focusable[0];
      const lastElement = focusable[focusable.length - 1];

      if (e.shiftKey) {
        if (
          document.activeElement === firstElement ||
          !modal.contains(document.activeElement)
        ) {
          e.preventDefault();
          lastElement?.focus();
        }
      } else {
        if (
          document.activeElement === lastElement ||
          !modal.contains(document.activeElement)
        ) {
          e.preventDefault();
          firstElement?.focus();
        }
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Suppress the backdrop's click / Escape close handler while the
  // confirm mutation is in flight; otherwise the user could dismiss
  // the modal mid-request and never see the result (the buttons are
  // already disabled, but the backdrop wasn't).
  return (
    <ModalBackdrop
      onClose={isConfirming ? undefined : onCancel}
      closeOnEscape={!isConfirming}
    >
      <div
        ref={modalRef}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        data-testid="confirmation-modal"
        className="bg-base-secondary p-4 rounded-xl flex flex-col gap-4 border border-border outline-none"
      >
        <p>{text}</p>
        <div className="w-full flex justify-end gap-2">
          <BrandButton
            testId="cancel-button"
            type="button"
            onClick={onCancel}
            variant="secondary"
            isDisabled={isConfirming}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            testId="confirm-button"
            type="button"
            onClick={onConfirm}
            variant="primary"
            isDisabled={isConfirming}
          >
            {confirmText ?? t(I18nKey.BUTTON$CONFIRM)}
          </BrandButton>
        </div>
      </div>
    </ModalBackdrop>
  );
}
