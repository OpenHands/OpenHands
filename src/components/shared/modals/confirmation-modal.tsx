import React from "react";
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
  const modalRef = React.useRef<HTMLDivElement>(null);
  const cancelButtonRef = React.useRef<HTMLButtonElement>(null);
  const confirmButtonRef = React.useRef<HTMLButtonElement>(null);

  React.useEffect(() => {
    const previousFocus = document.activeElement;
    cancelButtonRef.current?.focus();
    return () => {
      if (previousFocus instanceof HTMLElement && previousFocus.isConnected) {
        previousFocus.focus();
      }
    };
  }, []);

  React.useEffect(() => {
    if (isConfirming) {
      modalRef.current?.focus();
    } else if (document.activeElement === modalRef.current) {
      cancelButtonRef.current?.focus();
    }
  }, [isConfirming]);

  const keepFocusInside = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "Tab") return;

    const buttons = [cancelButtonRef.current, confirmButtonRef.current].filter(
      (button): button is HTMLButtonElement => !!button && !button.disabled,
    );
    if (buttons.length === 0) {
      event.preventDefault();
      modalRef.current?.focus();
      return;
    }

    const first = buttons[0];
    const last = buttons[buttons.length - 1];
    const focusIsInside = modalRef.current?.contains(document.activeElement);
    if (
      event.shiftKey &&
      (!focusIsInside || document.activeElement === first)
    ) {
      event.preventDefault();
      last.focus();
    } else if (
      !event.shiftKey &&
      (!focusIsInside || document.activeElement === last)
    ) {
      event.preventDefault();
      first.focus();
    }
  };

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
        data-testid="confirmation-modal"
        tabIndex={-1}
        onKeyDown={keepFocusInside}
        className="bg-base-secondary p-4 rounded-xl flex flex-col gap-4 border border-border"
      >
        <p>{text}</p>
        <div className="w-full flex justify-end gap-2">
          <BrandButton
            ref={cancelButtonRef}
            testId="cancel-button"
            type="button"
            onClick={onCancel}
            variant="secondary"
            isDisabled={isConfirming}
          >
            {t(I18nKey.BUTTON$CANCEL)}
          </BrandButton>
          <BrandButton
            ref={confirmButtonRef}
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
