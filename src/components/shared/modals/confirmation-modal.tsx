import React, { useEffect, useRef } from "react";
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
  const openerRef = useRef<HTMLElement | null>(null);

  // Focus restoration on unmount
  useEffect(() => {
    openerRef.current = (document.activeElement as HTMLElement) ?? null;
    return () => {
      if (openerRef.current && document.body.contains(openerRef.current)) {
        openerRef.current.focus();
      }
    };
  }, []);

  // Initial focus: target the first focusable element (Cancel button)
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      const modal = modalRef.current;
      if (!modal) return;
      const focusable = Array.from(
        modal.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      );
      if (focusable.length > 0) {
        focusable[0]?.focus();
      } else {
        modal.focus();
      }
    }, 0);

    return () => clearTimeout(timeoutId);
  }, []);

  // Focus trap
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;

      const modal = modalRef.current;
      if (!modal) return;

      const focusable = Array.from(
        modal.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ),
      );

      if (focusable.length === 0) {
        e.preventDefault();
        modal.focus();
        return;
      }

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement;

      if (!modal.contains(active)) {
        e.preventDefault();
        if (e.shiftKey) {
          last?.focus();
        } else {
          first?.focus();
        }
        return;
      }

      if (e.shiftKey) {
        if (active === first || active === modal) {
          e.preventDefault();
          last?.focus();
        }
      } else if (active === modal) {
        e.preventDefault();
        first?.focus();
      } else if (active === last) {
        e.preventDefault();
        first?.focus();
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
