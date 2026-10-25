import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import XMarkIcon from "#/icons/x-mark.svg?react";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import {
  MODAL_MAX_WIDTH_VIEWPORT,
  modalWidthClassName,
} from "#/components/shared/modals/modal-body";
import { modalTitleLgMediumClassName } from "#/utils/modal-classes";
import { cn } from "#/utils/utils";

interface DeleteConfirmationModalProps {
  automationName: string;
  isOpen: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  /** Overrides the default automation copy, e.g. for a saved draft. */
  title?: string;
  message?: string;
  confirmLabel?: string;
  confirmTestId?: string;
  /** Disables Delete while the deletion is in flight. */
  isConfirming?: boolean;
}

export function DeleteConfirmationModal({
  automationName,
  isOpen,
  onConfirm,
  onCancel,
  title: titleOverride,
  message,
  confirmLabel,
  confirmTestId,
  isConfirming = false,
}: DeleteConfirmationModalProps) {
  const { t } = useTranslation("openhands");
  const cancelButtonRef = React.useRef<HTMLButtonElement>(null);

  // Move keyboard focus into the dialog when it opens, onto the
  // non-destructive choice, so Enter or Space cannot delete by accident.
  // On close, hand focus back to the control that opened the dialog.
  React.useEffect(() => {
    if (!isOpen) return undefined;
    const opener = document.activeElement;
    cancelButtonRef.current?.focus();
    return () => {
      if (opener instanceof HTMLElement && opener.isConnected) opener.focus();
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const title = titleOverride ?? t(I18nKey.AUTOMATIONS$DELETE_CONFIRM_TITLE);

  // ModalBackdrop exposes the named `role="dialog"` with `aria-modal` and
  // closes on Escape or a backdrop click.
  return (
    <ModalBackdrop onClose={onCancel} aria-label={title}>
      <div
        className={cn(
          "relative rounded-xl border border-border bg-surface p-6",
          modalWidthClassName("sm"),
          MODAL_MAX_WIDTH_VIEWPORT,
        )}
      >
        <button
          type="button"
          onClick={onCancel}
          className="absolute right-4 top-4 text-muted hover:text-foreground"
          aria-label={t(I18nKey.BUTTON$CLOSE)}
        >
          <XMarkIcon className="size-5" />
        </button>

        <h2 className={modalTitleLgMediumClassName}>{title}</h2>
        <p className="mt-2 text-sm text-muted">
          {message ??
            t(I18nKey.AUTOMATIONS$DELETE_CONFIRM_MESSAGE, {
              name: automationName,
            })}
        </p>

        <div className="mt-6 flex justify-end gap-3">
          <button
            ref={cancelButtonRef}
            type="button"
            onClick={onCancel}
            className="rounded-lg border border-border px-4 py-2 text-sm text-contrast hover:bg-surface-raised"
          >
            {t(I18nKey.AUTOMATIONS$CANCEL)}
          </button>
          <button
            type="button"
            data-testid={confirmTestId}
            onClick={onConfirm}
            disabled={isConfirming}
            className="rounded-lg bg-danger px-4 py-2 text-sm text-white hover:bg-danger/80 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {confirmLabel ?? t(I18nKey.AUTOMATIONS$DELETE)}
          </button>
        </div>
      </div>
    </ModalBackdrop>
  );
}
