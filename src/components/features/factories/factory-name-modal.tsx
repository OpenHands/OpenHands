import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import XMarkIcon from "#/icons/x-mark.svg?react";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import {
  MODAL_MAX_WIDTH_VIEWPORT,
  modalWidthClassName,
} from "#/components/shared/modals/modal-body";
import { modalTitleLgMediumClassName } from "#/utils/modal-classes";
import { cn } from "#/utils/utils";

interface FactoryNameModalProps {
  isOpen: boolean;
  title: string;
  submitLabel: string;
  /** The name the field starts with; empty when creating. */
  initialName?: string;
  isPending?: boolean;
  onSubmit: (name: string) => void;
  onCancel: () => void;
}

/** One name field, shared by creating and renaming a factory. */
export function FactoryNameModal({
  isOpen,
  title,
  submitLabel,
  initialName = "",
  isPending = false,
  onSubmit,
  onCancel,
}: FactoryNameModalProps) {
  const { t } = useTranslation("openhands");
  const [name, setName] = React.useState(initialName);

  React.useEffect(() => {
    if (isOpen) setName(initialName);
  }, [isOpen, initialName]);

  if (!isOpen) return null;

  const trimmed = name.trim();
  const canSubmit = trimmed !== "" && !isPending;

  return (
    <ModalBackdrop onClose={onCancel} aria-label={title}>
      <form
        data-testid="factory-name-modal"
        className={cn(
          "relative w-full rounded-xl border border-border bg-surface p-6",
          modalWidthClassName("sm"),
          MODAL_MAX_WIDTH_VIEWPORT,
        )}
        onSubmit={(event) => {
          event.preventDefault();
          if (canSubmit) onSubmit(trimmed);
        }}
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
        <label className="mt-4 flex flex-col gap-2 text-sm">
          {t(I18nKey.FACTORIES$NAME_LABEL)}
          <input
            data-testid="factory-name-input"
            type="text"
            // eslint-disable-next-line jsx-a11y/no-autofocus
            autoFocus
            value={name}
            maxLength={200}
            onChange={(event) => setName(event.target.value)}
            className="rounded-lg border border-border bg-base-secondary px-3 py-2 text-sm text-content"
          />
        </label>
        <div className="mt-6 flex justify-end gap-3">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-lg border border-border px-4 py-2 text-sm text-contrast hover:bg-surface-raised"
          >
            {t(I18nKey.AUTOMATIONS$CANCEL)}
          </button>
          <BrandButton
            type="submit"
            variant="primary"
            testId="factory-name-submit"
            isDisabled={!canSubmit}
          >
            {submitLabel}
          </BrandButton>
        </div>
      </form>
    </ModalBackdrop>
  );
}
