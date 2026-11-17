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
import type { Factory } from "#/types/automation";

const NO_FACTORY = "";

interface ChangeFactoryModalProps {
  automationName: string;
  isOpen: boolean;
  factories: Factory[];
  /** The factory the automation is in now, if any. */
  currentFactoryId?: string | null;
  isLoading?: boolean;
  isPending?: boolean;
  /** The chosen factory id, or `null` to take the automation out of its factory. */
  onConfirm: (factoryId: string | null) => void;
  onCancel: () => void;
}

/** Pick the factory an automation belongs to, or none. */
export function ChangeFactoryModal({
  automationName,
  isOpen,
  factories,
  currentFactoryId,
  isLoading = false,
  isPending = false,
  onConfirm,
  onCancel,
}: ChangeFactoryModalProps) {
  const { t } = useTranslation("openhands");
  const current = currentFactoryId ?? NO_FACTORY;
  const [target, setTarget] = React.useState(current);

  React.useEffect(() => {
    if (isOpen) setTarget(current);
  }, [isOpen, current]);

  if (!isOpen) return null;

  const title = t(I18nKey.AUTOMATIONS$CHANGE_FACTORY);

  return (
    <ModalBackdrop onClose={onCancel} aria-label={title}>
      <div
        className={cn(
          "relative w-full rounded-xl border border-border bg-surface p-6",
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
          {t(I18nKey.AUTOMATIONS$CHANGE_FACTORY_MESSAGE, {
            name: automationName,
          })}
        </p>
        <label className="mt-4 flex flex-col gap-2 text-sm">
          {t(I18nKey.AUTOMATIONS$CHANGE_FACTORY_TARGET)}
          <select
            data-testid="change-factory-target"
            value={target}
            disabled={isLoading}
            onChange={(event) => setTarget(event.target.value)}
            className="rounded-lg border border-border bg-base-secondary px-3 py-2 text-sm text-content"
          >
            <option value={NO_FACTORY}>
              {t(I18nKey.AUTOMATIONS$CHANGE_FACTORY_NONE)}
            </option>
            {factories.map((factory) => (
              <option key={factory.id} value={factory.id}>
                {factory.name}
              </option>
            ))}
          </select>
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
            type="button"
            variant="primary"
            testId="change-factory-confirm"
            isDisabled={isLoading || isPending || target === current}
            onClick={() => onConfirm(target === NO_FACTORY ? null : target)}
          >
            {t(I18nKey.BUTTON$SAVE)}
          </BrandButton>
        </div>
      </div>
    </ModalBackdrop>
  );
}
