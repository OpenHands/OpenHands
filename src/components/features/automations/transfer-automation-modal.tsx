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
import type { CloudServiceAccount } from "#/api/cloud/types";
import type { TransferAutomationRequest } from "#/types/automation";

const SELF_TARGET = "self";

interface TransferAutomationModalProps {
  automationName: string;
  isOpen: boolean;
  /** Active service accounts of the org the automation can move to. */
  serviceAccounts: CloudServiceAccount[];
  /** Id of the current owner, left out of the targets. */
  currentOwnerId?: string;
  /** Whether "me" is offered: only `manage_automations` holders may take ownership. */
  canTransferToSelf: boolean;
  isPending?: boolean;
  onConfirm: (body: TransferAutomationRequest) => void;
  onCancel: () => void;
}

/**
 * Pick the new owner of an automation: a service account of the org, or the
 * caller. The service turns the automation off after a transfer.
 */
export function TransferAutomationModal({
  automationName,
  isOpen,
  serviceAccounts,
  currentOwnerId,
  canTransferToSelf,
  isPending = false,
  onConfirm,
  onCancel,
}: TransferAutomationModalProps) {
  const { t } = useTranslation("openhands");
  const targets = serviceAccounts.filter(
    (account) => !account.is_disabled && account.id !== currentOwnerId,
  );
  const [target, setTarget] = React.useState<string>("");

  React.useEffect(() => {
    if (!isOpen) return;
    setTarget(canTransferToSelf ? SELF_TARGET : (targets[0]?.id ?? ""));
    // Reset only when the dialog opens.
  }, [isOpen]);

  if (!isOpen) return null;

  const title = t(I18nKey.AUTOMATIONS$TRANSFER_TITLE);
  const handleConfirm = () => {
    if (!target) return;
    onConfirm(
      target === SELF_TARGET
        ? { to: "self" }
        : { to: "service", service_account_id: target },
    );
  };

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
          {t(I18nKey.AUTOMATIONS$TRANSFER_MESSAGE, { name: automationName })}
        </p>
        <label className="mt-4 flex flex-col gap-2 text-sm">
          {t(I18nKey.AUTOMATIONS$TRANSFER_TARGET)}
          <select
            data-testid="transfer-automation-target"
            value={target}
            onChange={(event) => setTarget(event.target.value)}
            className="rounded-lg border border-border bg-base-secondary px-3 py-2 text-sm text-content"
          >
            {canTransferToSelf ? (
              <option value={SELF_TARGET}>
                {t(I18nKey.AUTOMATIONS$OWNER_ME)}
              </option>
            ) : null}
            {targets.map((account) => (
              <option key={account.id} value={account.id}>
                {account.name}
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
            testId="transfer-automation-confirm"
            isDisabled={!target || isPending}
            onClick={handleConfirm}
          >
            {t(I18nKey.AUTOMATIONS$TRANSFER)}
          </BrandButton>
        </div>
      </div>
    </ModalBackdrop>
  );
}
