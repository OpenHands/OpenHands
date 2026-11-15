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
import type { ChangeRunAsRequest } from "#/types/automation";

const SELF_TARGET = "self";

interface ChangeRunAsModalProps {
  automationName: string;
  isOpen: boolean;
  /** Active service accounts of the org the automation can move to. */
  serviceAccounts: CloudServiceAccount[];
  /** Id of the current run-as identity, left out of the targets. */
  currentRunAsId?: string;
  /** Whether "me" is offered: only `manage_automations` holders may make it run as themselves. */
  canChangeToSelf: boolean;
  /** The service account list is still loading. */
  isLoading?: boolean;
  /** The service account list failed to load. */
  isError?: boolean;
  isPending?: boolean;
  onConfirm: (body: ChangeRunAsRequest) => void;
  onCancel: () => void;
}

/**
 * Pick who an automation runs as: a service account of the org, or the
 * caller. The service turns the automation off after the change.
 */
export function ChangeRunAsModal({
  automationName,
  isOpen,
  serviceAccounts,
  currentRunAsId,
  canChangeToSelf,
  isLoading = false,
  isError = false,
  isPending = false,
  onConfirm,
  onCancel,
}: ChangeRunAsModalProps) {
  const { t } = useTranslation("openhands");
  const targets = serviceAccounts.filter(
    (account) => !account.is_disabled && account.id !== currentRunAsId,
  );
  const [target, setTarget] = React.useState<string>("");
  const defaultTarget = canChangeToSelf ? SELF_TARGET : (targets[0]?.id ?? "");

  // Reset when the dialog opens, and again when a list that arrives later
  // changes the default target.
  React.useEffect(() => {
    if (!isOpen) return;
    setTarget(defaultTarget);
  }, [isOpen, defaultTarget]);

  // Nothing to pick: say why instead of showing an empty select.
  let emptyMessage: string | null = null;
  if (targets.length === 0 && !canChangeToSelf) {
    if (isLoading) emptyMessage = t(I18nKey.AUTOMATIONS$CHANGE_RUN_AS_LOADING);
    else if (isError)
      emptyMessage = t(I18nKey.AUTOMATIONS$CHANGE_RUN_AS_LOAD_ERROR);
    else emptyMessage = t(I18nKey.AUTOMATIONS$CHANGE_RUN_AS_NO_TARGETS);
  }

  if (!isOpen) return null;

  const title = t(I18nKey.AUTOMATIONS$CHANGE_RUN_AS_TITLE);
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
          {t(I18nKey.AUTOMATIONS$CHANGE_RUN_AS_MESSAGE, {
            name: automationName,
          })}
        </p>
        {emptyMessage ? (
          <p
            data-testid="change-run-as-empty"
            className="mt-4 text-sm text-muted"
          >
            {emptyMessage}
          </p>
        ) : null}
        <label
          className={cn(
            "mt-4 flex flex-col gap-2 text-sm",
            emptyMessage && "hidden",
          )}
        >
          {t(I18nKey.AUTOMATIONS$CHANGE_RUN_AS_TARGET)}
          <select
            data-testid="change-run-as-target"
            value={target}
            onChange={(event) => setTarget(event.target.value)}
            className="rounded-lg border border-border bg-base-secondary px-3 py-2 text-sm text-content"
          >
            {canChangeToSelf ? (
              <option value={SELF_TARGET}>
                {t(I18nKey.AUTOMATIONS$RUN_AS_ME)}
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
            testId="change-run-as-confirm"
            isDisabled={!target || isPending}
            onClick={handleConfirm}
          >
            {t(I18nKey.AUTOMATIONS$CHANGE_RUN_AS)}
          </BrandButton>
        </div>
      </div>
    </ModalBackdrop>
  );
}
