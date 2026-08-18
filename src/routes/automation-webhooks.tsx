import React from "react";
import { useTranslation } from "react-i18next";
import { BackNavButton } from "#/components/shared/buttons/back-nav-button";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { BrandButton } from "#/components/features/settings/brand-button";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import {
  useWebhooks,
  useDeleteWebhook,
  useRotateWebhookSecret,
} from "#/hooks/query/use-webhooks";
import {
  settingsListScrollContainerClassName,
  settingsListTableHeadClassName,
  settingsListTableHeaderCellClassName,
} from "#/utils/settings-list-classes";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";
import {
  WebhookListItem,
  WebhookListItemSkeleton,
} from "#/components/features/automations/webhooks/webhook-list-item";
import { WebhookForm } from "#/components/features/automations/webhooks/webhook-form";
import { WebhookSecretReveal } from "#/components/features/automations/webhooks/webhook-secret-reveal";
import type { CustomWebhook } from "#/types/webhook";

export const handle = { hideTitle: true };

export function AutomationWebhooksScreen() {
  const { t } = useTranslation("openhands");
  const { data, isLoading } = useWebhooks();
  const { mutate: deleteWebhook } = useDeleteWebhook();
  const { mutate: rotateSecret } = useRotateWebhookSecret();

  const [view, setView] = React.useState<"list" | "add" | "edit">("list");
  const [selected, setSelected] = React.useState<CustomWebhook | null>(null);
  const [confirmDeleteVisible, setConfirmDeleteVisible] = React.useState(false);
  const [rotatedSecret, setRotatedSecret] = React.useState<string | null>(null);

  const handleBackToList = () => {
    setView("list");
    setSelected(null);
  };

  const handleConfirmDelete = () => {
    if (selected) {
      deleteWebhook(selected.id, {
        onSettled: () => setConfirmDeleteVisible(false),
      });
    }
  };

  const handleRotate = (webhook: CustomWebhook) => {
    rotateSecret(webhook.id, {
      onSuccess: ({ webhook_secret: secret }) => setRotatedSecret(secret),
    });
  };

  const webhooks = data?.webhooks ?? [];
  const isFormView = view === "add" || view === "edit";
  const formTitle =
    view === "add"
      ? t(I18nKey.AUTOMATIONS$WEBHOOKS$ADD_TITLE)
      : t(I18nKey.AUTOMATIONS$WEBHOOKS$EDIT_TITLE);

  return (
    <div
      data-testid="automation-webhooks-screen"
      className="flex flex-col gap-6"
    >
      {view === "list" && (
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0 space-y-1">
            <Typography.H2>
              {t(I18nKey.AUTOMATIONS$WEBHOOKS$TITLE)}
            </Typography.H2>
            <p className="text-sm leading-5 text-tertiary-light">
              {t(I18nKey.AUTOMATIONS$WEBHOOKS$SUBLINE)}
            </p>
          </div>
          <BrandButton
            testId="add-webhook-button"
            type="button"
            variant="primary"
            className="shrink-0 whitespace-nowrap"
            onClick={() => setView("add")}
            isDisabled={isLoading}
          >
            {t(I18nKey.AUTOMATIONS$WEBHOOKS$ADD_NEW)}
          </BrandButton>
        </div>
      )}

      {isFormView && (
        <div className="flex flex-col gap-2">
          <BackNavButton testId="back-to-webhooks" onClick={handleBackToList}>
            {t(I18nKey.BUTTON$BACK)}
          </BackNavButton>
          <Typography.H2>{formTitle}</Typography.H2>
        </div>
      )}

      {rotatedSecret && (
        <WebhookSecretReveal
          secret={rotatedSecret}
          onDismiss={() => setRotatedSecret(null)}
        />
      )}

      {isLoading && view === "list" && (
        <ul>
          <WebhookListItemSkeleton />
          <WebhookListItemSkeleton />
        </ul>
      )}

      {view === "list" && !isLoading && webhooks.length === 0 && (
        <div
          data-testid="webhooks-empty"
          className={extensionModuleEmptyStateClassName}
        >
          <p className="text-sm text-[var(--oh-muted)]">
            {t(I18nKey.AUTOMATIONS$WEBHOOKS$EMPTY)}
          </p>
        </div>
      )}

      {view === "list" && !isLoading && webhooks.length > 0 && (
        <div className={settingsListScrollContainerClassName}>
          <table className="w-full min-w-full table-fixed">
            <thead className={settingsListTableHeadClassName}>
              <tr>
                <th
                  className={cn(settingsListTableHeaderCellClassName, "w-1/4")}
                >
                  {t(I18nKey.SETTINGS$NAME)}
                </th>
                <th
                  className={cn(settingsListTableHeaderCellClassName, "w-1/5")}
                >
                  {t(I18nKey.AUTOMATIONS$WEBHOOKS$SOURCE)}
                </th>
                <th
                  className={cn(settingsListTableHeaderCellClassName, "w-2/5")}
                >
                  {t(I18nKey.AUTOMATIONS$WEBHOOKS$URL)}
                </th>
                <th
                  className={cn(
                    settingsListTableHeaderCellClassName,
                    "w-1/5 text-right",
                  )}
                >
                  {t(I18nKey.SETTINGS$ACTIONS)}
                </th>
              </tr>
            </thead>
            <tbody>
              {webhooks.map((webhook) => (
                <WebhookListItem
                  key={webhook.id}
                  webhook={webhook}
                  onEdit={() => {
                    setSelected(webhook);
                    setView("edit");
                  }}
                  onRotateSecret={() => handleRotate(webhook)}
                  onDelete={() => {
                    setSelected(webhook);
                    setConfirmDeleteVisible(true);
                  }}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {isLoading && view === "list" && webhooks.length === 0 && (
        <div className="flex justify-center p-4">
          <LoadingSpinner size="small" />
        </div>
      )}

      {isFormView && (
        <WebhookForm
          mode={view === "add" ? "add" : "edit"}
          webhook={selected}
          onCancel={handleBackToList}
          onDone={handleBackToList}
        />
      )}

      {confirmDeleteVisible && selected && (
        <ConfirmationModal
          text={t(I18nKey.AUTOMATIONS$WEBHOOKS$CONFIRM_DELETE, {
            name: selected.name,
          })}
          onConfirm={handleConfirmDelete}
          onCancel={() => setConfirmDeleteVisible(false)}
        />
      )}
    </div>
  );
}

export default AutomationWebhooksScreen;
