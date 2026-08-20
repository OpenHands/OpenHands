import { useQueryClient } from "@tanstack/react-query";
import React, { useCallback, useRef } from "react";
import { useTranslation } from "react-i18next";
import { BackNavButton } from "#/components/shared/buttons/back-nav-button";
import { useSearchSecrets } from "#/hooks/query/use-get-secrets";
import { useDeleteSecret } from "#/hooks/mutation/use-delete-secret";
import { SecretForm } from "#/components/features/settings/secrets-settings/secret-form";
import {
  SecretListItem,
  SecretListItemSkeleton,
} from "#/components/features/settings/secrets-settings/secret-list-item";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import {
  settingsListScrollContainerClassName,
  settingsListTableHeadClassName,
  settingsListTableHeaderCellClassName,
} from "#/utils/settings-list-classes";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";

export const handle = { hideTitle: true };

export function SecretsSettingsScreen() {
  const queryClient = useQueryClient();
  const { t } = useTranslation("openhands");
  const tableContainerRef = useRef<HTMLDivElement>(null);

  const {
    data: secrets,
    isLoading: isLoadingSecrets,
    hasNextPage,
    isFetchingNextPage,
    onLoadMore,
  } = useSearchSecrets();

  const { mutate: deleteSecret } = useDeleteSecret();

  const [view, setView] = React.useState<
    "list" | "add-secret-form" | "edit-secret-form"
  >("list");
  const [selectedSecret, setSelectedSecret] = React.useState<string | null>(
    null,
  );
  const [confirmationModalIsVisible, setConfirmationModalIsVisible] =
    React.useState(false);

  const handleScroll = useCallback(
    (e: React.UIEvent<HTMLDivElement>) => {
      const target = e.currentTarget;
      const isNearBottom =
        target.scrollHeight - target.scrollTop <= target.clientHeight + 100;

      if (isNearBottom && hasNextPage && !isFetchingNextPage) {
        onLoadMore();
      }
    },
    [hasNextPage, isFetchingNextPage, onLoadMore],
  );

  const invalidateSecrets = () => {
    queryClient.invalidateQueries({
      queryKey: ["secrets-search"],
    });
    queryClient.invalidateQueries({
      queryKey: ["secrets"],
    });
  };

  const handleDeleteSecret = (secret: string) => {
    deleteSecret(secret, {
      onSettled: () => {
        setConfirmationModalIsVisible(false);
      },
      onSuccess: invalidateSecrets,
      onError: invalidateSecrets,
    });
  };

  const onConfirmDeleteSecret = () => {
    if (selectedSecret) handleDeleteSecret(selectedSecret);
  };

  const onCancelDeleteSecret = () => {
    setConfirmationModalIsVisible(false);
  };

  const handleBackToList = () => {
    setView("list");
    setSelectedSecret(null);
  };

  const isFormView = view === "add-secret-form" || view === "edit-secret-form";
  const formTitle =
    view === "add-secret-form"
      ? t(I18nKey.SECRETS$ADD_A_SECRET)
      : t(I18nKey.SECRETS$EDIT_A_SECRET);

  return (
    <div data-testid="secrets-settings-screen" className="flex flex-col gap-6">
      {view === "list" ? (
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0 space-y-1">
            <Typography.H2>{t(I18nKey.SETTINGS$NAV_SECRETS)}</Typography.H2>
            <p
              data-testid="settings-page-subtitle"
              className="text-sm leading-5 text-tertiary-light"
            >
              {t(I18nKey.SETTINGS$PAGE_SECRETS_SUBLINE)}
            </p>
          </div>
          <BrandButton
            testId="add-secret-button"
            type="button"
            variant="primary"
            className="shrink-0 whitespace-nowrap"
            onClick={() => setView("add-secret-form")}
            isDisabled={isLoadingSecrets}
          >
            {t(I18nKey.SECRETS$ADD_NEW_SECRET)}
          </BrandButton>
        </div>
      ) : null}

      {isFormView ? (
        <div className="flex flex-col gap-2">
          <BackNavButton testId="back-to-secrets" onClick={handleBackToList}>
            {t(I18nKey.BUTTON$BACK)}
          </BackNavButton>
          <Typography.H2 testId="secret-editor-title">
            {formTitle}
          </Typography.H2>
        </div>
      ) : null}

      {isLoadingSecrets && view === "list" && (
        <ul>
          <SecretListItemSkeleton />
          <SecretListItemSkeleton />
          <SecretListItemSkeleton />
        </ul>
      )}

      {view === "list" && !isLoadingSecrets && secrets?.length === 0 && (
        <div
          data-testid="secrets-empty"
          className={extensionModuleEmptyStateClassName}
        >
          <p className="text-sm text-[var(--oh-muted)]">
            {t(I18nKey.SECRETS$EMPTY)}
          </p>
        </div>
      )}

      {view === "list" && !isLoadingSecrets && (secrets?.length ?? 0) > 0 && (
        <div
          ref={tableContainerRef}
          className={settingsListScrollContainerClassName}
          onScroll={handleScroll}
        >
          <table className="w-full min-w-full table-fixed">
            <thead className={settingsListTableHeadClassName}>
              <tr>
                <th
                  className={cn(settingsListTableHeaderCellClassName, "w-1/4")}
                >
                  {t(I18nKey.SETTINGS$NAME)}
                </th>
                <th
                  className={cn(settingsListTableHeaderCellClassName, "w-1/2")}
                >
                  {t(I18nKey.SECRETS$DESCRIPTION)}
                </th>
                <th
                  className={cn(
                    settingsListTableHeaderCellClassName,
                    "w-1/4 text-right",
                  )}
                >
                  {t(I18nKey.SETTINGS$ACTIONS)}
                </th>
              </tr>
            </thead>
            <tbody>
              {secrets?.map((secret) => (
                <SecretListItem
                  key={secret.name}
                  title={secret.name}
                  description={secret.description}
                  onEdit={() => {
                    setView("edit-secret-form");
                    setSelectedSecret(secret.name);
                  }}
                  onDelete={() => {
                    setConfirmationModalIsVisible(true);
                    setSelectedSecret(secret.name);
                  }}
                />
              ))}
            </tbody>
          </table>

          {isFetchingNextPage && (
            <div className="flex justify-center p-4">
              <LoadingSpinner size="small" />
            </div>
          )}
        </div>
      )}

      {(view === "add-secret-form" || view === "edit-secret-form") && (
        <SecretForm
          mode={view === "add-secret-form" ? "add" : "edit"}
          selectedSecret={selectedSecret}
          onCancel={handleBackToList}
        />
      )}

      {confirmationModalIsVisible && (
        <ConfirmationModal
          text={t(I18nKey.SECRETS$CONFIRM_DELETE_KEY)}
          onConfirm={onConfirmDeleteSecret}
          onCancel={onCancelDeleteSecret}
        />
      )}
    </div>
  );
}

export default SecretsSettingsScreen;
