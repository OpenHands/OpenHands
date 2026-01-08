import { useQueryClient } from "@tanstack/react-query";
import React from "react";
import { useTranslation } from "react-i18next";
import { useGetSecrets } from "#/hooks/query/use-get-secrets";
import { useDeleteSecret } from "#/hooks/mutation/use-delete-secret";
import { SecretForm } from "#/components/features/settings/secrets-settings/secret-form";
import {
  SecretListItem,
  SecretListItemSkeleton,
} from "#/components/features/settings/secrets-settings/secret-list-item";
import { BrandButton } from "#/components/features/settings/brand-button";
import { GetSecretsResponse } from "#/api/secrets-service.types";
import { I18nKey } from "#/i18n/declaration";
import { useModalStore } from "#/stores/modal-store";

function SecretsSettingsScreen() {
  const queryClient = useQueryClient();
  const { t } = useTranslation();

  const { data: secrets, isLoading: isLoadingSecrets } = useGetSecrets();
  const { mutate: deleteSecret } = useDeleteSecret();
  const openModal = useModalStore((state) => state.openModal);
  const closeModal = useModalStore((state) => state.closeModal);

  const [view, setView] = React.useState<
    "list" | "add-secret-form" | "edit-secret-form"
  >("list");
  const [selectedSecret, setSelectedSecret] = React.useState<string | null>(
    null,
  );

  const deleteSecretOptimistically = (secret: string) => {
    queryClient.setQueryData<GetSecretsResponse["custom_secrets"]>(
      ["secrets"],
      (oldSecrets) => {
        if (!oldSecrets) return [];
        return oldSecrets.filter((s) => s.name !== secret);
      },
    );
  };

  const revertOptimisticUpdate = () => {
    queryClient.invalidateQueries({ queryKey: ["secrets"] });
  };

  const handleDeleteSecret = (secret: string) => {
    deleteSecretOptimistically(secret);
    deleteSecret(secret, {
      onSettled: () => {
        closeModal();
      },
      onError: revertOptimisticUpdate,
    });
  };

  const handleOpenDeleteConfirmation = (secretName: string) => {
    setSelectedSecret(secretName);
    openModal("confirmation", {
      text: t("SECRETS$CONFIRM_DELETE_KEY"),
      onConfirm: () => {
        handleDeleteSecret(secretName);
      },
    });
  };

  return (
    <div data-testid="secrets-settings-screen" className="flex flex-col gap-5">
      {isLoadingSecrets && view === "list" && (
        <ul>
          <SecretListItemSkeleton />
          <SecretListItemSkeleton />
          <SecretListItemSkeleton />
        </ul>
      )}

      {view === "list" && (
        <BrandButton
          testId="add-secret-button"
          type="button"
          variant="primary"
          onClick={() => setView("add-secret-form")}
          isDisabled={isLoadingSecrets}
        >
          {t("SECRETS$ADD_NEW_SECRET")}
        </BrandButton>
      )}

      {view === "list" && (
        <div className="border border-tertiary rounded-md overflow-hidden">
          <table className="w-full">
            <thead className="bg-base-tertiary">
              <tr className="flex w-full items-center">
                <th className="w-1/4 text-left p-3 text-sm font-medium">
                  {t(I18nKey.SETTINGS$NAME)}
                </th>
                <th className="w-1/2 text-left p-3 text-sm font-medium">
                  {t(I18nKey.SECRETS$DESCRIPTION)}
                </th>
                <th className="w-1/4 text-right p-3 text-sm font-medium">
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
                    handleOpenDeleteConfirmation(secret.name);
                  }}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {(view === "add-secret-form" || view === "edit-secret-form") && (
        <SecretForm
          mode={view === "add-secret-form" ? "add" : "edit"}
          selectedSecret={selectedSecret}
          onCancel={() => setView("list")}
        />
      )}
    </div>
  );
}

export default SecretsSettingsScreen;
