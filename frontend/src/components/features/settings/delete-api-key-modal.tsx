import React from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ApiKey } from "#/api/api-keys";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { useDeleteApiKey } from "#/hooks/mutation/use-delete-api-key";
import { ApiKeyModalBase } from "./api-key-modal-base";

interface DeleteApiKeyModalProps {
  keyToDelete: ApiKey | null;
  onClose: () => void;
  onDeleted: () => void;
}

export function DeleteApiKeyModal({
  keyToDelete,
  onClose,
  onDeleted,
}: DeleteApiKeyModalProps) {
  const { t } = useTranslation();
  const deleteApiKeyMutation = useDeleteApiKey();

  const handleDeleteKey = async () => {
    if (!keyToDelete) return;

    try {
      await deleteApiKeyMutation.mutateAsync(keyToDelete.id);
      displaySuccessToast(t(I18nKey.SETTINGS$API_KEY_DELETED));
      onDeleted();
      onClose();
    } catch {
      displayErrorToast(t(I18nKey.ERROR$GENERIC));
    }
  };

  if (!keyToDelete) return null;

  const modalFooter = (
    <>
      <BrandButton
        type="button"
        variant="danger"
        className="grow"
        onClick={handleDeleteKey}
        isDisabled={deleteApiKeyMutation.isPending}
      >
        {deleteApiKeyMutation.isPending ? (
          <LoadingSpinner size="small" />
        ) : (
          t(I18nKey.BUTTON$DELETE)
        )}
      </BrandButton>
      <BrandButton
        type="button"
        variant="secondary"
        className="grow"
        onClick={onClose}
        isDisabled={deleteApiKeyMutation.isPending}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
    </>
  );

  return (
    <ApiKeyModalBase
      title={t(I18nKey.SETTINGS$DELETE_API_KEY)}
      footer={modalFooter}
    >
      <div data-testid="delete-api-key-modal">
        <p className="text-sm break-all">
          {t(I18nKey.SETTINGS$DELETE_API_KEY_CONFIRMATION, {
            name: keyToDelete.name,
          })}
        </p>
      </div>
    </ApiKeyModalBase>
  );
}
