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
import { ApiKeyModalBase } from "./api-key-modal-base";
import { useDeleteApiKey } from "#/hooks/mutation/use-delete-api-key";

interface DeleteApiKeyModalProps {
  isOpen: boolean;
  keyToDelete: ApiKey | null;
  onClose: () => void;
}

export function DeleteApiKeyModal({
  isOpen,
  keyToDelete,
  onClose,
}: DeleteApiKeyModalProps) {
  const { t } = useTranslation();
  const deleteApiKeyMutation = useDeleteApiKey();

  const handleDeleteKey = async () => {
    if (!keyToDelete) return;

    try {
      await deleteApiKeyMutation.mutateAsync(keyToDelete.id);
      displaySuccessToast(t(I18nKey.SETTINGS$API_KEY_DELETED));
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
      isOpen={isOpen && !!keyToDelete}
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
