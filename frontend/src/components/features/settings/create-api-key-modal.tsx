import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { CreateApiKeyResponse } from "#/api/api-keys";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { mutateWithToast } from "#/utils/mutate-with-toast";
import { ApiKeyModalBase } from "./api-key-modal-base";
import { useCreateApiKey } from "#/hooks/mutation/use-create-api-key";

interface CreateApiKeyModalProps {
  isOpen: boolean;
  onClose: () => void;
  onKeyCreated: (newKey: CreateApiKeyResponse) => void;
}

export function CreateApiKeyModal({
  isOpen,
  onClose,
  onKeyCreated,
}: CreateApiKeyModalProps) {
  const { t } = useTranslation();
  const [newKeyName, setNewKeyName] = useState("");

  const createApiKeyMutation = useCreateApiKey();

  const handleCreateKey = async () => {
    if (!newKeyName.trim()) {
      displayErrorToast(t(I18nKey.ERROR$REQUIRED_FIELD));
      return;
    }

    const newKey = await mutateWithToast(createApiKeyMutation, newKeyName, {
      success: t(I18nKey.SETTINGS$API_KEY_CREATED),
      error: t(I18nKey.ERROR$GENERIC),
    }).catch(() => null);

    if (newKey) {
      onKeyCreated(newKey);
      setNewKeyName("");
    }
  };

  const handleCancel = () => {
    setNewKeyName("");
    onClose();
  };

  const modalFooter = (
    <>
      <BrandButton
        type="button"
        variant="primary"
        className="grow"
        onClick={handleCreateKey}
        isDisabled={createApiKeyMutation.isPending || !newKeyName.trim()}
      >
        {createApiKeyMutation.isPending ? (
          <LoadingSpinner size="small" />
        ) : (
          t(I18nKey.BUTTON$CREATE)
        )}
      </BrandButton>
      <BrandButton
        type="button"
        variant="secondary"
        className="grow"
        onClick={handleCancel}
        isDisabled={createApiKeyMutation.isPending}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
    </>
  );

  return (
    <ApiKeyModalBase
      isOpen={isOpen}
      title={t(I18nKey.SETTINGS$CREATE_API_KEY)}
      footer={modalFooter}
    >
      <div data-testid="create-api-key-modal">
        <p className="text-sm text-gray-300">
          {t(I18nKey.SETTINGS$CREATE_API_KEY_DESCRIPTION)}
        </p>
        <SettingsInput
          testId="api-key-name-input"
          label={t(I18nKey.SETTINGS$NAME)}
          placeholder={t(I18nKey.SETTINGS$API_KEY_NAME_PLACEHOLDER)}
          value={newKeyName}
          onChange={(value) => setNewKeyName(value)}
          className="w-full mt-4"
          type="text"
        />
      </div>
    </ApiKeyModalBase>
  );
}
