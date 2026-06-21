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
import { useAuthScopes } from "#/hooks/query/use-auth-scopes";

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
  const { data: authScopes, isLoading: isLoadingScopes } = useAuthScopes();
  const [selectedScopes, setSelectedScopes] = useState<string[]>([]);

  // Set default scopes when data is loaded
  React.useEffect(() => {
    if (authScopes && isOpen) {
      setSelectedScopes(
        authScopes.filter((s) => s.is_default).map((s) => s.name),
      );
    }
  }, [authScopes, isOpen]);

  const createApiKeyMutation = useCreateApiKey();

  const handleCreateKey = async () => {
    if (!newKeyName.trim()) {
      displayErrorToast(t(I18nKey.ERROR$REQUIRED_FIELD));
      return;
    }

    if (selectedScopes.length === 0) {
      displayErrorToast("Please select at least one scope.");
      return;
    }

    const newKey = await mutateWithToast(
      createApiKeyMutation,
      { name: newKeyName, scopes: selectedScopes },
      {
        success: t(I18nKey.SETTINGS$API_KEY_CREATED),
        error: t(I18nKey.ERROR$GENERIC),
      },
    ).catch(() => null);

    if (newKey) {
      onKeyCreated(newKey);
      setNewKeyName("");
      // Reset selected scopes on next open
    }
  };

  const handleCancel = () => {
    setNewKeyName("");
    onClose();
  };

  const toggleScope = (scopeName: string) => {
    setSelectedScopes((prev) =>
      prev.includes(scopeName)
        ? prev.filter((s) => s !== scopeName)
        : [...prev, scopeName],
    );
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

        <div className="mt-6">
          {/* eslint-disable-next-line i18next/no-literal-string */}
          <div className="block text-sm font-medium mb-2">API Key Scopes</div>
          <div className="space-y-3 bg-base-tertiary p-4 rounded-md">
            {isLoadingScopes ? (
              <div className="flex justify-center py-2">
                <LoadingSpinner size="small" />
              </div>
            ) : (
              authScopes?.map((scope) => (
                <div key={scope.name} className="flex items-start gap-3 group">
                  <input
                    id={`scope-${scope.name}`}
                    type="checkbox"
                    className="mt-1 flex-shrink-0 cursor-pointer accent-blue-500"
                    checked={selectedScopes.includes(scope.name)}
                    onChange={() => toggleScope(scope.name)}
                    aria-label={scope.name}
                  />
                  <label
                    htmlFor={`scope-${scope.name}`}
                    className="cursor-pointer"
                  >
                    <div className="text-sm font-semibold">{scope.name}</div>
                    <div className="text-xs text-gray-400 group-hover:text-gray-300 transition-colors">
                      {scope.description}
                    </div>
                  </label>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </ApiKeyModalBase>
  );
}
