import React, { useState } from "react";
import { useTranslation, Trans } from "react-i18next";
import { FaTrash, FaEye, FaEyeSlash, FaCopy } from "react-icons/fa6";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ApiKey, CreateApiKeyResponse } from "#/api/api-keys";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { CreateApiKeyModal } from "./create-api-key-modal";
import { DeleteApiKeyModal } from "./delete-api-key-modal";
import { NewApiKeyModal } from "./new-api-key-modal";
import { useApiKeys } from "#/hooks/query/use-api-keys";
import { useLlmApiKey } from "#/hooks/query/use-llm-api-key";
import { useRefreshLlmApiKey } from "#/hooks/mutation/use-refresh-llm-api-key";

interface LlmApiKeyManagerProps {
  llmApiKey: { key: string | null } | undefined;
  isLoadingLlmKey: boolean;
  refreshLlmApiKey: ReturnType<typeof useRefreshLlmApiKey>;
}

function LlmApiKeyManager({
  llmApiKey,
  isLoadingLlmKey,
  refreshLlmApiKey,
}: LlmApiKeyManagerProps) {
  const { t } = useTranslation();
  const [showLlmApiKey, setShowLlmApiKey] = useState(false);

  const handleRefreshLlmApiKey = () => {
    refreshLlmApiKey.mutate(undefined, {
      onSuccess: () => {
        displaySuccessToast(
          t(I18nKey.SETTINGS$API_KEY_REFRESHED, {
            defaultValue: "API key refreshed successfully",
          }),
        );
      },
      onError: () => {
        displayErrorToast(t(I18nKey.ERROR$GENERIC));
      },
    });
  };

  if (isLoadingLlmKey || !llmApiKey) {
    return null;
  }

  return (
    <div className="border-b border-gray-200 pb-6 mb-6 flex flex-col gap-6">
      <h3 className="text-xl font-medium text-white">
        {t(I18nKey.SETTINGS$LLM_API_KEY)}
      </h3>
      <div className="flex items-center justify-between">
        <BrandButton
          type="button"
          variant="primary"
          onClick={handleRefreshLlmApiKey}
          isDisabled={refreshLlmApiKey.isPending}
        >
          {refreshLlmApiKey.isPending ? (
            <LoadingSpinner size="small" />
          ) : (
            t(I18nKey.SETTINGS$REFRESH_LLM_API_KEY)
          )}
        </BrandButton>
      </div>
      <div>
        <p className="text-sm text-gray-300 mb-2">
          {t(I18nKey.SETTINGS$LLM_API_KEY_DESCRIPTION)}
        </p>
        <div className="flex items-center gap-2">
          <div className="flex-1 bg-base-tertiary rounded-md py-2 flex items-center">
            <div className="flex-1">
              {llmApiKey.key ? (
                <div className="flex items-center">
                  {showLlmApiKey ? (
                    <span className="text-white font-mono">
                      {llmApiKey.key}
                    </span>
                  ) : (
                    <span className="text-white">{"•".repeat(20)}</span>
                  )}
                </div>
              ) : (
                <span className="text-white">
                  {t(I18nKey.API$NO_KEY_AVAILABLE)}
                </span>
              )}
            </div>
            <div className="flex items-center">
              {llmApiKey.key && (
                <button
                  type="button"
                  className="text-white hover:text-gray-300 mr-2 cursor-pointer"
                  aria-label={showLlmApiKey ? "Hide API key" : "Show API key"}
                  title={showLlmApiKey ? "Hide API key" : "Show API key"}
                  onClick={() => setShowLlmApiKey(!showLlmApiKey)}
                >
                  {showLlmApiKey ? (
                    <FaEyeSlash size={20} />
                  ) : (
                    <FaEye size={20} />
                  )}
                </button>
              )}
              <button
                type="button"
                className="text-white hover:text-gray-300 mr-2 cursor-pointer"
                aria-label="Copy API key"
                title="Copy API key"
                onClick={() => {
                  if (llmApiKey.key) {
                    navigator.clipboard.writeText(llmApiKey.key);
                    displaySuccessToast(t(I18nKey.SETTINGS$API_KEY_COPIED));
                  }
                }}
              >
                <FaCopy size={20} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

interface ApiKeysTableProps {
  apiKeys: ApiKey[];
  isLoading: boolean;
  onDeleteKey: (key: ApiKey) => void;
}

function ApiKeysTable({ apiKeys, isLoading, onDeleteKey }: ApiKeysTableProps) {
  const { t } = useTranslation();

  const formatDate = (dateString: string | null) => {
    if (!dateString) return "Never";
    return new Date(dateString).toLocaleString();
  };

  if (isLoading) {
    return (
      <div className="flex justify-center p-4">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  if (!Array.isArray(apiKeys) || apiKeys.length === 0) {
    return null;
  }

  return (
    <div className="border border-tertiary rounded-md overflow-hidden">
      <table className="w-full">
        <thead className="bg-base-tertiary">
          <tr>
            <th className="text-left p-3 text-sm font-medium">
              {t(I18nKey.SETTINGS$NAME)}
            </th>
            <th className="text-left p-3 text-sm font-medium">
              {t(I18nKey.SETTINGS$CREATED_AT)}
            </th>
            <th className="text-left p-3 text-sm font-medium">
              {t(I18nKey.SETTINGS$LAST_USED)}
            </th>
            <th className="text-right p-3 text-sm font-medium">
              {t(I18nKey.SETTINGS$ACTIONS)}
            </th>
          </tr>
        </thead>
        <tbody>
          {apiKeys.map((key) => (
            <tr key={key.id} className="border-t border-tertiary">
              <td
                className="p-3 text-sm truncate max-w-[160px]"
                title={key.name}
              >
                {key.name}
              </td>
              <td className="p-3 text-sm">{formatDate(key.created_at)}</td>
              <td className="p-3 text-sm">{formatDate(key.last_used_at)}</td>
              <td className="p-3 text-right">
                <button
                  type="button"
                  onClick={() => onDeleteKey(key)}
                  aria-label={`Delete ${key.name}`}
                  className="cursor-pointer"
                >
                  <FaTrash size={16} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ApiKeysManager() {
  const { t } = useTranslation();
  const { data: apiKeys = [], isLoading, error } = useApiKeys();
  const { data: llmApiKey, isLoading: isLoadingLlmKey } = useLlmApiKey();
  const refreshLlmApiKey = useRefreshLlmApiKey();
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [keyToDelete, setKeyToDelete] = useState<ApiKey | null>(null);
  const [newlyCreatedKey, setNewlyCreatedKey] =
    useState<CreateApiKeyResponse | null>(null);
  const [showNewKeyModal, setShowNewKeyModal] = useState(false);

  // Display error toast if the query fails
  if (error) {
    displayErrorToast(t(I18nKey.ERROR$GENERIC));
  }

  const handleKeyCreated = (newKey: CreateApiKeyResponse) => {
    setNewlyCreatedKey(newKey);
    setCreateModalOpen(false);
    setShowNewKeyModal(true);
  };

  const handleCloseCreateModal = () => {
    setCreateModalOpen(false);
  };

  const handleCloseDeleteModal = () => {
    setDeleteModalOpen(false);
    setKeyToDelete(null);
  };

  const handleCloseNewKeyModal = () => {
    setShowNewKeyModal(false);
    setNewlyCreatedKey(null);
  };

  const handleDeleteKey = (key: ApiKey) => {
    setKeyToDelete(key);
    setDeleteModalOpen(true);
  };

  return (
    <>
      <div className="flex flex-col gap-6">
        <LlmApiKeyManager
          llmApiKey={llmApiKey}
          isLoadingLlmKey={isLoadingLlmKey}
          refreshLlmApiKey={refreshLlmApiKey}
        />

        <h3 className="text-xl font-medium text-white">
          {t(I18nKey.SETTINGS$OPENHANDS_API_KEYS)}
        </h3>

        <div className="flex items-center justify-between">
          <BrandButton
            type="button"
            variant="primary"
            onClick={() => setCreateModalOpen(true)}
          >
            {t(I18nKey.SETTINGS$CREATE_API_KEY)}
          </BrandButton>
        </div>

        <p className="text-sm text-gray-300">
          <Trans
            i18nKey={I18nKey.SETTINGS$API_KEYS_DESCRIPTION}
            components={{
              a: (
                <a
                  href="https://docs.all-hands.dev/usage/cloud/cloud-api"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-400 hover:underline"
                >
                  API documentation
                </a>
              ),
            }}
          />
        </p>

        <ApiKeysTable
          apiKeys={apiKeys}
          isLoading={isLoading}
          onDeleteKey={handleDeleteKey}
        />
      </div>

      {/* Create API Key Modal */}
      <CreateApiKeyModal
        isOpen={createModalOpen}
        onClose={handleCloseCreateModal}
        onKeyCreated={handleKeyCreated}
      />

      {/* Delete API Key Modal */}
      <DeleteApiKeyModal
        isOpen={deleteModalOpen}
        keyToDelete={keyToDelete}
        onClose={handleCloseDeleteModal}
      />

      {/* Show New API Key Modal */}
      <NewApiKeyModal
        isOpen={showNewKeyModal}
        newlyCreatedKey={newlyCreatedKey}
        onClose={handleCloseNewKeyModal}
      />
    </>
  );
}
