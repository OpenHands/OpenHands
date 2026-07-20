import React, { useMemo, useState } from "react";
import { useTranslation, Trans } from "react-i18next";
import { useNavigate } from "react-router";
import { FaTrash, FaEye, FaEyeSlash, FaCopy } from "react-icons/fa6";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ApiKey, CreateApiKeyResponse } from "#/api/api-keys";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { mutateWithToast } from "#/utils/mutate-with-toast";
import { CreateApiKeyModal } from "./create-api-key-modal";
import { DeleteApiKeyModal } from "./delete-api-key-modal";
import { NewApiKeyModal } from "./new-api-key-modal";
import { useApiKeys } from "#/hooks/query/use-api-keys";
import { useLlmApiKey } from "#/hooks/query/use-llm-api-key";
import { useRefreshLlmApiKey } from "#/hooks/mutation/use-refresh-llm-api-key";
import { useOrganizations } from "#/hooks/query/use-organizations";

interface LlmApiKeyManagerProps {
  llmApiKey: { key: string | null } | undefined;
  isLoadingLlmKey: boolean;
  isPaymentRequired: boolean;
  refreshLlmApiKey: ReturnType<typeof useRefreshLlmApiKey>;
}

function LlmApiKeyPaywall() {
  const { t } = useTranslation();
  const navigate = useNavigate();

  return (
    <div className="border-b border-gray-200 pb-6 mb-6 flex flex-col gap-6">
      <h3 className="text-xl font-medium text-white">
        {t(I18nKey.SETTINGS$LLM_API_KEY)}
      </h3>
      <div className="bg-base-tertiary rounded-md p-4 flex flex-col gap-4">
        <p className="text-sm text-gray-300">
          {t(I18nKey.SETTINGS$LLM_API_KEY_PAYWALL_MESSAGE)}
        </p>
        <div>
          <BrandButton
            type="button"
            variant="primary"
            onClick={() => navigate("/settings/billing")}
          >
            {t(I18nKey.SETTINGS$LLM_API_KEY_BUY_NOW)}
          </BrandButton>
        </div>
      </div>
    </div>
  );
}

function LlmApiKeyManager({
  llmApiKey,
  isLoadingLlmKey,
  isPaymentRequired,
  refreshLlmApiKey,
}: LlmApiKeyManagerProps) {
  const { t } = useTranslation();
  const [showLlmApiKey, setShowLlmApiKey] = useState(false);

  const handleRefreshLlmApiKey = async () => {
    await mutateWithToast(refreshLlmApiKey, undefined, {
      success: t(I18nKey.SETTINGS$API_KEY_REFRESHED),
      error: t(I18nKey.ERROR$GENERIC),
    });
  };

  // Show paywall if payment is required
  if (isPaymentRequired) {
    return <LlmApiKeyPaywall />;
  }

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

type ApiKeyStatus = "active" | "pending" | "expired";

const getApiKeyStatus = (key: ApiKey): ApiKeyStatus => {
  const now = Date.now();
  if (key.expires_at && new Date(key.expires_at).getTime() < now) {
    return "expired";
  }
  if (key.not_before && new Date(key.not_before).getTime() > now) {
    return "pending";
  }
  return "active";
};

const STATUS_BADGE_CLASSES: Record<ApiKeyStatus, string> = {
  active: "bg-green-500/20 text-green-300",
  pending: "bg-yellow-500/20 text-yellow-300",
  expired: "bg-red-500/20 text-red-300",
};

function ApiKeyStatusBadge({ status }: { status: ApiKeyStatus }) {
  const { t } = useTranslation();
  const labelKey = {
    active: I18nKey.SETTINGS$API_KEY_STATUS_ACTIVE,
    pending: I18nKey.SETTINGS$API_KEY_STATUS_PENDING,
    expired: I18nKey.SETTINGS$API_KEY_STATUS_EXPIRED,
  }[status];

  return (
    <span
      className={`inline-block px-2 py-0.5 rounded text-xs font-medium ${STATUS_BADGE_CLASSES[status]}`}
    >
      {t(labelKey)}
    </span>
  );
}

interface ApiKeysTableProps {
  apiKeys: ApiKey[];
  isLoading: boolean;
  onDeleteKey: (key: ApiKey) => void;
}

function ApiKeyScopeBadge({
  orgId,
  orgLabel,
}: {
  orgId: string | null;
  orgLabel: string;
}) {
  const { t } = useTranslation();
  if (orgId === null) {
    // Unbound key -- usable against any org via X-Org-Id.
    return (
      <span
        className="inline-block px-2 py-0.5 rounded text-xs font-medium bg-blue-500/20 text-blue-300"
        title={t(I18nKey.SETTINGS$API_KEY_ORG_ALL_ORGS_DESCRIPTION)}
      >
        {t(I18nKey.SETTINGS$API_KEY_SCOPE_ALL_ORGS)}
      </span>
    );
  }
  // Bound key -- show the org's display name (or the fallback label
  // when the org is no longer in the user's membership list).
  return (
    <span
      className="inline-block px-2 py-0.5 rounded text-xs font-medium bg-tertiary text-gray-300"
      title={t(I18nKey.SETTINGS$API_KEY_SCOPE_BOUND_TITLE)}
    >
      {orgLabel}
    </span>
  );
}

function ApiKeysTable({ apiKeys, isLoading, onDeleteKey }: ApiKeysTableProps) {
  const { t } = useTranslation();
  const { data: organizationsData } = useOrganizations();

  // Map org id -> display label. Personal workspaces are rendered using
  // the same "Personal Workspace" string as the create modal so the
  // list, modal and header all read the same way.
  const orgLabelsById = useMemo(() => {
    const map = new Map<string, string>();
    for (const org of organizationsData?.organizations ?? []) {
      map.set(
        org.id,
        org.is_personal ? t(I18nKey.ORG$PERSONAL_WORKSPACE) : org.name,
      );
    }
    return map;
  }, [organizationsData, t]);

  const resolveOrgLabel = (orgId: string | null): string => {
    if (orgId === null) {
      // Unbound keys are rendered as their own badge; this shouldn't
      // be reached but keep a sensible fallback.
      return t(I18nKey.SETTINGS$API_KEY_SCOPE_ALL_ORGS);
    }
    return (
      orgLabelsById.get(orgId) ??
      // Fallback when the user can no longer see the org (e.g. they
      // left it). The id is short and unambiguous; the title carries
      // the full sentence.
      orgId.slice(0, 8)
    );
  };

  const formatDate = (dateString: string | null) => {
    if (!dateString) return "—";
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
            <th className="text-left p-3 text-sm font-medium">
              {t(I18nKey.SETTINGS$API_KEY_STATUS)}
            </th>
            <th className="text-left p-3 text-sm font-medium">
              {t(I18nKey.SETTINGS$API_KEY_SCOPE)}
            </th>
            <th className="text-right p-3 text-sm font-medium">
              {t(I18nKey.SETTINGS$ACTIONS)}
            </th>
          </tr>
        </thead>
        <tbody>
          {apiKeys.map((key) => {
            const status = getApiKeyStatus(key);
            const dimmed =
              status === "expired" || status === "pending" ? "opacity-60" : "";
            return (
              <tr key={key.id} className={`border-t border-tertiary ${dimmed}`}>
                <td
                  className="p-3 text-sm truncate max-w-[160px]"
                  title={key.name}
                >
                  {key.name}
                </td>
                <td className="p-3 text-sm">{formatDate(key.created_at)}</td>
                <td className="p-3 text-sm">{formatDate(key.last_used_at)}</td>
                <td className="p-3 text-sm">
                  <div className="flex flex-col gap-1">
                    <ApiKeyStatusBadge status={status} />
                    {key.not_before && (
                      <span
                        className="text-xs text-gray-400"
                        title={t(I18nKey.SETTINGS$API_KEY_NOT_BEFORE)}
                      >
                        {t(I18nKey.SETTINGS$API_KEY_NOT_BEFORE)}:{" "}
                        {formatDate(key.not_before)}
                      </span>
                    )}
                    {key.expires_at && (
                      <span
                        className="text-xs text-gray-400"
                        title={t(I18nKey.SETTINGS$API_KEY_EXPIRES_AT)}
                      >
                        {t(I18nKey.SETTINGS$API_KEY_EXPIRES_AT)}:{" "}
                        {formatDate(key.expires_at)}
                      </span>
                    )}
                  </div>
                </td>
                <td className="p-3 text-sm">
                  <ApiKeyScopeBadge
                    orgId={key.org_id}
                    orgLabel={resolveOrgLabel(key.org_id)}
                  />
                </td>
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
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function ApiKeysManager() {
  const { t } = useTranslation();
  const { data: apiKeys = [], isLoading, error } = useApiKeys();
  const {
    data: llmApiKey,
    isLoading: isLoadingLlmKey,
    isPaymentRequired,
  } = useLlmApiKey();
  const refreshLlmApiKey = useRefreshLlmApiKey();
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [deleteModalOpen, setDeleteModalOpen] = useState(false);
  const [keyToDelete, setKeyToDelete] = useState<ApiKey | null>(null);
  const [newlyCreatedKey, setNewlyCreatedKey] =
    useState<CreateApiKeyResponse | null>(null);
  const [showNewKeyModal, setShowNewKeyModal] = useState(false);

  // Display error toast if the query fails (but not for payment required)
  if (error && !isPaymentRequired) {
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
          isPaymentRequired={isPaymentRequired}
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
