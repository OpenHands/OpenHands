import React from "react";
import { useTranslation } from "react-i18next";
import { FaKey, FaTrash } from "react-icons/fa";
import { useSSHKeys } from "#/hooks/query/use-ssh-keys";
import { useAddSSHKey } from "#/hooks/mutation/use-add-ssh-key";
import { useDeleteSSHKey } from "#/hooks/mutation/use-delete-ssh-key";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { SSHKey } from "#/api/ssh-keys-service";
import { I18nKey } from "#/i18n/declaration";
import { createPermissionGuard } from "#/utils/org/permission-guard";

export const clientLoader = createPermissionGuard("manage_secrets");

function SSHKeysSettingsScreen() {
  const { t } = useTranslation();

  const { data: sshKeys, isLoading } = useSSHKeys();
  const { mutate: addSSHKey, isPending: isAdding } = useAddSSHKey();
  const { mutate: deleteSSHKey } = useDeleteSSHKey();

  const [showAddForm, setShowAddForm] = React.useState(false);
  const [newKey, setNewKey] = React.useState("");
  const [newLabel, setNewLabel] = React.useState("");
  const [error, setError] = React.useState<string | null>(null);
  const [selectedKeyId, setSelectedKeyId] = React.useState<string | null>(null);
  const [showDeleteConfirm, setShowDeleteConfirm] = React.useState(false);

  const handleAddKey = () => {
    setError(null);
    if (!newKey.trim()) {
      setError(t(I18nKey.SSH_KEYS$INVALID_KEY));
      return;
    }

    addSSHKey(
      { key: newKey.trim(), label: newLabel.trim() || null },
      {
        onSuccess: () => {
          setNewKey("");
          setNewLabel("");
          setShowAddForm(false);
        },
        onError: (err: Error) => {
          if (err.message.includes("already exists")) {
            setError(t(I18nKey.SSH_KEYS$KEY_EXISTS));
          } else if (err.message.includes("Invalid")) {
            setError(t(I18nKey.SSH_KEYS$INVALID_KEY));
          } else {
            setError(err.message);
          }
        },
      },
    );
  };

  const handleDeleteKey = () => {
    if (selectedKeyId) {
      deleteSSHKey(selectedKeyId, {
        onSettled: () => {
          setShowDeleteConfirm(false);
          setSelectedKeyId(null);
        },
      });
    }
  };

  const truncateKey = (key: string) => {
    if (key.length <= 60) return key;
    return `${key.slice(0, 30)}...${key.slice(-20)}`;
  };

  return (
    <div data-testid="ssh-keys-settings-screen" className="flex flex-col gap-5">
      <div className="text-sm text-content-secondary">
        {t(I18nKey.SSH_KEYS$DESCRIPTION)}
      </div>

      {!showAddForm && (
        <BrandButton
          testId="add-ssh-key-button"
          type="button"
          variant="primary"
          onClick={() => setShowAddForm(true)}
          isDisabled={isLoading}
        >
          {t(I18nKey.SSH_KEYS$ADD_KEY)}
        </BrandButton>
      )}

      {showAddForm && (
        <div className="border border-tertiary rounded-md p-4 flex flex-col gap-3">
          <div>
            <label
              htmlFor="ssh-key-input"
              className="block text-sm font-medium mb-1"
            >
              {t(I18nKey.SSH_KEYS$KEY_LABEL)}
            </label>
            <textarea
              id="ssh-key-input"
              value={newKey}
              onChange={(e) => setNewKey(e.target.value)}
              placeholder={t(I18nKey.SSH_KEYS$KEY_PLACEHOLDER)}
              className="w-full h-24 p-2 border border-tertiary rounded-md bg-base-secondary text-sm font-mono"
              disabled={isAdding}
            />
          </div>
          <div>
            <label
              htmlFor="ssh-key-label"
              className="block text-sm font-medium mb-1"
            >
              {t(I18nKey.SSH_KEYS$LABEL_LABEL)}
            </label>
            <input
              id="ssh-key-label"
              type="text"
              value={newLabel}
              onChange={(e) => setNewLabel(e.target.value)}
              placeholder={t(I18nKey.SSH_KEYS$LABEL_PLACEHOLDER)}
              className="w-full p-2 border border-tertiary rounded-md bg-base-secondary text-sm"
              disabled={isAdding}
            />
          </div>
          {error && <div className="text-red-500 text-sm">{error}</div>}
          <div className="flex gap-2">
            <BrandButton
              testId="save-ssh-key-button"
              type="button"
              variant="primary"
              onClick={handleAddKey}
              isDisabled={isAdding || !newKey.trim()}
            >
              {isAdding
                ? t(I18nKey.SSH_KEYS$ADDING)
                : t(I18nKey.SSH_KEYS$ADD_KEY)}
            </BrandButton>
            <BrandButton
              testId="cancel-ssh-key-button"
              type="button"
              variant="secondary"
              onClick={() => {
                setShowAddForm(false);
                setNewKey("");
                setNewLabel("");
                setError(null);
              }}
              isDisabled={isAdding}
            >
              {t(I18nKey.COMMON$CANCEL)}
            </BrandButton>
          </div>
        </div>
      )}

      {isLoading && (
        <div className="text-content-secondary text-sm">
          {t(I18nKey.VSCODE$LOADING)}
        </div>
      )}

      {sshKeys && sshKeys.length > 0 && (
        <div className="border border-tertiary rounded-md overflow-hidden">
          <table className="w-full min-w-full table-fixed">
            <thead className="bg-base-tertiary">
              <tr>
                <th
                  className="w-16 text-center p-3 text-sm font-medium"
                  aria-label="Key icon"
                />
                <th className="w-1/4 text-left p-3 text-sm font-medium">
                  {t(I18nKey.SSH_KEYS$LABEL_LABEL)}
                </th>
                <th className="w-1/2 text-left p-3 text-sm font-medium">
                  {t(I18nKey.SSH_KEYS$PUBLIC_KEY_HEADER)}
                </th>
                <th className="w-24 text-right p-3 text-sm font-medium">
                  {t(I18nKey.SETTINGS$ACTIONS)}
                </th>
              </tr>
            </thead>
            <tbody>
              {sshKeys.map((sshKey: SSHKey) => (
                <tr key={sshKey.id} className="border-t border-tertiary">
                  <td className="p-3 text-center">
                    <FaKey className="w-4 h-4 text-content-secondary inline" />
                  </td>
                  <td className="p-3 text-sm">
                    {sshKey.label || (
                      <span className="text-content-tertiary italic">
                        {t(I18nKey.SSH_KEYS$NO_LABEL)}
                      </span>
                    )}
                  </td>
                  <td className="p-3 text-sm font-mono text-content-secondary truncate">
                    {truncateKey(sshKey.key)}
                  </td>
                  <td className="p-3 text-right">
                    <button
                      type="button"
                      onClick={() => {
                        setSelectedKeyId(sshKey.id);
                        setShowDeleteConfirm(true);
                      }}
                      className="text-red-500 hover:text-red-400 p-1"
                      aria-label={t(I18nKey.BUTTON$DELETE)}
                    >
                      <FaTrash className="w-4 h-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {sshKeys && sshKeys.length === 0 && !showAddForm && (
        <div className="text-content-secondary text-sm text-center py-8 border border-tertiary rounded-md">
          {t(I18nKey.SSH_KEYS$NO_KEYS)}
        </div>
      )}

      {showDeleteConfirm && (
        <ConfirmationModal
          text={t(I18nKey.SSH_KEYS$CONFIRM_DELETE)}
          onConfirm={handleDeleteKey}
          onCancel={() => {
            setShowDeleteConfirm(false);
            setSelectedKeyId(null);
          }}
        />
      )}
    </div>
  );
}

export default SSHKeysSettingsScreen;
