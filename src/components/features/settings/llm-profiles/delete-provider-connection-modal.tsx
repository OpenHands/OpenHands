import { useRef } from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LoadingSpinner } from "#/components/shared/loading-spinner";
import { ApiKeyModalBase } from "#/components/features/settings/api-key-modal-base";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import { useDeleteProviderConnection } from "#/hooks/mutation/use-delete-provider-connection";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { I18nKey } from "#/i18n/declaration";

interface DeleteProviderConnectionModalProps {
  connection: ProviderConnection | null;
  onClose: () => void;
}

export function DeleteProviderConnectionModal({
  connection,
  onClose,
}: DeleteProviderConnectionModalProps) {
  const { t } = useTranslation("openhands");
  const deleteConnection = useDeleteProviderConnection();
  const cancelButtonRef = useRef<HTMLButtonElement>(null);

  if (!connection) return null;

  const handleDelete = async () => {
    try {
      await deleteConnection.mutateAsync(connection.id);
      displaySuccessToast(
        t(I18nKey.SETTINGS$PROVIDER_CONNECTION_DELETED, {
          name: connection.display_name,
        }),
      );
      onClose();
    } catch (error) {
      // The agent-server returns 409 with a message naming the profiles that
      // still reference this connection; surface it verbatim.
      displayErrorToast(getApiErrorMessage(error, t(I18nKey.ERROR$GENERIC)));
    }
  };

  const handleClose = () => {
    if (!deleteConnection.isPending) onClose();
  };

  const footer = (
    <>
      <BrandButton
        ref={cancelButtonRef}
        type="button"
        variant="tertiary"
        onClick={handleClose}
        isDisabled={deleteConnection.isPending}
      >
        {t(I18nKey.BUTTON$CANCEL)}
      </BrandButton>
      <BrandButton
        testId="delete-provider-connection-confirm"
        type="button"
        variant="danger"
        onClick={handleDelete}
        isDisabled={deleteConnection.isPending}
        aria-busy={deleteConnection.isPending}
      >
        {deleteConnection.isPending ? (
          <>
            <LoadingSpinner size="small" />
            <span className="sr-only">{t(I18nKey.BUTTON$DELETE)}</span>
          </>
        ) : (
          t(I18nKey.BUTTON$DELETE)
        )}
      </BrandButton>
    </>
  );

  return (
    <ApiKeyModalBase
      isOpen
      title={t(I18nKey.SETTINGS$PROVIDER_CONNECTION_DELETE_TITLE)}
      footer={footer}
      onClose={handleClose}
      initialFocusRef={cancelButtonRef}
    >
      <p className="text-sm break-all">
        {t(I18nKey.SETTINGS$PROVIDER_CONNECTION_DELETE_CONFIRMATION, {
          name: connection.display_name,
        })}
      </p>
    </ApiKeyModalBase>
  );
}
