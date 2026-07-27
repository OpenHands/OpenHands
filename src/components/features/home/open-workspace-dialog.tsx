import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BaseModalTitle } from "#/components/shared/modals/confirmation-modals/base-modal";
import { I18nKey } from "#/i18n/declaration";
import { LocalWorkspace } from "#/types/workspace";
import { useUserProviders } from "#/hooks/use-user-providers";
import { WorkspaceSelectionForm } from "./workspace-selection-form";

interface OpenWorkspaceDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (workspace: LocalWorkspace) => void;
}

export function OpenWorkspaceDialog({
  isOpen,
  onClose,
  onConfirm,
}: OpenWorkspaceDialogProps) {
  const { t } = useTranslation("openhands");
  const { isLoadingSettings } = useUserProviders();

  if (!isOpen) return null;

  return (
    <ModalBackdrop onClose={onClose}>
      <ModalBody
        width="sm"
        className="relative items-start border border-[var(--oh-border)] !gap-4"
      >
        <ModalCloseButton
          onClose={onClose}
          testId="close-open-workspace-dialog"
        />
        <div className="w-full pr-6">
          <BaseModalTitle title={t(I18nKey.HOME$OPEN_WORKSPACE)} />
        </div>

        <div className="w-full" data-testid="open-workspace-dialog-body">
          <WorkspaceSelectionForm
            isLoadingSettings={isLoadingSettings}
            onConfirm={(workspace) => {
              onConfirm(workspace);
              onClose();
            }}
          />
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
