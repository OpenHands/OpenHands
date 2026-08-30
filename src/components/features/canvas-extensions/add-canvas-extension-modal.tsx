import React from "react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { useInstallCanvasExtension } from "#/hooks/mutation/use-manage-canvas-extensions";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { modalTitleLgClassName } from "#/utils/modal-classes";

interface AddCanvasExtensionModalProps {
  onClose: () => void;
}

export function AddCanvasExtensionModal({
  onClose,
}: AddCanvasExtensionModalProps) {
  const { t } = useTranslation("openhands");
  const install = useInstallCanvasExtension();
  const [source, setSource] = React.useState("");
  const [ref, setRef] = React.useState("");
  const [repoPath, setRepoPath] = React.useState("");

  const trimmedSource = source.trim();
  const canSubmit = trimmedSource.length > 0 && !install.isPending;

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!canSubmit) return;
    install.mutate(
      {
        source: trimmedSource,
        ref: ref.trim() || null,
        repo_path: repoPath.trim() || null,
      },
      { onSuccess: onClose },
    );
  };

  return (
    <ModalBackdrop
      onClose={onClose}
      aria-label={t(I18nKey.SETTINGS$CANVAS_EXTENSIONS_ADD_BUTTON)}
    >
      <form
        onSubmit={handleSubmit}
        data-testid="add-canvas-extension-modal"
        className="relative flex w-[520px] max-w-[90vw] max-h-[85vh] flex-col rounded-xl border border-[var(--oh-border)] bg-base-secondary"
      >
        <ModalCloseButton
          onClose={onClose}
          testId="add-canvas-extension-modal-close"
        />
        <header className="flex-shrink-0 px-6 pb-4 pt-6">
          <h2 className={cn("pr-6", modalTitleLgClassName)}>
            {t(I18nKey.SETTINGS$CANVAS_EXTENSIONS_ADD_BUTTON)}
          </h2>
          <p className="mt-4 text-sm text-tertiary-light">
            {t(I18nKey.SETTINGS$CANVAS_EXTENSIONS_ADD_MODAL_INTRO)}
          </p>
        </header>
        <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-6 custom-scrollbar">
          <SettingsInput
            testId="add-canvas-extension-source-input"
            label={t(I18nKey.SETTINGS$PLUGINS_SOURCE_LABEL)}
            type="text"
            value={source}
            onChange={setSource}
            placeholder={t(I18nKey.SETTINGS$PLUGINS_SOURCE_PLACEHOLDER)}
            showRequiredTag
          />
          <SettingsInput
            testId="add-canvas-extension-ref-input"
            label={t(I18nKey.SETTINGS$PLUGINS_REF_LABEL)}
            type="text"
            value={ref}
            onChange={setRef}
            showOptionalTag
          />
          <SettingsInput
            testId="add-canvas-extension-repo-path-input"
            label={t(I18nKey.SETTINGS$PLUGINS_REPO_PATH_LABEL)}
            type="text"
            value={repoPath}
            onChange={setRepoPath}
            showOptionalTag
          />
        </div>
        <footer className="flex flex-shrink-0 justify-end gap-2 px-6 pb-6 pt-4">
          <BrandButton
            type="button"
            variant="secondary"
            onClick={onClose}
            testId="add-canvas-extension-dismiss"
          >
            {t(I18nKey.BUTTON$CLOSE)}
          </BrandButton>
          <BrandButton
            type="submit"
            variant="primary"
            testId="add-canvas-extension-submit"
            isDisabled={!canSubmit}
          >
            {t(
              install.isPending
                ? I18nKey.SETTINGS$PLUGINS_INSTALLING
                : I18nKey.SETTINGS$PLUGINS_INSTALL,
            )}
          </BrandButton>
        </footer>
      </form>
    </ModalBackdrop>
  );
}
