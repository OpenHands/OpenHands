import React from "react";
import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsInput } from "#/components/features/settings/settings-input";
import { useInstallPlugin } from "#/hooks/mutation/use-install-plugin";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { modalTitleLgClassName } from "#/utils/modal-classes";

interface AddPluginModalProps {
  onClose: () => void;
  onInstalled?: () => void;
}

export function AddPluginModal({ onClose, onInstalled }: AddPluginModalProps) {
  const { t } = useTranslation("openhands");
  const installPlugin = useInstallPlugin();

  const [source, setSource] = React.useState("");
  const [ref, setRef] = React.useState("");
  const [repoPath, setRepoPath] = React.useState("");

  const trimmedSource = source.trim();
  const canSubmit = trimmedSource.length > 0 && !installPlugin.isPending;

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!canSubmit) return;

    installPlugin.mutate(
      {
        source: trimmedSource,
        ref: ref.trim() || null,
        repo_path: repoPath.trim() || null,
      },
      {
        onSuccess: () => {
          onInstalled?.();
          onClose();
        },
      },
    );
  };

  return (
    <ModalBackdrop
      onClose={onClose}
      aria-label={t(I18nKey.SETTINGS$PLUGINS_ADD_MODAL_TITLE)}
    >
      <form
        onSubmit={handleSubmit}
        data-testid="add-plugin-modal"
        className="relative flex w-[520px] max-w-[90vw] max-h-[85vh] flex-col rounded-xl border border-[var(--oh-border)] bg-base-secondary"
      >
        <ModalCloseButton onClose={onClose} testId="add-plugin-modal-close" />

        <header className="flex-shrink-0 px-6 pb-4 pt-6">
          <h2 className={cn("pr-6", modalTitleLgClassName)}>
            {t(I18nKey.SETTINGS$PLUGINS_ADD_MODAL_TITLE)}
          </h2>
          <p className="mt-4 text-sm text-tertiary-light">
            {t(I18nKey.SETTINGS$PLUGINS_ADD_MODAL_INTRO)}
          </p>
        </header>

        <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-6 custom-scrollbar">
          <SettingsInput
            testId="add-plugin-source-input"
            label={t(I18nKey.SETTINGS$PLUGINS_SOURCE_LABEL)}
            type="text"
            value={source}
            onChange={setSource}
            placeholder={t(I18nKey.SETTINGS$PLUGINS_SOURCE_PLACEHOLDER)}
            showRequiredTag
          />
          <SettingsInput
            testId="add-plugin-ref-input"
            label={t(I18nKey.SETTINGS$PLUGINS_REF_LABEL)}
            type="text"
            value={ref}
            onChange={setRef}
            showOptionalTag
          />
          <SettingsInput
            testId="add-plugin-repo-path-input"
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
            testId="add-plugin-modal-dismiss"
          >
            {t(I18nKey.BUTTON$CLOSE)}
          </BrandButton>
          <BrandButton
            type="submit"
            variant="primary"
            testId="add-plugin-submit"
            isDisabled={!canSubmit}
          >
            {t(
              installPlugin.isPending
                ? I18nKey.SETTINGS$PLUGINS_INSTALLING
                : I18nKey.SETTINGS$PLUGINS_INSTALL,
            )}
          </BrandButton>
        </footer>
      </form>
    </ModalBackdrop>
  );
}
