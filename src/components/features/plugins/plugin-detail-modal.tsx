import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BrandButton } from "#/components/features/settings/brand-button";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { SkillIconBadge } from "#/components/features/skills/skill-icon-badge";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { modalTitleLgClassName } from "#/utils/modal-classes";
import { extensionModuleCardPillClassName } from "#/utils/extension-module-card-classes";
import type { PluginViewModel } from "./build-plugins-view-model";
import { PluginFilesSection } from "./plugin-files-section";

interface PluginDetailModalProps {
  plugin: PluginViewModel;
  isBusy?: boolean;
  isDisabled?: boolean;
  onToggle: (enabled: boolean) => void;
  onInstall: () => void;
  onUninstall: () => void;
  onRefresh: () => void;
  onClose: () => void;
  onStartConversation?: () => void;
}

export function PluginDetailModal({
  plugin,
  isBusy = false,
  isDisabled = false,
  onToggle,
  onInstall,
  onUninstall,
  onRefresh,
  onClose,
  onStartConversation,
}: PluginDetailModalProps) {
  const { t } = useTranslation("openhands");
  const actionsDisabled = isDisabled || isBusy;

  return (
    <ModalBackdrop onClose={onClose} aria-label={plugin.name}>
      <div
        data-testid="plugin-detail-modal"
        data-plugin-name={plugin.name}
        className="relative flex w-[640px] max-w-[90vw] max-h-[85vh] flex-col rounded-xl border border-[var(--oh-border)] bg-base-secondary"
      >
        <ModalCloseButton
          onClose={onClose}
          testId="plugin-detail-modal-close"
        />

        <header className="flex-shrink-0 px-6 pb-4 pt-6">
          <h2 className={cn("pr-6", modalTitleLgClassName)}>{plugin.name}</h2>
          {plugin.source ? (
            <p className="mt-1 break-all text-xs text-tertiary-alt">
              {plugin.source}
            </p>
          ) : null}
        </header>

        <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-6 custom-scrollbar">
          {plugin.description ? (
            <p className="text-sm leading-relaxed text-tertiary-light">
              {plugin.description}
            </p>
          ) : null}

          <div className="flex flex-wrap gap-2">
            {plugin.isLocal ? (
              <span className={extensionModuleCardPillClassName}>
                {t(I18nKey.SETTINGS$PLUGINS_FILTER_LOCAL)}
              </span>
            ) : null}
            {plugin.version ? (
              <span className={extensionModuleCardPillClassName}>
                {t(I18nKey.SETTINGS$SKILLS_VERSION, {
                  version: plugin.version,
                })}
              </span>
            ) : null}
            {plugin.ref ? (
              <span className={extensionModuleCardPillClassName}>
                {t(I18nKey.SETTINGS$PLUGINS_REF_LABEL)} {plugin.ref}
              </span>
            ) : null}
            {plugin.repoPath ? (
              <span className={extensionModuleCardPillClassName}>
                {t(I18nKey.SETTINGS$PLUGINS_REPO_PATH_LABEL)} {plugin.repoPath}
              </span>
            ) : null}
          </div>

          {plugin.installed ? (
            <SettingsSwitch
              testId={`plugin-modal-toggle-${plugin.name}`}
              isToggled={plugin.enabled}
              onToggle={onToggle}
              isDisabled={actionsDisabled}
              togglePosition="right"
            >
              {t(
                plugin.enabled
                  ? I18nKey.SETTINGS$SKILLS_ENABLED
                  : I18nKey.SETTINGS$SKILLS_DISABLED,
              )}
            </SettingsSwitch>
          ) : null}

          {plugin.skills?.length ? (
            <section className="flex min-w-0 flex-col gap-2">
              <h3 className="text-sm font-medium text-white">
                {t(I18nKey.SETTINGS$PLUGINS_SKILLS_IN_BUNDLE)}
              </h3>
              <ul className="flex min-w-0 flex-col gap-2">
                {plugin.skills.map((skill) => (
                  <li
                    key={skill.name}
                    data-testid={`plugin-bundled-skill-${skill.name}`}
                    className="flex items-start gap-3 rounded-lg border border-[var(--oh-border)] bg-[rgba(255,255,255,0.04)] px-3 py-2.5"
                  >
                    <SkillIconBadge skillName={skill.name} />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-semibold text-white">
                        {skill.name}
                      </p>
                      {skill.description ? (
                        <p className="mt-0.5 break-words text-xs leading-relaxed text-tertiary-light">
                          {skill.description}
                        </p>
                      ) : null}
                    </div>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}

          {plugin.path && plugin.files?.length ? (
            <PluginFilesSection basePath={plugin.path} files={plugin.files} />
          ) : null}
        </div>

        <footer className="flex flex-shrink-0 flex-wrap justify-end gap-2 px-6 pb-6 pt-4">
          {plugin.installed ? (
            <>
              <BrandButton
                type="button"
                variant="secondary"
                testId={`plugin-detail-refresh-${plugin.name}`}
                isDisabled={actionsDisabled}
                onClick={onRefresh}
              >
                {t(I18nKey.SETTINGS$PLUGINS_REFRESH)}
              </BrandButton>
              <BrandButton
                type="button"
                variant="secondary"
                testId={`plugin-detail-uninstall-${plugin.name}`}
                isDisabled={actionsDisabled}
                onClick={onUninstall}
              >
                {t(I18nKey.SETTINGS$PLUGINS_UNINSTALL)}
              </BrandButton>
            </>
          ) : plugin.isLocal ? null : (
            <BrandButton
              type="button"
              variant="secondary"
              testId={`plugin-detail-install-${plugin.name}`}
              isDisabled={actionsDisabled}
              onClick={onInstall}
            >
              {t(
                isBusy
                  ? I18nKey.SETTINGS$PLUGINS_INSTALLING
                  : I18nKey.SETTINGS$PLUGINS_INSTALL,
              )}
            </BrandButton>
          )}
          {plugin.source && onStartConversation ? (
            <BrandButton
              type="button"
              variant="secondary"
              testId={`plugin-detail-start-conversation-${plugin.name}`}
              onClick={onStartConversation}
            >
              {t(I18nKey.COMMON$START_CONVERSATION)}
            </BrandButton>
          ) : null}
          <BrandButton
            type="button"
            variant="secondary"
            testId="plugin-detail-modal-dismiss"
            onClick={onClose}
          >
            {t(I18nKey.BUTTON$CLOSE)}
          </BrandButton>
        </footer>
      </div>
    </ModalBackdrop>
  );
}
