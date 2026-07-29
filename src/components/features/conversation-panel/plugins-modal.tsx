import { useTranslation } from "react-i18next";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { BaseModalTitle } from "#/components/shared/modals/confirmation-modals/base-modal";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { useConversationPlugins } from "#/hooks/use-conversation-plugins";
import {
  getPluginDisplayName,
  getPluginSourceLabel,
  isLocalPluginSource,
  pluginReferenceKey,
} from "#/utils/plugin-display";

interface PluginsModalProps {
  onClose: () => void;
}

/**
 * Display-only view of the plugins loaded into the current conversation,
 * captured in client-side metadata at creation (explicitly attached plugins
 * plus the enabled installed plugins the SDK auto-loads). Mirrors
 * {@link SkillsModal}. The agent-server doesn't expose a live conversation's
 * loaded plugins, so this reads that client-side snapshot.
 */
export function PluginsModal({ onClose }: PluginsModalProps) {
  const { t } = useTranslation("openhands");
  const plugins = useConversationPlugins();

  return (
    <ModalBackdrop onClose={onClose}>
      <ModalBody
        width="lg"
        className="relative flex max-h-[80vh] flex-col items-start border border-[var(--oh-border)]"
        testID="plugins-modal"
      >
        <ModalCloseButton onClose={onClose} testId="close-plugins-modal" />
        <div className="flex w-full flex-col gap-2 pr-10">
          <BaseModalTitle title={t(I18nKey.PLUGINS_MODAL$TITLE)} />
          <Typography.Text className="text-sm text-[var(--oh-muted)]">
            {t(I18nKey.PLUGINS_MODAL$DESCRIPTION)}
          </Typography.Text>
        </div>

        <div className="w-full overflow-auto rounded-md border border-[var(--oh-border)] bg-surface-raised custom-scrollbar-always">
          {plugins.length === 0 ? (
            <div className="flex items-center justify-center p-6">
              <Typography.Text className="text-[var(--oh-muted)]">
                {t(I18nKey.PLUGINS_MODAL$EMPTY)}
              </Typography.Text>
            </div>
          ) : (
            <ul className="divide-y divide-[var(--oh-border)]">
              {plugins.map((plugin) => (
                <li
                  key={pluginReferenceKey(plugin)}
                  data-testid={`active-plugin-${getPluginDisplayName(plugin)}`}
                  className="flex flex-col gap-1 p-4"
                >
                  <Typography.Text className="font-semibold text-white">
                    {getPluginDisplayName(plugin)}
                  </Typography.Text>
                  <Typography.Text className="text-xs text-tertiary-alt">
                    {isLocalPluginSource(plugin)
                      ? t(I18nKey.PLUGINS_MODAL$SOURCE_LOCAL)
                      : getPluginSourceLabel(plugin)}
                  </Typography.Text>
                </li>
              ))}
            </ul>
          )}
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
