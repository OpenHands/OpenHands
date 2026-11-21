import { EnterpriseDocsCallout } from "#/components/shared/enterprise-docs-callout";
import { useTranslation } from "react-i18next";
import {
  SdkSectionPage,
  type SettingsSourceConfig,
} from "#/components/features/settings/sdk-settings/sdk-section-page";
import { useConversationWorkspace } from "#/hooks/query/use-conversation-workspace";
import { useSettings } from "#/hooks/query/use-settings";
import { I18nKey } from "#/i18n/declaration";

const CONTAINER_FIELD_KEYS = new Set([
  "container_memory",
  "container_cpus",
  "container_browser",
]);

function RuntimeSettingsScreen() {
  const { t } = useTranslation();
  const { data: settings } = useSettings();
  const { dockerSelectable, serverInfoLoading } = useConversationWorkspace();
  const dockerUnavailable = !serverInfoLoading && !dockerSelectable;
  const savedDocker =
    settings?.conversation_settings?.default_runtime === "docker";

  const source: SettingsSourceConfig = {
    settingsSource: "conversation_settings",
    sectionKeys: ["runtime"],
  };
  if (dockerUnavailable) {
    source.excludeKeys = CONTAINER_FIELD_KEYS;
    // A saved Docker default stays selectable so the field shows what is saved.
    if (!savedDocker) source.excludeChoices = { default_runtime: ["docker"] };
  }

  return (
    <div className="flex flex-col gap-8">
      <SdkSectionPage
        settingsSources={[source]}
        header={
          dockerUnavailable && savedDocker
            ? () => (
                <p
                  data-testid="runtime-docker-unavailable"
                  className="text-sm text-muted"
                >
                  {t(I18nKey.SETTINGS$RUNTIME_DOCKER_UNAVAILABLE)}
                </p>
              )
            : undefined
        }
        testId="runtime-settings-screen"
      />
      <EnterpriseDocsCallout
        ctaId="custom_sandboxes"
        surface="runtime_settings"
      />
    </div>
  );
}

export default RuntimeSettingsScreen;
