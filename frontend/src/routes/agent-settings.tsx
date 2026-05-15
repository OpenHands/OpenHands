import React from "react";
import { AxiosError } from "axios";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LlmSettingsInputsSkeleton } from "#/components/features/settings/llm-settings/llm-settings-inputs-skeleton";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { useAgentSettingsSchema } from "#/hooks/query/use-agent-settings-schema";
import { useSettings } from "#/hooks/query/use-settings";
import { I18nKey } from "#/i18n/declaration";
import { SettingsFieldSchema } from "#/types/settings";
import { Typography } from "#/ui/typography";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { createPermissionGuard } from "#/utils/org/permission-guard";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import {
  resolveSchemaFieldDescription,
  resolveSchemaFieldLabel,
} from "#/utils/sdk-settings-field-metadata";

const ENABLE_SUB_AGENTS_FIELD_KEY = "enable_sub_agents";

function findEnableSubAgentsField(
  fields: SettingsFieldSchema[] | undefined,
): SettingsFieldSchema | undefined {
  return fields?.find((field) => field.key === ENABLE_SUB_AGENTS_FIELD_KEY);
}

function getEnableSubAgentsValue(
  settingsValue: unknown,
  field: SettingsFieldSchema | undefined,
) {
  if (typeof settingsValue === "boolean") {
    return settingsValue;
  }

  return field?.default === true;
}

export const clientLoader = createPermissionGuard("view_llm_settings");

export default function AgentSettingsScreen() {
  const { t } = useTranslation();
  const { mutate: saveSettings, isPending } = useSaveSettings();
  const { data: settings, isLoading: isSettingsLoading } = useSettings();
  const { data: schema, isLoading: isSchemaLoading } = useAgentSettingsSchema(
    settings?.agent_settings_schema,
  );

  const fields = React.useMemo(
    () => schema?.sections.flatMap((section) => section.fields),
    [schema],
  );
  const field = findEnableSubAgentsField(fields);
  const initialIsEnabled = React.useMemo(
    () =>
      getEnableSubAgentsValue(
        settings?.agent_settings?.[ENABLE_SUB_AGENTS_FIELD_KEY],
        field,
      ),
    [field, settings?.agent_settings],
  );
  const [isEnabled, setIsEnabled] = React.useState(initialIsEnabled);

  React.useEffect(() => {
    setIsEnabled(initialIsEnabled);
  }, [initialIsEnabled]);

  const isDirty = isEnabled !== initialIsEnabled;

  const handleError = React.useCallback(
    (error: AxiosError) => {
      const msg = retrieveAxiosErrorMessage(error);
      displayErrorToast(msg || t(I18nKey.ERROR$GENERIC));
    },
    [t],
  );

  const handleSave = () => {
    if (!isDirty) return;

    saveSettings(
      {
        agent_settings_diff: {
          enable_sub_agents: isEnabled,
        },
      },
      {
        onError: handleError,
        onSuccess: () => {
          displaySuccessToast(t(I18nKey.SETTINGS$SAVED_WARNING));
        },
      },
    );
  };

  if (isSettingsLoading || isSchemaLoading) {
    return <LlmSettingsInputsSkeleton />;
  }

  if (!field) {
    return (
      <Typography.Paragraph className="text-tertiary-alt">
        {t(I18nKey.SETTINGS$SDK_SCHEMA_UNAVAILABLE)}
      </Typography.Paragraph>
    );
  }

  const label = resolveSchemaFieldLabel(t, field.key, field.label);
  const description = resolveSchemaFieldDescription(
    t,
    field.key,
    field.description,
  );

  return (
    <div data-testid="agent-settings-screen" className="h-full relative">
      <div className="flex flex-col gap-8 pb-20">
        <section className="grid gap-4 xl:grid-cols-2">
          <div className="flex flex-col gap-1.5">
            <SettingsSwitch
              testId="agent-settings-enable-sub-agents"
              isToggled={isEnabled}
              onToggle={setIsEnabled}
            >
              {label}
            </SettingsSwitch>
            {description ? (
              <Typography.Paragraph className="text-tertiary-alt text-xs leading-5">
                {description}
              </Typography.Paragraph>
            ) : null}
          </div>
        </section>
      </div>

      <div className="sticky bottom-0 bg-base py-4">
        <BrandButton
          testId="save-button"
          type="button"
          variant="primary"
          isDisabled={isPending || !isDirty}
          onClick={handleSave}
        >
          {isPending
            ? t(I18nKey.SETTINGS$SAVING)
            : t(I18nKey.SETTINGS$SAVE_CHANGES)}
        </BrandButton>
      </div>
    </div>
  );
}
