import React from "react";
import { AxiosError } from "axios";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LlmSettingsInputsSkeleton } from "#/components/features/settings/llm-settings/llm-settings-inputs-skeleton";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { usePermission } from "#/hooks/organizations/use-permissions";
import {
  useAgentSettingsSchema,
  useConversationSettingsSchema,
} from "#/hooks/query/use-agent-settings-schema";
import { useConfig } from "#/hooks/query/use-config";
import { useMe } from "#/hooks/query/use-me";
import { useSettings } from "#/hooks/query/use-settings";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";
import { Settings, SettingsSchema, SettingsScope } from "#/types/settings";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import {
  buildInitialSettingsFormValues,
  buildSdkSettingsPayloadForView,
  getVisibleSettingsSections,
  hasAdvancedSettings,
  hasMinorSettings,
  inferInitialView,
  SettingsDirtyState,
  SettingsFormValues,
  type SettingsValueSource,
  type SettingsView,
} from "#/utils/sdk-settings-schema";
import { SchemaField } from "./schema-field";
import { ViewToggle } from "./view-toggle";

const EMPTY_EXCLUDE_KEYS = new Set<string>();

const VIEW_ORDER: Record<SettingsView, number> = {
  basic: 0,
  advanced: 1,
  all: 2,
};

const getLessDetailedView = (
  currentView: SettingsView,
  nextView: SettingsView,
): SettingsView =>
  VIEW_ORDER[nextView] < VIEW_ORDER[currentView] ? nextView : currentView;

export interface SdkSectionHeaderProps {
  values: SettingsFormValues;
  isDisabled: boolean;
  view: SettingsView;
  onChange: (key: string, value: string | boolean) => void;
}

/**
 * A generic SDK-schema–driven settings page that renders fields
 * from one or more schema sections.
 *
 * @param sectionKeys - which schema section(s) this page owns (e.g. ["condenser"])
 * @param excludeKeys - field keys to skip (rendered elsewhere by the caller)
 * @param header      - optional render prop receiving shared state to render above fields
 * @param testId      - data-testid for the page wrapper
 */
export function SdkSectionPage({
  sectionKeys,
  excludeKeys = EMPTY_EXCLUDE_KEYS,
  scope = "personal",
  settingsSource = "agent_settings",
  header,
  extraDirty = false,
  buildPayload,
  onSaveSuccess,
  getInitialView,
  testId = "sdk-section-settings-screen",
}: {
  sectionKeys: string[];
  excludeKeys?: Set<string>;
  scope?: SettingsScope;
  settingsSource?: SettingsValueSource;

  header?: (props: SdkSectionHeaderProps) => React.ReactNode;
  extraDirty?: boolean;
  buildPayload?: (
    payload: ReturnType<typeof buildSdkSettingsPayloadForView>,
    context: {
      values: SettingsFormValues;
      dirty: SettingsDirtyState;
      view: SettingsView;
    },
  ) => Record<string, unknown>;
  onSaveSuccess?: () => void;
  getInitialView?: (
    settings: Settings,
    filteredSchema: SettingsSchema,
  ) => SettingsView;
  testId?: string;
}) {
  const { t } = useTranslation();
  const { mutate: saveSettings, isPending } = useSaveSettings(scope);
  const { data: settings, isLoading, isFetching } = useSettings(scope);
  const agentSchemaQuery = useAgentSettingsSchema(
    settings?.agent_settings_schema,
  );
  const conversationSchemaQuery = useConversationSettingsSchema(
    settings?.conversation_settings_schema,
  );
  const schema =
    settingsSource === "conversation_settings"
      ? conversationSchemaQuery.data
      : agentSchemaQuery.data;
  const isSchemaLoading =
    settingsSource === "conversation_settings"
      ? conversationSchemaQuery.isLoading
      : agentSchemaQuery.isLoading;
  const { data: config } = useConfig();
  const { data: me } = useMe();
  const { hasPermission } = usePermission(me?.role ?? "member");

  const isOssMode = config?.app_mode === "oss";
  const isReadOnly =
    scope === "org" && !isOssMode ? !hasPermission("edit_llm_settings") : false;

  const [view, setView] = React.useState<SettingsView>("basic");
  const [values, setValues] = React.useState<SettingsFormValues>({});
  const [dirty, setDirty] = React.useState<SettingsDirtyState>({});
  const hasHydratedViewRef = React.useRef(false);

  const sectionKeysSignature = React.useMemo(
    () => JSON.stringify(sectionKeys),
    [sectionKeys],
  );
  const stableSectionKeys = React.useMemo(
    () => JSON.parse(sectionKeysSignature) as string[],
    [sectionKeysSignature],
  );

  // Build a filtered schema containing only the requested sections
  const filteredSchema = React.useMemo(() => {
    if (!schema) return null;
    const sectionSet = new Set(stableSectionKeys);
    return {
      ...schema,
      sections: schema.sections.filter((s) => sectionSet.has(s.key)),
    };
  }, [schema, stableSectionKeys]);

  const showAdvanced = hasAdvancedSettings(filteredSchema);
  const showAll = hasMinorSettings(filteredSchema);

  const initialValues = React.useMemo(() => {
    if (!settings || !filteredSchema) return null;
    return buildInitialSettingsFormValues(
      settings,
      filteredSchema,
      settingsSource,
    );
  }, [settings, filteredSchema, settingsSource]);

  const initialView = React.useMemo(() => {
    if (!settings || !filteredSchema) return null;
    return getInitialView
      ? getInitialView(settings, filteredSchema)
      : inferInitialView(settings, filteredSchema, settingsSource);
  }, [settings, filteredSchema, getInitialView, settingsSource]);

  React.useEffect(() => {
    hasHydratedViewRef.current = false;
    setView("basic");
    setValues({});
    setDirty({});
  }, [scope, settingsSource, sectionKeysSignature]);

  React.useEffect(() => {
    if (!initialValues || !initialView) return;

    setValues(initialValues);
    setDirty({});
    setView((currentView) => {
      if (!hasHydratedViewRef.current) {
        hasHydratedViewRef.current = true;
        return initialView;
      }

      return getLessDetailedView(currentView, initialView);
    });
  }, [initialValues, initialView]);

  const visibleSections = React.useMemo(() => {
    if (!filteredSchema) return [];
    return getVisibleSettingsSections(
      filteredSchema,
      values,
      view,
      excludeKeys,
    );
  }, [filteredSchema, values, view, excludeKeys]);

  const handleFieldChange = React.useCallback(
    (fieldKey: string, nextValue: string | boolean) => {
      setValues((prev) => ({ ...prev, [fieldKey]: nextValue }));
      setDirty((prev) => ({ ...prev, [fieldKey]: true }));
    },
    [],
  );

  const handleError = React.useCallback(
    (error: AxiosError) => {
      const msg = retrieveAxiosErrorMessage(error);
      displayErrorToast(msg || t(I18nKey.ERROR$GENERIC));
    },
    [t],
  );

  const handleSave = () => {
    if (!filteredSchema || isReadOnly) return;

    let payload: Record<string, unknown>;
    try {
      const basePayload = buildSdkSettingsPayloadForView(
        filteredSchema,
        values,
        dirty,
        view,
      );
      const defaultPayload =
        settingsSource === "conversation_settings"
          ? { conversation_settings: basePayload }
          : { agent_settings: basePayload };
      payload = buildPayload
        ? buildPayload(basePayload, { values, dirty, view })
        : defaultPayload;
    } catch (error) {
      displayErrorToast(
        error instanceof Error ? error.message : t(I18nKey.ERROR$GENERIC),
      );
      return;
    }

    if (Object.keys(payload).length === 0) return;

    saveSettings(payload, {
      onError: handleError,
      onSuccess: () => {
        displaySuccessToast(t(I18nKey.SETTINGS$SAVED_WARNING));
        setDirty({});
        onSaveSuccess?.();
      },
    });
  };

  if (isLoading || isFetching || isSchemaLoading) {
    return <LlmSettingsInputsSkeleton />;
  }

  if (!filteredSchema || filteredSchema.sections.length === 0) {
    return (
      <Typography.Paragraph className="text-tertiary-alt">
        {t(I18nKey.SETTINGS$SDK_SCHEMA_UNAVAILABLE)}
      </Typography.Paragraph>
    );
  }

  if (Object.keys(values).length === 0) return <LlmSettingsInputsSkeleton />;

  return (
    <div data-testid={testId} className="h-full relative">
      <ViewToggle
        view={view}
        setView={setView}
        showAdvanced={showAdvanced}
        showAll={showAll}
        isDisabled={isReadOnly}
      />

      <div className="flex flex-col gap-8 pb-20">
        {header?.({
          values,
          isDisabled: isReadOnly,
          view,
          onChange: handleFieldChange,
        })}

        {visibleSections.map((section) => (
          <section key={section.key} className="flex flex-col gap-4">
            <div className="grid gap-4 xl:grid-cols-2">
              {section.fields.map((field) => (
                <SchemaField
                  key={field.key}
                  field={field}
                  value={values[field.key]}
                  isDisabled={isReadOnly}
                  onChange={(nextValue) =>
                    handleFieldChange(field.key, nextValue)
                  }
                />
              ))}
            </div>
          </section>
        ))}
      </div>

      {!isReadOnly ? (
        <div className="sticky bottom-0 bg-base py-4">
          <BrandButton
            testId="save-button"
            type="button"
            variant="primary"
            isDisabled={
              isPending || (Object.keys(dirty).length === 0 && !extraDirty)
            }
            onClick={handleSave}
          >
            {isPending
              ? t(I18nKey.SETTINGS$SAVING)
              : t(I18nKey.SETTINGS$SAVE_CHANGES)}
          </BrandButton>
        </div>
      ) : null}
    </div>
  );
}
