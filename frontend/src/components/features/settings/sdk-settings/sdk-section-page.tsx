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

const getMoreDetailedView = (
  currentView: SettingsView,
  nextView: SettingsView,
): SettingsView =>
  VIEW_ORDER[nextView] > VIEW_ORDER[currentView] ? nextView : currentView;

const normalizeView = (
  view: SettingsView,
  {
    showAdvanced,
    showAll,
  }: {
    showAdvanced: boolean;
    showAll: boolean;
  },
): SettingsView => {
  if (view === "all") {
    if (showAll) {
      return "all";
    }

    return showAdvanced ? "advanced" : "basic";
  }

  if (view === "advanced") {
    if (showAdvanced) {
      return "advanced";
    }

    return showAll ? "all" : "basic";
  }

  return "basic";
};

const PAYLOAD_DIFF_KEY: Record<SettingsValueSource, string> = {
  agent_settings: "agent_settings_diff",
  conversation_settings: "conversation_settings_diff",
};

export interface SettingsSourceConfig {
  /** Which schema/values bucket on `settings` this source pulls from. */
  settingsSource: SettingsValueSource;
  /** Section keys (e.g. ["llm"]) within that schema to render. */
  sectionKeys: string[];
  /** Field keys to skip (rendered elsewhere by the caller). */
  excludeKeys?: Set<string>;
}

export interface SdkSectionHeaderProps {
  values: SettingsFormValues;
  isDisabled: boolean;
  view: SettingsView;
  onChange: (key: string, value: string | boolean) => void;
}

interface ResolvedSource extends SettingsSourceConfig {
  filteredSchema: SettingsSchema | null;
}

/**
 * A generic SDK-schema–driven settings page that renders fields from one or
 * more schema sections.
 *
 * The `settingsSources` array specifies which schema(s)/section(s) the page
 * owns. The page tracks values/dirty state per source, renders sections from
 * each source in order (filtered by the schema's `prominence` field for the
 * selected view), and emits a combined save payload like
 * `{ conversation_settings_diff: {...}, agent_settings_diff: {...} }` —
 * including only the keys for sources that actually have dirty changes.
 *
 * @param settingsSources  one or more schemas to render fields from
 * @param header           render prop above the fields (receives unified state)
 * @param buildPayload     customize the save payload before submission
 * @param testId           data-testid on the page wrapper
 */
export function SdkSectionPage({
  settingsSources,
  scope = "personal",
  header,
  extraDirty = false,
  buildPayload,
  onSaveSuccess,
  getInitialView,
  forceShowAdvancedView = false,
  allowAllView = true,
  trailingActions,
  testId = "sdk-section-settings-screen",
}: {
  settingsSources: SettingsSourceConfig[];
  scope?: SettingsScope;

  header?: (props: SdkSectionHeaderProps) => React.ReactNode;
  extraDirty?: boolean;
  /**
   * Customize the save payload. Receives the wrapped default payload (e.g.
   * `{ agent_settings_diff: { llm: { model: "gpt-4" } } }`) plus the unified
   * form context. Return the payload to actually send.
   */
  buildPayload?: (
    defaultPayload: Record<string, unknown>,
    context: {
      values: SettingsFormValues;
      dirty: SettingsDirtyState;
      view: SettingsView;
    },
  ) => Record<string, unknown>;
  onSaveSuccess?: () => void;
  /**
   * Override the initial view per source. Called once per source on
   * hydration; the most-detailed result wins across sources.
   */
  getInitialView?: (
    settings: Settings,
    filteredSchema: SettingsSchema,
  ) => SettingsView;
  // Extra buttons slotted into the Basic/Advanced/All control strip,
  // after the view toggles. Used by the LLM page to drop a Profiles
  // navigation button into the same row.
  trailingActions?: React.ReactNode;
  forceShowAdvancedView?: boolean;
  allowAllView?: boolean;
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
  const { data: config } = useConfig();
  const { data: me } = useMe();
  const { hasPermission } = usePermission(me?.role ?? "member");

  const isOssMode = config?.app_mode === "oss";
  const isReadOnly =
    scope === "org" && !isOssMode ? !hasPermission("edit_llm_settings") : false;

  // Route all downstream memos through a JSON signature so callers passing
  // a fresh `settingsSources` array reference on every render don't
  // invalidate component state (e.g. the selected view).
  const sourcesSignature = React.useMemo(
    () =>
      JSON.stringify(
        settingsSources.map((s) => ({
          source: s.settingsSource,
          sectionKeys: s.sectionKeys,
          excludeKeys: s.excludeKeys ? Array.from(s.excludeKeys).sort() : null,
        })),
      ),
    [settingsSources],
  );

  // Stable list of source configs; reference only changes when the
  // signature changes (i.e. semantic content has actually changed).
  const resolvedSourceConfigs = React.useMemo<SettingsSourceConfig[]>(() => {
    const parsed = JSON.parse(sourcesSignature) as Array<{
      source: SettingsValueSource;
      sectionKeys: string[];
      excludeKeys: string[] | null;
    }>;
    return parsed.map((p) => ({
      settingsSource: p.source,
      sectionKeys: p.sectionKeys,
      excludeKeys: p.excludeKeys ? new Set(p.excludeKeys) : undefined,
    }));
  }, [sourcesSignature]);

  const getSchemaForSource = React.useCallback(
    (source: SettingsValueSource) =>
      source === "conversation_settings"
        ? conversationSchemaQuery.data
        : agentSchemaQuery.data,
    [agentSchemaQuery.data, conversationSchemaQuery.data],
  );

  // Are we waiting on any schema this page actually uses?
  const isSchemaLoading = resolvedSourceConfigs.some((src) =>
    src.settingsSource === "conversation_settings"
      ? conversationSchemaQuery.isLoading
      : agentSchemaQuery.isLoading,
  );

  // Build a per-source filtered schema (sections matching its sectionKeys).
  const resolvedSources = React.useMemo<ResolvedSource[]>(
    () =>
      resolvedSourceConfigs.map((src) => {
        const schema = getSchemaForSource(src.settingsSource);
        if (!schema) {
          return { ...src, filteredSchema: null };
        }
        const sectionSet = new Set(src.sectionKeys);
        const filteredSchema: SettingsSchema = {
          ...schema,
          sections: schema.sections.filter((s) => sectionSet.has(s.key)),
        };
        return { ...src, filteredSchema };
      }),
    [resolvedSourceConfigs, getSchemaForSource],
  );

  const showAdvanced =
    forceShowAdvancedView ||
    resolvedSources.some((src) => hasAdvancedSettings(src.filteredSchema));
  const showAll =
    allowAllView &&
    resolvedSources.some((src) => hasMinorSettings(src.filteredSchema));

  const [view, setView] = React.useState<SettingsView>("basic");
  const [valuesBySource, setValuesBySource] = React.useState<
    Partial<Record<SettingsValueSource, SettingsFormValues>>
  >({});
  const [dirtyBySource, setDirtyBySource] = React.useState<
    Partial<Record<SettingsValueSource, SettingsDirtyState>>
  >({});
  const hasHydratedViewRef = React.useRef(false);

  const initialValuesBySource = React.useMemo<Partial<
    Record<SettingsValueSource, SettingsFormValues>
  > | null>(() => {
    if (!settings) return null;
    const result: Partial<Record<SettingsValueSource, SettingsFormValues>> = {};
    for (const src of resolvedSources) {
      if (!src.filteredSchema) return null;
      result[src.settingsSource] = {
        ...(result[src.settingsSource] ?? {}),
        ...buildInitialSettingsFormValues(
          settings,
          src.filteredSchema,
          src.settingsSource,
        ),
      };
    }
    return result;
  }, [settings, resolvedSources]);

  const initialView = React.useMemo(() => {
    if (!settings) return null;
    let result: SettingsView | null = null;
    for (const src of resolvedSources) {
      if (!src.filteredSchema) return null;
      const perSource = getInitialView
        ? getInitialView(settings, src.filteredSchema)
        : inferInitialView(settings, src.filteredSchema, src.settingsSource);
      result = result ? getMoreDetailedView(result, perSource) : perSource;
    }
    if (!result) return null;
    return normalizeView(result, { showAdvanced, showAll });
  }, [settings, resolvedSources, getInitialView, showAdvanced, showAll]);

  React.useEffect(() => {
    hasHydratedViewRef.current = false;
    setView("basic");
    setValuesBySource({});
    setDirtyBySource({});
  }, [scope, sourcesSignature]);

  React.useEffect(() => {
    if (!initialValuesBySource || !initialView) return;

    setValuesBySource(initialValuesBySource);
    setDirtyBySource({});
    setView((currentView) => {
      if (!hasHydratedViewRef.current) {
        hasHydratedViewRef.current = true;
        return initialView;
      }

      return getLessDetailedView(currentView, initialView);
    });
  }, [initialValuesBySource, initialView]);

  // Map from field key → source it belongs to. Used by the header callback
  // to route generic onChange calls to the right source's bucket.
  const fieldKeyToSource = React.useMemo(() => {
    const map = new Map<string, SettingsValueSource>();
    for (const src of resolvedSources) {
      if (src.filteredSchema) {
        for (const section of src.filteredSchema.sections) {
          for (const field of section.fields) {
            if (!map.has(field.key)) {
              map.set(field.key, src.settingsSource);
            }
          }
        }
      }
    }
    return map;
  }, [resolvedSources]);

  // Unified views over per-source state, used for header callbacks and for
  // dependency-resolution in `isSettingsFieldVisible`.
  const flatValues = React.useMemo<SettingsFormValues>(() => {
    const merged: SettingsFormValues = {};
    for (const src of resolvedSources) {
      Object.assign(merged, valuesBySource[src.settingsSource] ?? {});
    }
    return merged;
  }, [resolvedSources, valuesBySource]);

  const flatDirty = React.useMemo<SettingsDirtyState>(() => {
    const merged: SettingsDirtyState = {};
    for (const src of resolvedSources) {
      Object.assign(merged, dirtyBySource[src.settingsSource] ?? {});
    }
    return merged;
  }, [resolvedSources, dirtyBySource]);

  const handleFieldChange = React.useCallback(
    (fieldKey: string, nextValue: string | boolean) => {
      const sourceKey = fieldKeyToSource.get(fieldKey);
      if (!sourceKey) return;
      setValuesBySource((prev) => ({
        ...prev,
        [sourceKey]: {
          ...(prev[sourceKey] ?? {}),
          [fieldKey]: nextValue,
        },
      }));
      setDirtyBySource((prev) => ({
        ...prev,
        [sourceKey]: {
          ...(prev[sourceKey] ?? {}),
          [fieldKey]: true,
        },
      }));
    },
    [fieldKeyToSource],
  );

  const handleError = React.useCallback(
    (error: AxiosError) => {
      const msg = retrieveAxiosErrorMessage(error);
      displayErrorToast(msg || t(I18nKey.ERROR$GENERIC));
    },
    [t],
  );

  const handleSave = () => {
    if (isReadOnly) return;
    if (resolvedSources.some((src) => !src.filteredSchema)) return;

    let payload: Record<string, unknown>;
    try {
      const defaultPayload: Record<string, unknown> = {};
      for (const src of resolvedSources) {
        const schema = src.filteredSchema!;
        const sourceValues = valuesBySource[src.settingsSource] ?? {};
        const sourceDirty = dirtyBySource[src.settingsSource] ?? {};
        const diff = buildSdkSettingsPayloadForView(
          schema,
          sourceValues,
          sourceDirty,
          view,
        );
        if (Object.keys(diff).length > 0) {
          const diffKey = PAYLOAD_DIFF_KEY[src.settingsSource];
          defaultPayload[diffKey] = {
            ...((defaultPayload[diffKey] as
              | Record<string, unknown>
              | undefined) ?? {}),
            ...diff,
          };
        }
      }

      payload = buildPayload
        ? buildPayload(defaultPayload, {
            values: flatValues,
            dirty: flatDirty,
            view,
          })
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
        setDirtyBySource({});
        onSaveSuccess?.();
      },
    });
  };

  if (isLoading || isFetching || isSchemaLoading) {
    return <LlmSettingsInputsSkeleton />;
  }

  const hasAnyVisibleSection = resolvedSources.some(
    (src) => src.filteredSchema && src.filteredSchema.sections.length > 0,
  );

  if (!hasAnyVisibleSection) {
    return (
      <Typography.Paragraph className="text-tertiary-alt">
        {t(I18nKey.SETTINGS$SDK_SCHEMA_UNAVAILABLE)}
      </Typography.Paragraph>
    );
  }

  if (Object.keys(flatValues).length === 0) {
    return <LlmSettingsInputsSkeleton />;
  }

  const isDirty = Object.keys(flatDirty).length > 0;

  return (
    <div data-testid={testId} className="h-full relative">
      <ViewToggle
        view={view}
        setView={setView}
        showAdvanced={showAdvanced}
        showAll={showAll}
        isDisabled={isReadOnly}
        trailing={trailingActions}
      />

      <div className="flex flex-col gap-8 pb-20">
        {header?.({
          values: flatValues,
          isDisabled: isReadOnly,
          view,
          onChange: handleFieldChange,
        })}

        {resolvedSources.map((src) => {
          if (!src.filteredSchema) return null;
          const sourceValues = valuesBySource[src.settingsSource] ?? {};
          const visibleSections = getVisibleSettingsSections(
            src.filteredSchema,
            // Use flatValues for dependency resolution so cross-source
            // `depends_on` works; in practice fields only depend on keys
            // within the same source.
            { ...flatValues, ...sourceValues },
            view,
            src.excludeKeys ?? EMPTY_EXCLUDE_KEYS,
          );
          return visibleSections.map((section) => (
            <section
              key={`${src.settingsSource}:${section.key}`}
              className="flex flex-col gap-4"
            >
              <div className="grid gap-4 xl:grid-cols-2">
                {section.fields.map((field) => (
                  <SchemaField
                    key={field.key}
                    field={field}
                    value={sourceValues[field.key]}
                    isDisabled={isReadOnly}
                    onChange={(nextValue) =>
                      handleFieldChange(field.key, nextValue)
                    }
                  />
                ))}
              </div>
            </section>
          ));
        })}
      </div>

      {!isReadOnly ? (
        <div className="sticky bottom-0 bg-base py-4">
          <BrandButton
            testId="save-button"
            type="button"
            variant="primary"
            isDisabled={isPending || (!isDirty && !extraDirty)}
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
