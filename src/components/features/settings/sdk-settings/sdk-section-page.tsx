import React from "react";
import { AxiosError } from "axios";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { LlmSettingsInputsSkeleton } from "#/components/features/settings/llm-settings/llm-settings-inputs-skeleton";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import {
  useAgentSettingsSchema,
  useConversationSettingsSchema,
} from "#/hooks/query/use-agent-settings-schema";
import { useSettings } from "#/hooks/query/use-settings";
import { I18nKey } from "#/i18n/declaration";
import { Settings, SettingsSchema, SettingsScope } from "#/types/settings";
import { extensionModuleEmptyStateClassName } from "#/utils/extension-module-card-classes";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";
import {
  buildInitialSettingsFormValues,
  buildSdkSettingsPayload,
  buildSdkSettingsPayloadForView,
  getVisibleSettingsSections,
  hasAdvancedSettings,
  hasCriticalSettings,
  hasMinorSettings,
  inferInitialView,
  isValidSettingsSchema,
  SettingsDirtyState,
  SettingsFormValues,
  type SettingsValueSource,
  type SettingsView,
} from "#/utils/sdk-settings-schema";
import { FIELD_FULL_WIDTH_KEYS, SchemaField } from "./schema-field";
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
    showBasic,
    showAdvanced,
    showAll,
  }: {
    showBasic: boolean;
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

  // A critical-less page has nothing to render in the basic tier; bump up.
  if (!showBasic) {
    if (showAdvanced) return "advanced";
    if (showAll) return "all";
  }
  return "basic";
};

const PAYLOAD_DIFF_KEY: Record<SettingsValueSource, string> = {
  agent_settings: "agent_settings_diff",
  conversation_settings: "conversation_settings_diff",
};

type ValuesBySource = Partial<Record<SettingsValueSource, SettingsFormValues>>;

/**
 * The values to display: the server baseline with the user's edits laid over
 * it, per source. A source present in either side appears in the result.
 */
const mergeOverlay = (
  baseline: ValuesBySource,
  edits: ValuesBySource,
): ValuesBySource => {
  const sources = new Set<SettingsValueSource>([
    ...(Object.keys(baseline) as SettingsValueSource[]),
    ...(Object.keys(edits) as SettingsValueSource[]),
  ]);
  const merged: ValuesBySource = {};
  for (const source of sources) {
    merged[source] = { ...(baseline[source] ?? {}), ...(edits[source] ?? {}) };
  }
  return merged;
};

/**
 * Dirty is a derived fact — the keys the user has an edit for — rather than a
 * flag that has to be cleared in step with the values it describes.
 */
const dirtyFromEdits = (
  edits: ValuesBySource,
): Partial<Record<SettingsValueSource, SettingsDirtyState>> => {
  const dirty: Partial<Record<SettingsValueSource, SettingsDirtyState>> = {};
  for (const [source, fields] of Object.entries(edits) as [
    SettingsValueSource,
    SettingsFormValues | undefined,
  ][]) {
    dirty[source] = Object.fromEntries(
      Object.keys(fields ?? {}).map((key) => [key, true]),
    );
  }
  return dirty;
};

/**
 * Drop overlay entries for schema fields that have gone away.
 *
 * A key the schema once defined and no longer does is unreachable — nothing
 * renders it and `buildSdkSettingsPayload` walks schema fields — yet it would
 * still count towards `dirty` and strand the form behind an enabled Save button
 * that submits nothing.
 *
 * `everSchemaKeys` is every key that has appeared in a baseline so far, and is
 * what separates that case from a key the schema never defined at all. Callers
 * legitimately drive values outside the schema and read them back off
 * `saveControl.values` — `llm-settings-local-view` does this with
 * `llm.provider_connection_id` to keep a profile's connection link — so a key
 * that was never a schema field is left alone. Pruning those would silently
 * unlink the profile on the next save.
 */
const pruneToBaseline = (
  edits: ValuesBySource,
  baseline: ValuesBySource,
  everSchemaKeys: ReadonlySet<string>,
): ValuesBySource => {
  const next: ValuesBySource = {};
  for (const [source, fields] of Object.entries(edits) as [
    SettingsValueSource,
    SettingsFormValues | undefined,
  ][]) {
    const known = baseline[source] ?? {};
    next[source] = Object.fromEntries(
      Object.entries(fields ?? {}).filter(
        ([key]) => key in known || !everSchemaKeys.has(key),
      ),
    );
  }
  return next;
};

/**
 * Remove the keys a save carried from the overlay, leaving the rest.
 *
 * Paired with folding the same keys into the baseline, this is the "rebase" a
 * successful save performs. Clearing the overlay outright would drop the
 * display back to a baseline still holding pre-save values until the refetch
 * lands, so saved fields would visibly flicker to their old values. A field
 * edited while the request was in flight was not submitted, so it stays.
 */
const dropSavedKeys = (
  edits: ValuesBySource,
  saved: ValuesBySource,
): ValuesBySource => {
  const next: ValuesBySource = {};
  for (const [source, fields] of Object.entries(edits) as [
    SettingsValueSource,
    SettingsFormValues | undefined,
  ][]) {
    next[source] = Object.fromEntries(
      Object.entries(fields ?? {}).filter(
        ([key]) => !(key in (saved[source] ?? {})),
      ),
    );
  }
  return next;
};

const getSchemaUnavailableMessage = (
  error: unknown,
  fallbackMessage: string,
): string => {
  if (!(error instanceof AxiosError)) {
    return fallbackMessage;
  }

  if (error.response?.status === 401) {
    return `${fallbackMessage} This agent server requires X-Session-API-Key. Set VITE_SESSION_API_KEY in the frontend to the same value used by the backend SESSION_API_KEY or OH_SESSION_API_KEYS_0.`;
  }

  if (error.response?.status === 404) {
    return `${fallbackMessage} This backend does not expose /api/settings/* schema endpoints. Upgrade to a recent openhands-agent-server release.`;
  }

  return fallbackMessage;
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
 * Snapshot of the page's save state, surfaced to the parent so it can
 * render its own Save/Next button (e.g. in onboarding) when
 * {@link SdkSectionPage}'s built-in button is hidden via
 * `hideSaveButton`.
 */
export interface SdkSectionSaveControl {
  /** Trigger a save of the currently-dirty fields. No-op while `isSaving` or `!isDirty`. */
  save: () => void;
  /** A save mutation is in flight. */
  isSaving: boolean;
  /** At least one field is dirty (or `extraDirty` was passed in). */
  isDirty: boolean;
  /** Current form values (for custom save flows). */
  values: SettingsFormValues;
  /** The active view tier (basic/advanced/all) the form is rendering. */
  view: SettingsView;
  /**
   * Returns the coerced, dirty-only payload as a nested object
   * (e.g. `{ llm: { temperature: 0.7 } }`). Lets a custom save flow persist
   * exactly the fields the user changed, with proper types, without
   * re-implementing schema-driven coercion. Throws if a field fails coercion.
   */
  getDirtyPayload: () => Record<string, unknown>;
}

/**
 * A generic SDK-schema-driven settings page that renders fields from one or
 * more schema sections.
 *
 * The `settingsSources` array specifies which schema(s)/section(s) the page
 * owns. The page tracks values/dirty state per source, renders sections from
 * each source in order (filtered by the schema's `prominence` field for the
 * selected view), and emits a combined save payload like
 * `{ conversation_settings_diff: {...}, agent_settings_diff: {...} }` ---
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
  initialValueOverrides,
  embedded = false,
  hideSaveButton = false,
  suppressSuccessToast = false,
  onSaveControlChange,
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
  getInitialView?: (
    settings: Settings,
    filteredSchema: SettingsSchema,
  ) => SettingsView;
  forceShowAdvancedView?: boolean;
  allowAllView?: boolean;
  /**
   * Per-field initial value overrides that win over the values
   * derived from `useSettings`. The keys of each override are also
   * marked dirty on hydration so the user can save the form without
   * having to touch the prefilled fields. Useful when the page is
   * embedded in a flow that wants to nudge brand-new users toward a
   * particular default (e.g. onboarding pre-filling OpenHands/Opus).
   */
  initialValueOverrides?: SettingsFormValues;
  embedded?: boolean;
  hideSaveButton?: boolean;
  /** Suppress the default success toast after save completes. */
  suppressSuccessToast?: boolean;
  /**
   * Fires whenever the save state changes (a mutation starts/finishes,
   * dirty status flips). Provides a stable `save()` callback the
   * parent can wire to its own button. Useful when the form is
   * embedded in a custom flow and the built-in Save button is hidden.
   */
  onSaveControlChange?: (control: SdkSectionSaveControl) => void;
  testId?: string;
}) {
  const { t } = useTranslation("openhands");
  const { mutate: saveSettings, isPending } = useSaveSettings(scope);
  const { data: settings, isLoading, isFetching } = useSettings(scope);
  const agentSchemaQuery = useAgentSettingsSchema(
    settings?.agent_settings_schema,
  );
  const conversationSchemaQuery = useConversationSettingsSchema(
    settings?.conversation_settings_schema,
  );
  const isReadOnly = false;

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

  const isSchemaLoading = resolvedSourceConfigs.some((src) =>
    src.settingsSource === "conversation_settings"
      ? conversationSchemaQuery.isLoading
      : agentSchemaQuery.isLoading,
  );

  const resolvedSources = React.useMemo<ResolvedSource[]>(
    () =>
      resolvedSourceConfigs.map((src) => {
        const schema = getSchemaForSource(src.settingsSource);
        if (!isValidSettingsSchema(schema)) {
          return { ...src, filteredSchema: null };
        }
        const sectionSet = new Set(src.sectionKeys);
        // The agent schema can carry more than one section per key — e.g. the
        // combined AgentSettings schema emits an "llm" section for both the
        // "openhands" and "acp" variants. Only the first (openhands) is used,
        // so keep the first section per key; otherwise every field renders
        // twice and React sees duplicate section keys.
        const seenKeys = new Set<string>();
        const filteredSchema: SettingsSchema = {
          ...schema,
          sections: schema.sections.filter((s) => {
            if (!sectionSet.has(s.key) || seenKeys.has(s.key)) return false;
            seenKeys.add(s.key);
            return true;
          }),
        };
        return { ...src, filteredSchema };
      }),
    [resolvedSourceConfigs, getSchemaForSource],
  );

  // The basic tier only exists when some field renders in it; a critical-less
  // page (e.g. Memory, whose only field is major) hides the Basic tab and
  // floors its view at "advanced".
  const showBasic = resolvedSources.some((src) =>
    hasCriticalSettings(src.filteredSchema),
  );
  const showAdvanced =
    forceShowAdvancedView ||
    resolvedSources.some((src) => hasAdvancedSettings(src.filteredSchema));
  const showAll =
    allowAllView &&
    resolvedSources.some((src) => hasMinorSettings(src.filteredSchema));

  const schemaUnavailableMessage = React.useMemo(() => {
    const firstError = resolvedSourceConfigs.reduce<unknown>(
      (err, src) =>
        err ??
        (src.settingsSource === "conversation_settings"
          ? conversationSchemaQuery.error
          : agentSchemaQuery.error),
      null,
    );
    return getSchemaUnavailableMessage(
      firstError,
      t(I18nKey.SETTINGS$SDK_SCHEMA_UNAVAILABLE),
    );
  }, [
    resolvedSourceConfigs,
    agentSchemaQuery.error,
    conversationSchemaQuery.error,
    t,
  ]);

  const overridesSignature = React.useMemo(
    () => (initialValueOverrides ? JSON.stringify(initialValueOverrides) : ""),
    [initialValueOverrides],
  );

  const [view, setView] = React.useState<SettingsView>("basic");
  /**
   * Server-derived values, replaced wholesale on every settings/schema
   * refetch. Never carries user input.
   */
  const [baselineBySource, setBaselineBySource] =
    React.useState<ValuesBySource>({});
  /**
   * The user's unsaved edits, keyed only by the fields they actually touched.
   * Survives a refetch of {@link baselineBySource}, which is what stops a
   * background refresh from silently discarding what someone has typed.
   */
  const [editsBySource, setEditsBySource] = React.useState<ValuesBySource>({});
  const hasHydratedViewRef = React.useRef(false);
  const seededOverridesRef = React.useRef<string | null>(null);
  /**
   * Every key that has appeared in a baseline for this scope/source pair, so
   * the prune can tell a schema field that disappeared from a key the schema
   * never defined.
   */
  const everSchemaKeysRef = React.useRef<Set<string>>(new Set());

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
    // Overrides deliberately do NOT go into the baseline. The baseline is what
    // the server says; an override is a local prefill, so it belongs in the
    // overlay with every other unsaved value. Merging it here would let it
    // outrank the server copy forever — including after the user has edited
    // and saved the field, whose confirming refetch would then be overwritten
    // by the stale prefill.
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
    return normalizeView(result, { showBasic, showAdvanced, showAll });
  }, [
    settings,
    resolvedSources,
    getInitialView,
    showBasic,
    showAdvanced,
    showAll,
  ]);

  // A scope or source change is a different form, so edits made against the
  // previous one are deliberately discarded rather than carried over.
  React.useEffect(() => {
    hasHydratedViewRef.current = false;
    seededOverridesRef.current = null;
    everSchemaKeysRef.current = new Set();
    setView("basic");
    setBaselineBySource({});
    setEditsBySource({});
  }, [scope, sourcesSignature]);

  React.useEffect(() => {
    if (!initialValuesBySource || !initialView) return;

    // The baseline always tracks the server. Edits are left alone: this effect
    // re-runs on every settings/schema refetch, and replacing the values here
    // is what used to discard unsaved input while the form stayed on screen.
    setBaselineBySource(initialValuesBySource);

    // Drop overlay entries for fields the new schema no longer defines. They
    // can never be displayed or submitted again — `buildSdkSettingsPayload`
    // walks schema fields — so leaving them would keep the form permanently
    // dirty behind a Save button that builds an empty payload and no-ops.
    // Keys the schema never defined are spared: a caller may drive a value
    // outside the schema and read it back off `saveControl.values`.
    for (const fields of Object.values(initialValuesBySource)) {
      for (const key of Object.keys(fields ?? {})) {
        everSchemaKeysRef.current.add(key);
      }
    }
    setEditsBySource((prev) =>
      pruneToBaseline(prev, initialValuesBySource, everSchemaKeysRef.current),
    );

    // Overrides are seeded into the overlay once per override set, so they
    // start dirty and are savable untouched. Re-seeding on every refetch would
    // resurrect a prefill the user had deliberately cleared or already saved.
    if (
      initialValueOverrides &&
      seededOverridesRef.current !== overridesSignature
    ) {
      seededOverridesRef.current = overridesSignature;
      const firstSource = resolvedSources[0]?.settingsSource;
      if (firstSource) {
        setEditsBySource((prev) => ({
          ...prev,
          // The new overrides win. This branch only runs when the caller
          // actually changed the override set, and a changed prefill is a
          // deliberate instruction — spreading `prev` last would let the
          // previous seed outrank it and the new values would never appear.
          [firstSource]: {
            ...(prev[firstSource] ?? {}),
            ...initialValueOverrides,
          },
        }));
      }
    }

    // The ref flip stays outside the updater: React double-invokes state
    // updaters in StrictMode, so mutating it in there makes the second
    // (kept) call take the already-hydrated branch and pin the view.
    if (!hasHydratedViewRef.current) {
      hasHydratedViewRef.current = true;
      setView(initialView);
    } else {
      setView((currentView) => getLessDetailedView(currentView, initialView));
    }
    // `overridesSignature` is a dependency because overrides no longer feed
    // `initialValuesBySource`; without it a changed override set would never
    // re-seed the overlay.
  }, [initialValuesBySource, initialView, overridesSignature]);

  // Displayed values and dirty state are both derived from the two stores
  // above, so they can never disagree with each other the way two
  // independently-mutated trees could.
  const valuesBySource = React.useMemo(
    () => mergeOverlay(baselineBySource, editsBySource),
    [baselineBySource, editsBySource],
  );
  const dirtyBySource = React.useMemo(
    () => dirtyFromEdits(editsBySource),
    [editsBySource],
  );

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
      // One write, to the overlay only. The displayed value and the dirty flag
      // both fall out of it, so they cannot drift apart.
      setEditsBySource((prev) => ({
        ...prev,
        [sourceKey]: {
          ...(prev[sourceKey] ?? {}),
          [fieldKey]: nextValue,
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

  const handleSaveRef = React.useRef<() => void>(() => {});
  const stableSave = React.useCallback(() => {
    handleSaveRef.current();
  }, []);

  // Stable accessor for the coerced, dirty-only payload. Mirrors the
  // `handleSaveRef` pattern so the exposed function reference stays stable
  // across renders while always reading the latest closure at call time.
  const buildDirtyPayloadRef = React.useRef<() => Record<string, unknown>>(
    () => ({}),
  );
  const stableGetDirtyPayload = React.useCallback(
    () => buildDirtyPayloadRef.current(),
    [],
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

    // The overlay as it stood when Save was pressed (`handleSaveRef` is rebound
    // every render, so this closure reads the current one).
    //
    // Not the same set as the payload: `buildSdkSettingsPayloadForView`
    // rewrites every view-invisible field to its schema default after the
    // dirty pass, and a caller-supplied `buildPayload` may drop more. These are
    // the edits the save *consumed* — they stop being pending either way, which
    // matches what `setDirtyBySource({})` did here before.
    const consumedEdits = editsBySource;

    saveSettings(payload, {
      onError: handleError,
      onSuccess: () => {
        if (!suppressSuccessToast) {
          displaySuccessToast(t(I18nKey.SETTINGS$SAVED_WARNING));
        }
        // Rebase rather than clear, so the form keeps showing what was saved
        // until the refetch confirms it.
        setBaselineBySource((prev) => mergeOverlay(prev, consumedEdits));
        setEditsBySource((prev) => dropSavedKeys(prev, consumedEdits));
        onSaveSuccess?.();
      },
    });
  };

  handleSaveRef.current = handleSave;
  // Dirty-only (NOT view-filtered): we must never inject defaults for
  // non-visible fields here, or a custom save flow would reset fields the
  // user never touched. `buildSdkSettingsPayloadForView` is reserved for the
  // built-in full-replace save above. With multiple sources, we merge each
  // source's nested payload at the top level so single-source consumers
  // (e.g. `LlmSettingsLocalView`) keep reading `.llm` etc. unchanged.
  buildDirtyPayloadRef.current = () => {
    const merged: Record<string, unknown> = {};
    for (const src of resolvedSources) {
      if (!src.filteredSchema) continue;
      const sourceValues = valuesBySource[src.settingsSource] ?? {};
      const sourceDirty = dirtyBySource[src.settingsSource] ?? {};
      Object.assign(
        merged,
        buildSdkSettingsPayload(src.filteredSchema, sourceValues, sourceDirty),
      );
    }
    return merged;
  };

  const isDirty = Object.keys(flatDirty).length > 0;
  const saveControlIsDirty = isDirty || extraDirty;
  React.useEffect(() => {
    if (!onSaveControlChange) return;
    onSaveControlChange({
      save: stableSave,
      isSaving: isPending,
      isDirty: saveControlIsDirty,
      values: flatValues,
      view,
      getDirtyPayload: stableGetDirtyPayload,
    });
  }, [isPending, saveControlIsDirty, flatValues, view]);

  // Keep existing form content visible during background refetches to avoid
  // flashing the full skeleton (notably during onboarding Next transitions).
  const isInitialSettingsLoad = (isLoading || isFetching) && !settings;
  if (isInitialSettingsLoad || isSchemaLoading) {
    return <LlmSettingsInputsSkeleton />;
  }

  const hasAnyVisibleSection = resolvedSources.some(
    (src) => src.filteredSchema && src.filteredSchema.sections.length > 0,
  );

  if (!hasAnyVisibleSection) {
    return (
      <div
        data-testid="sdk-schema-unavailable"
        className={extensionModuleEmptyStateClassName}
      >
        <p className="text-sm text-[var(--oh-muted)]">
          {schemaUnavailableMessage}
        </p>
      </div>
    );
  }

  if (Object.keys(flatValues).length === 0) {
    return <LlmSettingsInputsSkeleton />;
  }

  // Scrolling is owned by the settings shell (or onboarding wrapper), not a
  // nested scroll region. Save actions are inline after the last field.
  const bodyClassName = "flex flex-col gap-8";

  return (
    <div
      data-testid={testId}
      className={
        embedded
          ? "relative flex min-h-0 w-full flex-1 flex-col"
          : "relative w-full min-h-0"
      }
    >
      <ViewToggle
        view={view}
        setView={setView}
        showBasic={showBasic}
        showAdvanced={showAdvanced}
        showAll={showAll}
        isDisabled={isReadOnly}
      />

      <div className={bodyClassName}>
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
                  <div
                    key={field.key}
                    className={
                      FIELD_FULL_WIDTH_KEYS.has(field.key)
                        ? "xl:col-span-2"
                        : undefined
                    }
                  >
                    <SchemaField
                      field={field}
                      value={sourceValues[field.key]}
                      isDisabled={isReadOnly}
                      onChange={(nextValue) =>
                        handleFieldChange(field.key, nextValue)
                      }
                    />
                  </div>
                ))}
              </div>
            </section>
          ));
        })}

        {!isReadOnly && !hideSaveButton ? (
          <div className="flex justify-start pt-2">
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
    </div>
  );
}
