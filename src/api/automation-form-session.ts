import type {
  AutomationSetupDraft,
  AutomationSetupField,
  AutomationSetupFieldUpdateSource,
  AutomationSetupFormPatch,
  AutomationSetupFormValues,
  AutomationSetupKind,
  AutomationSetupPatchResult,
  AutomationSetupTriggerKind,
  AutomationSetupFrequency,
} from "#/api/automation-setup-types";

const AUTOMATION_FORM_SESSION_CHANGED_EVENT =
  "openhands:automation-form-session-changed";
const AUTOMATION_SETUP_KINDS: AutomationSetupKind[] = [
  "prompt",
  "plugin",
  "custom",
];
const AUTOMATION_SETUP_TRIGGER_KINDS: AutomationSetupTriggerKind[] = [
  "cron",
  "event",
];
const AUTOMATION_SETUP_FREQUENCIES: AutomationSetupFrequency[] = [
  "hourly",
  "daily",
  "weekdays",
  "weekly",
  "custom",
];
const STRING_FIELDS = [
  "name",
  "prompt",
  "repository",
  "pluginSource",
  "pluginRef",
  "customCode",
  "entrypoint",
  "setupScriptPath",
  "setupScript",
  "time",
  "timezone",
  "customSchedule",
  "eventSource",
  "eventKey",
  "eventFilter",
  "timeoutSeconds",
] as const satisfies readonly AutomationSetupField[];

const sessions = new Map<string, AutomationSetupDraft>();

function isAutomationSetupKind(value: unknown): value is AutomationSetupKind {
  return (
    typeof value === "string" &&
    AUTOMATION_SETUP_KINDS.includes(value as AutomationSetupKind)
  );
}

function normalizeString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function normalizeFormPatch(value: unknown): AutomationSetupFormPatch {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return {};
  }

  const source = value as Record<string, unknown>;
  const form: AutomationSetupFormPatch = {};
  for (const field of STRING_FIELDS) {
    const fieldValue = normalizeString(source[field]);
    if (fieldValue !== null) form[field] = fieldValue;
  }
  if (isAutomationSetupKind(source.kind)) form.kind = source.kind;
  if (
    typeof source.triggerKind === "string" &&
    AUTOMATION_SETUP_TRIGGER_KINDS.includes(
      source.triggerKind as AutomationSetupTriggerKind,
    )
  ) {
    form.triggerKind = source.triggerKind as AutomationSetupTriggerKind;
  }
  if (
    typeof source.frequency === "string" &&
    AUTOMATION_SETUP_FREQUENCIES.includes(
      source.frequency as AutomationSetupFrequency,
    )
  ) {
    form.frequency = source.frequency as AutomationSetupFrequency;
  }
  if (typeof source.showTimeout === "boolean") {
    form.showTimeout = source.showTimeout;
  }
  return form;
}

function normalizeFieldMetadata(
  value: AutomationSetupDraft["fieldMetadata"],
): AutomationSetupDraft["fieldMetadata"] {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return undefined;
  }

  const metadata: AutomationSetupDraft["fieldMetadata"] = {};
  for (const [field, entry] of Object.entries(value)) {
    if (
      typeof entry === "object" &&
      entry !== null &&
      !Array.isArray(entry) &&
      (entry.updatedBy === "agent" || entry.updatedBy === "user") &&
      typeof entry.updatedAt === "string" &&
      typeof entry.userDirty === "boolean"
    ) {
      metadata[field as AutomationSetupField] = entry;
    }
  }
  return Object.keys(metadata).length > 0 ? metadata : undefined;
}

function normalizeDraft(value: AutomationSetupDraft): AutomationSetupDraft {
  const form = normalizeFormPatch(value.form);
  const prompt = form.prompt ?? value.prompt;
  const kind = form.kind ?? value.kind;
  if (typeof prompt !== "string" || !isAutomationSetupKind(kind)) {
    return { prompt: "", kind: "prompt" };
  }

  const plugins = Array.isArray(value.plugins)
    ? value.plugins.filter(
        (plugin): plugin is string => typeof plugin === "string",
      )
    : [];
  const pluginSource = form.pluginSource ?? plugins[0];
  const normalizedForm: AutomationSetupFormPatch = {
    ...form,
    prompt,
    kind,
    ...(pluginSource ? { pluginSource } : {}),
  };
  const normalizedPlugins = pluginSource ? [pluginSource] : plugins;
  const fieldMetadata = normalizeFieldMetadata(value.fieldMetadata);
  const appliedAgentEventIds = Array.isArray(value.appliedAgentEventIds)
    ? [
        ...new Set(
          value.appliedAgentEventIds.filter((id) => typeof id === "string"),
        ),
      ]
    : undefined;

  return {
    prompt,
    kind,
    ...(normalizedPlugins.length > 0 ? { plugins: normalizedPlugins } : {}),
    form: normalizedForm,
    ...(fieldMetadata ? { fieldMetadata } : {}),
    ...(appliedAgentEventIds && appliedAgentEventIds.length > 0
      ? { appliedAgentEventIds }
      : {}),
  };
}

function isEmptyValue(value: AutomationSetupFormValues[AutomationSetupField]) {
  if (typeof value === "string") return value.trim() === "";
  return value === false;
}

function eventDetail(
  conversationId: string,
  draft: AutomationSetupDraft | null,
  result?: AutomationSetupPatchResult,
) {
  return { conversationId, draft, result };
}

function notifySessionChanged(
  conversationId: string,
  draft: AutomationSetupDraft | null,
  result?: AutomationSetupPatchResult,
) {
  if (typeof window === "undefined") return;
  window.dispatchEvent(
    new CustomEvent(AUTOMATION_FORM_SESSION_CHANGED_EVENT, {
      detail: eventDetail(conversationId, draft, result),
    }),
  );
}

export function initializeAutomationFormSession(
  conversationId: string,
  draft: AutomationSetupDraft,
) {
  sessions.set(conversationId, normalizeDraft(draft));
}

export function getAutomationFormSession(
  conversationId: string | null | undefined,
): AutomationSetupDraft | null {
  if (!conversationId) return null;
  return sessions.get(conversationId) ?? null;
}

export function patchAutomationFormSession(
  conversationId: string,
  patch: AutomationSetupFormPatch,
  options: {
    source: AutomationSetupFieldUpdateSource;
    overwriteUserEdits?: boolean;
    eventId?: string | null;
    updatedAt?: string;
  },
): AutomationSetupPatchResult {
  const existing = sessions.get(conversationId);
  const result: AutomationSetupPatchResult = {
    applied: [],
    skipped: [],
    duplicate: false,
  };
  if (!existing) return result;

  if (
    options.source === "agent" &&
    options.eventId &&
    existing.appliedAgentEventIds?.includes(options.eventId)
  ) {
    return { ...result, duplicate: true };
  }

  const normalizedPatch = normalizeFormPatch(patch);
  const nextForm: AutomationSetupFormPatch = { ...(existing.form ?? {}) };
  const nextMetadata: NonNullable<AutomationSetupDraft["fieldMetadata"]> = {
    ...(existing.fieldMetadata ?? {}),
  };
  const updatedAt = options.updatedAt ?? new Date().toISOString();

  for (const [fieldName, value] of Object.entries(normalizedPatch)) {
    const field = fieldName as AutomationSetupField;
    const currentValue = nextForm[field];
    const metadata = nextMetadata[field];
    const userDirty = metadata?.userDirty === true;
    if (
      options.source === "agent" &&
      userDirty &&
      !options.overwriteUserEdits
    ) {
      result.skipped.push(field);
      continue;
    }
    if (
      options.source === "agent" &&
      metadata?.updatedAt &&
      metadata.updatedAt > updatedAt &&
      !options.overwriteUserEdits
    ) {
      result.skipped.push(field);
      continue;
    }
    if (
      options.source === "agent" &&
      currentValue !== undefined &&
      !isEmptyValue(currentValue) &&
      userDirty &&
      !options.overwriteUserEdits
    ) {
      result.skipped.push(field);
      continue;
    }

    (nextForm as Record<string, unknown>)[field] = value;
    nextMetadata[field] = {
      updatedBy: options.source,
      updatedAt,
      userDirty: options.source === "user" || userDirty,
    };
    result.applied.push(field);
  }

  const nextDraft = normalizeDraft({
    ...existing,
    form: nextForm,
    prompt: nextForm.prompt ?? existing.prompt,
    kind: nextForm.kind ?? existing.kind,
    plugins:
      typeof nextForm.pluginSource === "string" && nextForm.pluginSource
        ? [nextForm.pluginSource]
        : existing.plugins,
    fieldMetadata: nextMetadata,
    appliedAgentEventIds:
      options.source === "agent" && options.eventId
        ? [...(existing.appliedAgentEventIds ?? []), options.eventId]
        : existing.appliedAgentEventIds,
  });

  sessions.set(conversationId, nextDraft);
  notifySessionChanged(conversationId, nextDraft, result);
  return result;
}

export function subscribeAutomationFormSession(
  conversationId: string,
  listener: (
    draft: AutomationSetupDraft | null,
    result?: AutomationSetupPatchResult,
  ) => void,
): () => void {
  if (typeof window === "undefined") return () => {};

  const handler = (event: Event) => {
    const detail = (event as CustomEvent<ReturnType<typeof eventDetail>>)
      .detail;
    if (detail?.conversationId !== conversationId) return;
    listener(detail.draft, detail.result);
  };
  window.addEventListener(AUTOMATION_FORM_SESSION_CHANGED_EVENT, handler);
  return () =>
    window.removeEventListener(AUTOMATION_FORM_SESSION_CHANGED_EVENT, handler);
}

export function clearAutomationFormSession(conversationId: string) {
  sessions.delete(conversationId);
  notifySessionChanged(conversationId, null);
}
