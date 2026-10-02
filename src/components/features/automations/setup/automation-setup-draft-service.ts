import { isSdkHttpError } from "#/api/agent-server-compatibility";
import type {
  AutomationSetupDraft,
  AutomationSetupFormPatch,
  AutomationSetupFormValues,
  AutomationSetupKind,
} from "#/api/automation-setup-types";
import { serializeAutomationSetupPluginList } from "#/api/automation-setup-plugins";
import { getAutomationEndpoint } from "#/manifests/automation-interface";
import type {
  InterfaceEndpointName,
  AutomationDraftApiResponse,
  AutomationDraftEndpoint,
} from "#/manifests/types";
import { I18nKey } from "#/i18n/declaration";
import type { AutomationRun } from "#/types/automation";

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object"
    ? (value as Record<string, unknown>)
    : null;
}

function getStringField(
  value: Record<string, unknown>,
  key: string,
): string | undefined {
  const field = value[key];
  return typeof field === "string" ? field : undefined;
}

function getFirstObject(
  value: Record<string, unknown>,
  key: string,
): Record<string, unknown> | null {
  const field = value[key];
  return Array.isArray(field) ? asRecord(field[0]) : null;
}

function getPluginEntries(
  value: Record<string, unknown>,
): { source: string; ref: string }[] {
  const field = value.plugins;
  if (!Array.isArray(field)) return [];
  return field.flatMap((item) => {
    const record = asRecord(item);
    if (!record) return [];
    return [
      {
        source: getStringField(record, "source") ?? "",
        ref: getStringField(record, "ref") ?? "",
      },
    ];
  });
}

function kindForDraftEndpoint(
  endpoint: AutomationDraftEndpoint,
): AutomationSetupKind {
  if (endpoint === "/v1") return "custom";
  if (endpoint === "/v1/preset/plugin") return "plugin";
  return "prompt";
}

export function setupDraftFromServerDraft(
  saved: AutomationDraftApiResponse,
): AutomationSetupDraft {
  const body = saved.draft as Record<string, unknown>;
  const trigger = asRecord(body.trigger);
  const repo = getFirstObject(body, "repos");
  const pluginEntries = getPluginEntries(body);
  const kind = kindForDraftEndpoint(saved.endpoint);
  const form: AutomationSetupFormPatch = { kind };

  const savedName = saved.name ?? getStringField(body, "name");
  if (savedName !== undefined) form.name = savedName;
  const savedPrompt = getStringField(body, "prompt");
  if (savedPrompt !== undefined) form.prompt = savedPrompt;
  const savedRepository =
    getStringField(repo ?? {}, "url") ??
    (typeof body.repository === "string" ? body.repository : undefined);
  if (savedRepository !== undefined) form.repository = savedRepository;
  if (pluginEntries.length > 0) {
    form.pluginSource = pluginEntries[0]?.source ?? "";
    form.pluginRef = pluginEntries[0]?.ref ?? "";
    form.pluginList = serializeAutomationSetupPluginList(pluginEntries);
  }
  const savedEntrypoint = getStringField(body, "entrypoint");
  if (savedEntrypoint !== undefined) form.entrypoint = savedEntrypoint;
  const savedSetupScriptPath = getStringField(body, "setup_script_path");
  if (savedSetupScriptPath !== undefined) {
    form.setupScriptPath = savedSetupScriptPath;
  }
  if (trigger) {
    form.triggerKind =
      getStringField(trigger, "type") === "event" ? "event" : "cron";
    const schedule = getStringField(trigger, "schedule");
    if (schedule) {
      form.frequency = "custom";
      form.customSchedule = schedule;
    }
    const savedTimezone = getStringField(trigger, "timezone");
    if (savedTimezone !== undefined) form.timezone = savedTimezone;
    const savedEventSource = getStringField(trigger, "source");
    if (savedEventSource !== undefined) form.eventSource = savedEventSource;
    const savedEventKey =
      getStringField(trigger, "on") ??
      (Array.isArray(trigger.on) && typeof trigger.on[0] === "string"
        ? trigger.on[0]
        : undefined);
    if (savedEventKey !== undefined) form.eventKey = savedEventKey;
    const savedEventFilter = getStringField(trigger, "filter");
    if (savedEventFilter !== undefined) form.eventFilter = savedEventFilter;
  }
  const savedModel = getStringField(body, "model");
  if (savedModel !== undefined) form.model = savedModel;
  const savedAgentProfileId = getStringField(body, "agent_profile_id");
  if (savedAgentProfileId !== undefined) {
    form.agentProfileId = savedAgentProfileId;
  }
  if (typeof body.timeout === "number") {
    form.showTimeout = true;
    form.timeoutSeconds = String(body.timeout);
  }

  const prompt = form.prompt ?? "";
  return {
    prompt,
    kind,
    ...(pluginEntries.length > 0
      ? { plugins: pluginEntries.map((entry) => entry.source).filter(Boolean) }
      : {}),
    form,
    serverDraftId: saved.id,
    materializedAutomationId: saved.materializedAutomationId,
  };
}

export function formFromServerDraft(
  saved: AutomationDraftApiResponse,
  base: AutomationSetupFormValues,
): AutomationSetupFormValues {
  const body = saved.draft as Record<string, unknown>;
  const trigger = asRecord(body.trigger);
  const plugin = getPluginEntries(body);
  const savedPlugins =
    plugin.length > 0
      ? plugin
      : [
          {
            source: base.pluginSource,
            ref: base.pluginRef,
          },
        ].filter((entry) => entry.source || entry.ref);
  const repo = getFirstObject(body, "repos");
  const endpointKind = kindForDraftEndpoint(saved.endpoint);

  return {
    ...base,
    kind: endpointKind,
    name: saved.name ?? getStringField(body, "name") ?? base.name,
    prompt: getStringField(body, "prompt") ?? base.prompt,
    repository:
      getStringField(repo ?? {}, "url") ??
      (typeof body.repository === "string" ? body.repository : base.repository),
    pluginSource: savedPlugins[0]?.source ?? "",
    pluginRef: savedPlugins[0]?.ref ?? "",
    pluginList:
      savedPlugins.length > 0
        ? serializeAutomationSetupPluginList(savedPlugins)
        : "",
    entrypoint: getStringField(body, "entrypoint") ?? base.entrypoint,
    setupScriptPath:
      getStringField(body, "setup_script_path") ?? base.setupScriptPath,
    triggerKind:
      getStringField(trigger ?? {}, "type") === "event" ? "event" : "cron",
    frequency: getStringField(trigger ?? {}, "schedule")
      ? "custom"
      : base.frequency,
    customSchedule:
      getStringField(trigger ?? {}, "schedule") ?? base.customSchedule,
    timezone: getStringField(trigger ?? {}, "timezone") ?? base.timezone,
    eventSource: getStringField(trigger ?? {}, "source") ?? base.eventSource,
    eventKey:
      getStringField(trigger ?? {}, "on") ??
      (Array.isArray(trigger?.on) && typeof trigger.on[0] === "string"
        ? trigger.on[0]
        : base.eventKey),
    eventFilter: getStringField(trigger ?? {}, "filter") ?? base.eventFilter,
    model: getStringField(body, "model") ?? base.model,
    agentProfileId:
      getStringField(body, "agent_profile_id") ?? base.agentProfileId,
    showTimeout: typeof body.timeout === "number" || base.showTimeout,
    timeoutSeconds:
      typeof body.timeout === "number"
        ? String(body.timeout)
        : base.timeoutSeconds,
  };
}

function endpointName(kind: AutomationSetupKind): InterfaceEndpointName {
  if (kind === "plugin") return "createPlugin";
  if (kind === "custom") return "createBundle";
  return "createPrompt";
}

export function presetKindForEndpoint(
  kind: AutomationSetupKind,
  pluginSource: string,
): AutomationSetupKind {
  if (kind === "custom") return "custom";
  return pluginSource.trim() ? "plugin" : "prompt";
}

export function draftEndpoint(
  kind: AutomationSetupKind,
  pluginSource = "",
): AutomationDraftEndpoint {
  const resolved = presetKindForEndpoint(kind, pluginSource);
  const path = getAutomationEndpoint(endpointName(resolved));
  if (
    path === "/v1" ||
    path === "/v1/preset/prompt" ||
    path === "/v1/preset/plugin"
  ) {
    return path;
  }
  return resolved === "plugin" ? "/v1/preset/plugin" : "/v1/preset/prompt";
}

export function draftValidationEndpoint(
  kind: AutomationSetupKind,
  pluginSource = "",
): string {
  return getAutomationEndpoint(
    endpointName(presetKindForEndpoint(kind, pluginSource)),
  );
}

export function getResponseStatus(error: unknown): number | null {
  if (isSdkHttpError(error)) return (error as { status: number }).status;
  if (!error || typeof error !== "object") return null;
  const response = (error as Record<string, unknown>).response;
  if (!response || typeof response !== "object") return null;
  const status = (response as Record<string, unknown>).status;
  return typeof status === "number" ? status : null;
}

export function isDraftEndpointUnavailable(error: unknown): boolean {
  const status = getResponseStatus(error);
  return status === 404 || status === 405;
}

export function extractDraftDispatchErrors(error: unknown): string | null {
  if (error && typeof error === "object") {
    const record = error as Record<string, unknown>;
    const response = record.response;
    if (response && typeof response === "object") {
      const data = (response as Record<string, unknown>).data;
      if (data && typeof data === "object") {
        const errors = (data as Record<string, unknown>).errors;
        if (Array.isArray(errors) && errors.length > 0) {
          const first = errors[0] as Record<string, unknown> | undefined;
          if (first && typeof first.message === "string") return first.message;
        }
        const message = (data as Record<string, unknown>).message;
        if (typeof message === "string") return message;
      }
    }
  }
  return null;
}

function draftRunFinishedSuccessfully(run: AutomationRun): boolean {
  return String(run.status).toUpperCase() === "COMPLETED";
}

function draftRunFinishedWithFailure(run: AutomationRun): boolean {
  const status = String(run.status).toUpperCase();
  return status === "FAILED" || status === "CANCELLED" || status === "SKIPPED";
}

export function getDraftExecutionStatusText(
  draft: AutomationDraftApiResponse,
  runs: AutomationRun[],
  t: (key: I18nKey) => string,
): string {
  const latestRun = runs[0];
  if (latestRun) {
    if (draftRunFinishedSuccessfully(latestRun)) {
      return t(I18nKey.AUTOMATION_SETUP$LATEST_TEST_PASSED);
    }
    if (draftRunFinishedWithFailure(latestRun)) {
      return t(I18nKey.AUTOMATION_SETUP$LATEST_TEST_FAILED);
    }
    return t(I18nKey.AUTOMATION_SETUP$LATEST_TEST_RUNNING);
  }
  return (
    draft.validationErrors?.[0]?.message ??
    t(I18nKey.AUTOMATION_SETUP$READY_TO_TEST)
  );
}

export function draftRunPageErrors(
  draft: AutomationDraftApiResponse | null,
  statusMessage: { kind: "success" | "error"; text: string } | null,
  runs: AutomationRun[],
): string[] {
  const messages: string[] = [];
  const add = (value: string | null | undefined) => {
    const text = value?.trim();
    if (text && !messages.includes(text)) messages.push(text);
  };
  for (const error of draft?.validationErrors ?? []) add(error.message);
  if (statusMessage?.kind === "error") add(statusMessage.text);
  const latestFailure = runs.find(draftRunFinishedWithFailure);
  add(latestFailure?.error_detail);
  return messages;
}
