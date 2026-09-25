import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import toast from "react-hot-toast";
import { useTranslation } from "react-i18next";
import {
  ArrowLeft,
  CalendarDays,
  Clock3,
  Code2,
  FileText,
  Globe2,
  Puzzle,
  Zap,
} from "lucide-react";
import AutomationService, {
  type CustomWebhookCreateResponse,
} from "#/api/automation-service/automation-service.api";
import { isSdkHttpError } from "#/api/agent-server-compatibility";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import {
  patchAutomationSetupDraft,
  subscribeAutomationSetupDraft,
  type AutomationSetupDraft,
  type AutomationSetupField,
  type AutomationSetupFormPatch,
  type AutomationSetupFormValues,
  type AutomationSetupKind,
} from "#/api/automation-setup-draft-store";
import {
  automationDetailPath,
  getAutomationEndpoint,
} from "#/manifests/automation-interface";
import { packTarGzip } from "#/utils/tar-gzip";
import { AvailableLanguages } from "#/i18n";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import {
  formControlFieldClassName,
  formControlMultilineFieldClassName,
  formControlTransitionClassName,
} from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import { useDeploymentCapabilities } from "#/hooks/query/use-manifest-capabilities";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { useNavigation } from "#/context/navigation-context";
import type {
  AutomationDraftApiResponse,
  AutomationDraftEndpoint,
  InterfaceEndpointName,
  SetupRequestBody,
} from "#/manifests/types";
import type { AutomationRun } from "#/types/automation";
import { formatRelativeTime } from "#/utils/format-relative-time";
import { ActivityLogItem } from "../detail/activity-log-item";
import {
  buildAutomationDraftTags,
  buildAutomationSetupModeTags,
  getAutomationDraftIdFromTags,
  hasAutomationSetupModeTag,
  removeAutomationDraftTags,
} from "#/utils/automation-draft-tags";

const DEFAULT_TIMEZONE = "America/New_York";
const DEFAULT_TIME = "09:00";
const DEFAULT_CUSTOM_SCHEDULE = "0 9 * * *";
const DEFAULT_EVENT_SOURCE = "github";
const DEFAULT_EVENT_KEY = "issue_comment.created";
const DEFAULT_CUSTOM_ENTRYPOINT = "python3 main.py";
const MAIN_PY_FILENAME = "main.py";
const DEFAULT_CUSTOM_SETUP_SCRIPT_PATH = "setup.sh";
const DEFAULT_CUSTOM_SETUP_SCRIPT = `#!/usr/bin/env bash
:
`;
const DEFAULT_TIMEOUT_SECONDS = "600";
const PREFLIGHT_TARBALL_PATH =
  "oh-internal://uploads/00000000-0000-0000-0000-000000000000";
export const AGENT_FIELD_STREAM_CHARACTER_DELAY_MS = 12;
export const AGENT_FIELD_STREAM_SETTLE_DELAY_MS = 160;

const AUTOMATION_SETUP_KINDS: AutomationSetupKind[] = [
  "prompt",
  "plugin",
  "custom",
];
const FREQUENCIES = [
  "hourly",
  "daily",
  "weekdays",
  "weekly",
  "custom",
] as const;
const AUTOMATION_SETUP_FIELD_RENDER_ORDER: AutomationSetupField[] = [
  "kind",
  "name",
  "prompt",
  "pluginSource",
  "pluginRef",
  "repository",
  "customCode",
  "entrypoint",
  "setupScriptPath",
  "setupScript",
  "triggerKind",
  "frequency",
  "time",
  "timezone",
  "customSchedule",
  "eventSource",
  "eventKey",
  "eventFilter",
  "showTimeout",
  "timeoutSeconds",
];

type CustomWebhookSignatureScheme =
  | "hmac_sha256_hex"
  | "standard_webhooks"
  | "slack_v0";

interface CustomWebhookFormState {
  enabled: boolean;
  name: string;
  eventKeyExpr: string;
  signatureHeader: string;
  signatureScheme: CustomWebhookSignatureScheme;
  webhookSecret: string;
}

const DEFAULT_CUSTOM_WEBHOOK_EVENT_KEY_EXPR = "type";
const DEFAULT_CUSTOM_WEBHOOK_SIGNATURE_HEADER = "X-Signature-256";
const DEFAULT_CUSTOM_WEBHOOK_SIGNATURE_SCHEME: CustomWebhookSignatureScheme =
  "hmac_sha256_hex";

const NON_CHARACTER_STREAM_FIELDS = new Set<AutomationSetupField>([
  "kind",
  "triggerKind",
  "frequency",
  "showTimeout",
  "time",
]);
const streamingFieldHighlightClassName =
  "rounded-xl ring-2 ring-[#D5C76B]/80 ring-offset-2 ring-offset-base shadow-[0_0_24px_rgba(213,199,107,0.24)]";

export function parseAutomationSetupRepositories(value: string): string[] {
  const seen = new Set<string>();
  return value
    .split(/[\n,]+/)
    .map((entry) => entry.trim())
    .filter((entry) => {
      if (!entry || seen.has(entry)) return false;
      seen.add(entry);
      return true;
    });
}

type Frequency = (typeof FREQUENCIES)[number];
type StatusMessage = { kind: "success" | "error"; text: string } | null;

interface AutomationSetupPanelProps {
  draft: AutomationSetupDraft;
  conversationId?: string | null;
  conversationTags?: Record<string, string> | null;
  toolbarPortal?: HTMLElement | null;
  showInlineHeader?: boolean;
  onClose: () => void;
}

function titleCase(value: string): string {
  return value
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 4)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(" ");
}

function deriveName(prompt: string): string {
  const withoutLead = prompt
    .replace(/^create\s+(an?\s+)?automation\s+(that|to)?\s*/i, "")
    .replace(/[.!?].*$/, "")
    .trim();
  const name = titleCase(withoutLead || prompt);
  return name || "New Automation";
}

function toCron(
  time: string,
  frequency: Frequency,
  customSchedule: string,
): string {
  if (frequency === "custom") return customSchedule || DEFAULT_CUSTOM_SCHEDULE;
  if (frequency === "hourly") return "0 * * * *";

  const [hour = "9", minute = "0"] = time.split(":");
  const cronTime = `${Number(minute)} ${Number(hour)}`;
  if (frequency === "weekdays") return `${cronTime} * * 1-5`;
  if (frequency === "weekly") return `${cronTime} * * 1`;
  return `${cronTime} * * *`;
}

function buildStarterPython(prompt: string): string {
  return `import json\nimport os\nimport urllib.request\n\n\ndef fire_callback(status="COMPLETED", error=None):\n    url = os.environ.get("AUTOMATION_CALLBACK_URL", "")\n    if not url:\n        return\n    body = {"status": status, "run_id": os.environ.get("AUTOMATION_RUN_ID", "")}\n    if error:\n        body["error"] = error\n    request = urllib.request.Request(\n        url,\n        data=json.dumps(body).encode(),\n        headers={\n            "Content-Type": "application/json",\n            "Authorization": f"Bearer {os.environ.get('AUTOMATION_CALLBACK_API_KEY', '')}",\n        },\n    )\n    urllib.request.urlopen(request, timeout=10)\n\n\ndef main():\n    prompt = ${JSON.stringify(prompt)}\n    print(f"Automation prompt: {prompt}")\n\n\nif __name__ == "__main__":\n    try:\n        main()\n        fire_callback("COMPLETED")\n    except Exception as exc:\n        fire_callback("FAILED", str(exc))\n        raise\n`;
}

function sortFieldsByRenderOrder(
  fields: AutomationSetupField[],
): AutomationSetupField[] {
  return [...fields].sort(
    (first, second) =>
      AUTOMATION_SETUP_FIELD_RENDER_ORDER.indexOf(first) -
      AUTOMATION_SETUP_FIELD_RENDER_ORDER.indexOf(second),
  );
}

function shouldCharacterStreamField(
  field: AutomationSetupField,
  value: AutomationSetupFormValues[AutomationSetupField],
): value is string {
  return typeof value === "string" && !NON_CHARACTER_STREAM_FIELDS.has(field);
}

function streamingHighlightClassName(isStreaming: boolean) {
  return isStreaming ? streamingFieldHighlightClassName : undefined;
}

function buildDefaultEventTestPayload(
  eventSource: string = DEFAULT_EVENT_SOURCE,
  eventKey: string = DEFAULT_EVENT_KEY,
): string {
  return `${JSON.stringify(
    {
      source: eventSource || DEFAULT_EVENT_SOURCE,
      event: eventKey || DEFAULT_EVENT_KEY,
      action: "test",
    },
    null,
    2,
  )}\n`;
}

function parseEventTestPayload(value: string): Record<string, unknown> | null {
  const trimmed = value.trim();
  if (!trimmed) return null;
  try {
    const parsed = JSON.parse(trimmed) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return null;
    }
    return parsed as Record<string, unknown>;
  } catch {
    return null;
  }
}

function buildInitialForm(
  draft: AutomationSetupDraft,
): AutomationSetupFormValues {
  const form = draft.form ?? {};
  const prompt = form.prompt ?? draft.prompt;
  const kind = form.kind ?? draft.kind;
  return {
    kind,
    name: form.name ?? deriveName(prompt),
    prompt,
    repository: form.repository ?? "",
    pluginSource: form.pluginSource ?? draft.plugins?.[0] ?? "",
    pluginRef: form.pluginRef ?? "",
    customCode: form.customCode ?? buildStarterPython(prompt),
    entrypoint: form.entrypoint ?? DEFAULT_CUSTOM_ENTRYPOINT,
    setupScriptPath: form.setupScriptPath ?? DEFAULT_CUSTOM_SETUP_SCRIPT_PATH,
    setupScript: form.setupScript ?? DEFAULT_CUSTOM_SETUP_SCRIPT,
    triggerKind: form.triggerKind ?? "cron",
    frequency: form.frequency ?? "daily",
    time: form.time ?? DEFAULT_TIME,
    timezone: form.timezone ?? DEFAULT_TIMEZONE,
    customSchedule: form.customSchedule ?? DEFAULT_CUSTOM_SCHEDULE,
    eventSource: form.eventSource ?? DEFAULT_EVENT_SOURCE,
    eventKey: form.eventKey ?? DEFAULT_EVENT_KEY,
    eventFilter: form.eventFilter ?? "",
    showTimeout: form.showTimeout ?? false,
    timeoutSeconds: form.timeoutSeconds ?? DEFAULT_TIMEOUT_SECONDS,
  };
}

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

function formFromServerDraft(
  saved: AutomationDraftApiResponse,
  fallback: AutomationSetupDraft,
): AutomationSetupFormValues {
  const base = buildInitialForm(fallback);
  const body = saved.draft as Record<string, unknown>;
  const trigger = asRecord(body.trigger);
  const plugin = getFirstObject(body, "plugins");
  const repo = getFirstObject(body, "repos");
  const endpointKind: AutomationSetupKind =
    saved.endpoint === "/v1"
      ? "custom"
      : saved.endpoint === "/v1/preset/plugin"
        ? "plugin"
        : "prompt";

  return {
    ...base,
    kind: endpointKind,
    name: saved.name ?? getStringField(body, "name") ?? base.name,
    prompt: getStringField(body, "prompt") ?? base.prompt,
    repository:
      getStringField(repo ?? {}, "url") ??
      (typeof body.repository === "string" ? body.repository : base.repository),
    pluginSource: getStringField(plugin ?? {}, "source") ?? base.pluginSource,
    pluginRef: getStringField(plugin ?? {}, "ref") ?? base.pluginRef,
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

/**
 * The server-backed draft endpoint the current kind posts to. Mirrors the
 * service's `DraftEndpoint` literal: the raw `"/v1"` path for custom bundles
 * (which ship their own tarball), and the preset paths otherwise.
 */
function draftEndpoint(kind: AutomationSetupKind): AutomationDraftEndpoint {
  const path = getAutomationEndpoint(endpointName(kind));
  if (
    path === "/v1" ||
    path === "/v1/preset/prompt" ||
    path === "/v1/preset/plugin"
  ) {
    return path;
  }
  // A manifest that remaps the endpoint is not expected for the assisted
  // flow; fall back to the preset prompt path so a draft can still be saved.
  return kind === "plugin" ? "/v1/preset/plugin" : "/v1/preset/prompt";
}

/**
 * Pull field-addressed validation errors out of a 422 dispatch response. The
 * service answers an undispatchable draft with `{ message, errors }`; older
 * transport errors surface as a thrown Error instead.
 */
function getResponseStatus(error: unknown): number | null {
  // Draft calls reach over two transports: local backends go through axios
  // (status under `response`), while Cloud backends go through callCloudProxy
  // (the shared client's HttpError carries `status` directly).
  if (isSdkHttpError(error)) return (error as { status: number }).status;
  if (!error || typeof error !== "object") return null;
  const response = (error as Record<string, unknown>).response;
  if (!response || typeof response !== "object") return null;
  const status = (response as Record<string, unknown>).status;
  return typeof status === "number" ? status : null;
}

function isDraftEndpointUnavailable(error: unknown): boolean {
  const status = getResponseStatus(error);
  return status === 404 || status === 405;
}

function extractDraftDispatchErrors(error: unknown): string | null {
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

function getDraftExecutionStatusText(
  draft: AutomationDraftApiResponse,
  runs: AutomationRun[],
  t: ReturnType<typeof useTranslation>["t"],
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

function DraftRunDetailsCard({
  draft,
  runs,
}: {
  draft: AutomationDraftApiResponse;
  runs: AutomationRun[];
}) {
  const { t, i18n } = useTranslation("openhands");
  const validationMessage = draft.validationErrors?.[0]?.message ?? null;
  const statusText = getDraftExecutionStatusText(draft, runs, t);

  return (
    <section
      data-testid="automation-setup-draft-details"
      className="rounded-2xl border border-[var(--oh-border)] bg-[var(--oh-surface)]"
    >
      <div className="border-b border-[var(--oh-border)] px-5 py-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-content">
              {t(I18nKey.AUTOMATION_SETUP$DRAFT_STATUS_TITLE)}
            </h3>
            <p className="mt-1 text-sm text-muted">
              {t(I18nKey.AUTOMATION_SETUP$DRAFT_STATUS_DESCRIPTION)}
            </p>
          </div>
          <span
            data-testid="automation-setup-draft-validity"
            className={cn(
              "rounded-full px-3 py-1 text-xs",
              validationMessage
                ? "bg-[var(--oh-warning)]/10 text-[var(--oh-warning)]"
                : "bg-[var(--oh-success)]/10 text-[var(--oh-success)]",
            )}
          >
            {statusText}
          </span>
        </div>
        <dl className="mt-4 grid gap-3 text-sm md:grid-cols-2">
          <div>
            <dt className="text-xs text-muted">
              {t(I18nKey.AUTOMATION_SETUP$LAST_SAVED)}
            </dt>
            <dd className="mt-1 text-content">
              {formatRelativeTime(
                draft.updatedAt,
                i18n?.language ?? AvailableLanguages[0].value,
                t,
              )}
            </dd>
          </div>
          {draft.materializedAutomationId ? (
            <div>
              <dt className="text-xs text-muted">
                {t(I18nKey.AUTOMATION_SETUP$TEST_AUTOMATION)}
              </dt>
              <dd className="mt-1 truncate font-mono text-xs text-content">
                {draft.materializedAutomationId}
              </dd>
            </div>
          ) : null}
        </dl>
      </div>

      <div className="px-5 py-3">
        <h4 className="text-sm font-medium text-content">
          {t(I18nKey.AUTOMATION_SETUP$TEST_RUNS)}
        </h4>
      </div>
      {runs.length > 0 ? (
        <div>
          {runs.map((run, index) => (
            <div
              key={run.id}
              data-testid="automation-setup-draft-run"
              className={cn(
                "border-t border-[var(--oh-border)]",
                index === 0 && "bg-[var(--oh-focus)]/10",
              )}
            >
              <ActivityLogItem run={run} />
            </div>
          ))}
        </div>
      ) : (
        <p className="border-t border-[var(--oh-border)] px-5 py-6 text-sm text-muted">
          {t(I18nKey.AUTOMATIONS$DETAIL$NO_RUNS)}
        </p>
      )}
    </section>
  );
}

function kindLabelKey(kind: AutomationSetupKind): I18nKey {
  if (kind === "plugin") return I18nKey.AUTOMATION_SETUP$TYPE_PLUGIN;
  if (kind === "custom") return I18nKey.AUTOMATION_SETUP$TYPE_CUSTOM;
  return I18nKey.AUTOMATION_SETUP$TYPE_PROMPT;
}

function frequencyLabelKey(frequency: Frequency): I18nKey {
  switch (frequency) {
    case "hourly":
      return I18nKey.AUTOMATION_SETUP$FREQUENCY_HOURLY;
    case "daily":
      return I18nKey.AUTOMATIONS$FREQUENCY_DAILY;
    case "weekdays":
      return I18nKey.AUTOMATION_SETUP$FREQUENCY_WEEKDAYS;
    case "weekly":
      return I18nKey.AUTOMATION_SETUP$FREQUENCY_WEEKLY;
    case "custom":
      return I18nKey.AUTOMATION_SETUP$TYPE_CUSTOM;
  }
}

export function AutomationSetupPanel({
  draft,
  conversationId,
  conversationTags,
  toolbarPortal,
  showInlineHeader = true,
  onClose,
}: AutomationSetupPanelProps) {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const deploymentCapabilities = useDeploymentCapabilities();
  const eventSourceOptions = deploymentCapabilities.data?.eventSources ?? [];
  const eventTypeOptions = deploymentCapabilities.data?.eventTypes ?? [];
  const [form, setForm] = useState(() => buildInitialForm(draft));
  const [eventTestPayload, setEventTestPayload] = useState(() =>
    buildDefaultEventTestPayload(DEFAULT_EVENT_SOURCE, DEFAULT_EVENT_KEY),
  );
  const [isEventTestPayloadDirty, setIsEventTestPayloadDirty] = useState(false);
  const [customWebhook, setCustomWebhook] = useState<CustomWebhookFormState>({
    enabled: false,
    name: "",
    eventKeyExpr: DEFAULT_CUSTOM_WEBHOOK_EVENT_KEY_EXPR,
    signatureHeader: DEFAULT_CUSTOM_WEBHOOK_SIGNATURE_HEADER,
    signatureScheme: DEFAULT_CUSTOM_WEBHOOK_SIGNATURE_SCHEME,
    webhookSecret: "",
  });
  const [customWebhookRegistration, setCustomWebhookRegistration] =
    useState<CustomWebhookCreateResponse | null>(null);

  const [fieldMetadata, setFieldMetadata] = useState(
    () => draft.fieldMetadata ?? {},
  );
  const [statusMessage, setStatusMessage] = useState<StatusMessage>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [streamingField, setStreamingField] =
    useState<AutomationSetupField | null>(null);
  // Server-backed draft from OpenHands/automation PR #439. Null until the
  // first save creates the draft row; the local store stays the reactive layer
  // the agent streams into, and the server row is the persisted source of truth.
  const [serverDraft, setServerDraft] =
    useState<AutomationDraftApiResponse | null>(null);
  const [draftRuns, setDraftRuns] = useState<AutomationRun[]>([]);
  const [isHydratingServerDraft, setIsHydratingServerDraft] = useState(false);
  const [saveState, setSaveState] = useState<
    "idle" | "saving" | "saved" | "error"
  >("idle");
  const [isTaggedDraftMissing, setIsTaggedDraftMissing] = useState(false);
  const propTaggedServerDraftId =
    getAutomationDraftIdFromTags(conversationTags);
  const [currentTaggedServerDraftId, setCurrentTaggedServerDraftId] = useState(
    propTaggedServerDraftId,
  );
  const taggedServerDraftId = currentTaggedServerDraftId;
  const serverDraftId =
    serverDraft?.id ?? (isTaggedDraftMissing ? null : taggedServerDraftId);
  const streamQueueRef = useRef<
    {
      field: AutomationSetupField;
      value: AutomationSetupFormValues[AutomationSetupField];
      metadata?: NonNullable<
        AutomationSetupDraft["fieldMetadata"]
      >[AutomationSetupField];
    }[]
  >([]);
  const isStreamProcessingRef = useRef(false);
  const streamGenerationRef = useRef(0);
  const streamTimeoutsRef = useRef<number[]>([]);
  const processQueuedStreamsRef = useRef<() => void>(() => {});

  useEffect(() => {
    setCurrentTaggedServerDraftId(propTaggedServerDraftId);
  }, [propTaggedServerDraftId]);

  useEffect(() => {
    if (!conversationId || hasAutomationSetupModeTag(conversationTags)) return;
    AgentServerConversationService.updateConversationTags(
      conversationId,
      buildAutomationSetupModeTags(conversationTags),
    ).catch((error: unknown) => {
      displayErrorToast(error instanceof Error ? error.message : null);
    });
  }, [conversationId, conversationTags]);

  const updateConversationDraftTags = useCallback(
    async (draftId: string | null) => {
      if (!conversationId) return;
      const nextTags = draftId
        ? buildAutomationDraftTags(conversationTags, draftId)
        : removeAutomationDraftTags(conversationTags);
      await AgentServerConversationService.updateConversationTags(
        conversationId,
        nextTags,
      );
      setCurrentTaggedServerDraftId(draftId);
    },
    [conversationId, conversationTags],
  );

  useEffect(() => {
    if (!taggedServerDraftId || serverDraft?.id === taggedServerDraftId) {
      return undefined;
    }

    let cancelled = false;
    setIsHydratingServerDraft(true);
    setIsTaggedDraftMissing(false);

    AutomationService.getServerDraft(taggedServerDraftId)
      .then((saved) => {
        if (cancelled) return;
        setServerDraft(saved);
        setDraftRuns([]);
        setForm(formFromServerDraft(saved, draft));
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        if (getResponseStatus(error) === 404) {
          setServerDraft(null);
          setIsTaggedDraftMissing(true);
          return;
        }
        displayErrorToast(error instanceof Error ? error.message : null);
      })
      .finally(() => {
        if (!cancelled) setIsHydratingServerDraft(false);
      });

    return () => {
      cancelled = true;
    };
  }, [draft, serverDraft?.id, taggedServerDraftId]);

  const {
    kind,
    name,
    prompt,
    repository,
    pluginSource,
    pluginRef,
    customCode,
    entrypoint,
    setupScriptPath,
    setupScript,
    triggerKind,
    frequency,
    time,
    timezone,
    customSchedule,
    eventSource,
    eventKey,
    eventFilter,
    showTimeout,
    timeoutSeconds,
  } = form;

  useEffect(() => {
    if (isEventTestPayloadDirty) return;
    setEventTestPayload(buildDefaultEventTestPayload(eventSource, eventKey));
  }, [eventSource, eventKey, isEventTestPayloadDirty]);

  const clearQueuedStreams = useCallback(() => {
    streamGenerationRef.current += 1;
    for (const timeoutId of streamTimeoutsRef.current) {
      window.clearTimeout(timeoutId);
    }
    streamTimeoutsRef.current = [];
    streamQueueRef.current = [];
    isStreamProcessingRef.current = false;
    setStreamingField(null);
  }, []);

  const scheduleStreamStep = useCallback(
    (callback: () => void, delay: number, generation: number) => {
      const timeoutId = window.setTimeout(() => {
        streamTimeoutsRef.current = streamTimeoutsRef.current.filter(
          (queuedTimeoutId) => queuedTimeoutId !== timeoutId,
        );
        if (generation !== streamGenerationRef.current) return;
        callback();
      }, delay);
      streamTimeoutsRef.current.push(timeoutId);
    },
    [],
  );

  const finishCurrentStream = useCallback(
    (
      field: AutomationSetupField,
      metadata:
        | NonNullable<
            AutomationSetupDraft["fieldMetadata"]
          >[AutomationSetupField]
        | undefined,
      generation: number,
    ) => {
      if (metadata) {
        setFieldMetadata((previous) => ({ ...previous, [field]: metadata }));
      }
      isStreamProcessingRef.current = false;
      scheduleStreamStep(
        () => processQueuedStreamsRef.current(),
        AGENT_FIELD_STREAM_SETTLE_DELAY_MS,
        generation,
      );
    },
    [scheduleStreamStep],
  );

  const processQueuedStreams = useCallback(() => {
    if (isStreamProcessingRef.current) return;
    const nextStream = streamQueueRef.current.shift();
    if (!nextStream) {
      setStreamingField(null);
      return;
    }

    const generation = streamGenerationRef.current;
    isStreamProcessingRef.current = true;
    setStreamingField(nextStream.field);

    if (!shouldCharacterStreamField(nextStream.field, nextStream.value)) {
      setForm((previous) => ({
        ...previous,
        [nextStream.field]: nextStream.value,
      }));
      finishCurrentStream(nextStream.field, nextStream.metadata, generation);
      return;
    }

    const streamValue = nextStream.value;
    setForm((previous) => ({ ...previous, [nextStream.field]: "" }));
    let nextCharacterIndex = 0;
    const streamNextCharacter = () => {
      nextCharacterIndex += 1;
      setForm((previous) => ({
        ...previous,
        [nextStream.field]: streamValue.slice(0, nextCharacterIndex),
      }));
      if (nextCharacterIndex < streamValue.length) {
        scheduleStreamStep(
          streamNextCharacter,
          AGENT_FIELD_STREAM_CHARACTER_DELAY_MS,
          generation,
        );
        return;
      }
      finishCurrentStream(nextStream.field, nextStream.metadata, generation);
    };

    scheduleStreamStep(
      streamNextCharacter,
      AGENT_FIELD_STREAM_CHARACTER_DELAY_MS,
      generation,
    );
  }, [finishCurrentStream, scheduleStreamStep]);

  processQueuedStreamsRef.current = processQueuedStreams;

  useEffect(
    () => () => {
      clearQueuedStreams();
    },
    [clearQueuedStreams],
  );

  useEffect(() => {
    if (!conversationId) return undefined;
    return subscribeAutomationSetupDraft(
      conversationId,
      (nextDraft, result) => {
        if (!nextDraft) return;
        const nextForm = buildInitialForm(nextDraft);
        const nextMetadata = nextDraft.fieldMetadata ?? {};
        const agentFields = sortFieldsByRenderOrder(
          (result?.applied ?? []).filter(
            (field) => nextMetadata[field]?.updatedBy === "agent",
          ),
        );

        if (agentFields.length === 0) {
          clearQueuedStreams();
          setForm(nextForm);
          setFieldMetadata(nextMetadata);
          return;
        }

        const protectedFields = new Set<AutomationSetupField>([
          ...agentFields,
          ...streamQueueRef.current.map((queuedStream) => queuedStream.field),
          ...(streamingField ? [streamingField] : []),
        ]);
        setForm((previous) => {
          const syncedForm = { ...nextForm };
          for (const field of protectedFields) {
            (syncedForm as Record<string, unknown>)[field] = previous[field];
          }
          return syncedForm;
        });
        setFieldMetadata((previous) => {
          const syncedMetadata = { ...nextMetadata };
          for (const field of agentFields) {
            if (previous[field]) {
              syncedMetadata[field] = previous[field];
            } else {
              delete syncedMetadata[field];
            }
          }
          return syncedMetadata;
        });
        streamQueueRef.current.push(
          ...agentFields.map((field) => ({
            field,
            value: nextForm[field],
            metadata: nextMetadata[field],
          })),
        );
        processQueuedStreamsRef.current();
      },
    );
  }, [clearQueuedStreams, conversationId, streamingField]);

  const updateField = <FieldName extends AutomationSetupField>(
    field: FieldName,
    value: AutomationSetupFormValues[FieldName],
  ) => {
    clearQueuedStreams();
    setStatusMessage(null);
    setSaveState("idle");
    setForm((previous) => ({ ...previous, [field]: value }));
    if (!conversationId) return;
    patchAutomationSetupDraft(
      conversationId,
      { [field]: value } as AutomationSetupFormPatch,
      { source: "user" },
    );
  };

  const agentUpdatedSuffix = (field: AutomationSetupField) =>
    fieldMetadata[field]?.updatedBy === "agent"
      ? t(I18nKey.AUTOMATION_SETUP$FILLED_BY_OPENHANDS)
      : undefined;

  const updateCustomWebhook = <FieldName extends keyof CustomWebhookFormState>(
    field: FieldName,
    value: CustomWebhookFormState[FieldName],
  ) => {
    setStatusMessage(null);
    setSaveState("idle");
    setCustomWebhookRegistration(null);
    setCustomWebhook((previous) => ({ ...previous, [field]: value }));
  };

  const normalizedName = () => name.trim() || deriveName(prompt);
  const buildTrigger = () =>
    triggerKind === "event"
      ? {
          type: "event",
          source: eventSource.trim() || DEFAULT_EVENT_SOURCE,
          on: eventKey.trim() || DEFAULT_EVENT_KEY,
          ...(eventFilter.trim() ? { filter: eventFilter.trim() } : {}),
        }
      : {
          type: "cron",
          schedule: toCron(time, frequency, customSchedule.trim()),
          timezone: timezone.trim() || DEFAULT_TIMEZONE,
        };
  const buildPresetBody = (): SetupRequestBody => {
    const body: SetupRequestBody = {
      name: normalizedName(),
      prompt: prompt.trim(),
      trigger: buildTrigger(),
      enabled: false,
    } as SetupRequestBody;
    const repositories = parseAutomationSetupRepositories(repository);
    if (repositories.length > 0) {
      body.repos = repositories.map((url) => ({ url, provider: "github" }));
    }
    if (showTimeout && timeoutSeconds.trim())
      body.timeout = Number(timeoutSeconds);
    if (kind === "plugin") {
      body.plugins = [
        {
          source: pluginSource.trim(),
          ...(pluginRef.trim() ? { ref: pluginRef.trim() } : {}),
        },
      ];
    }
    return body;
  };
  const buildCustomBody = (tarballPath: string): SetupRequestBody =>
    ({
      name: normalizedName(),
      trigger: buildTrigger(),
      tarball_path: tarballPath,
      entrypoint: entrypoint.trim(),
      setup_script_path: setupScriptPath.trim(),
      ...(showTimeout && timeoutSeconds.trim()
        ? { timeout: Number(timeoutSeconds) }
        : {}),
    }) as SetupRequestBody;
  /**
   * The request body sent to a server-backed draft. Custom drafts reference a
   * tarball path, but a draft is saved before the upload happens, so the
   * preflight stand-in path stands in until dispatch. Preset drafts carry the
   * same body the final create call would, minus the upload-only fields.
   */
  const draftRequestBody = (
    tarballPath: string = PREFLIGHT_TARBALL_PATH,
  ): SetupRequestBody =>
    kind === "custom" ? buildCustomBody(tarballPath) : buildPresetBody();
  const draftValidationByField = useMemo(() => {
    const byField = new Map<string, string>();
    for (const error of serverDraft?.validationErrors ?? []) {
      if (error.field && !byField.has(error.field)) {
        byField.set(error.field, error.message);
      }
    }
    return byField;
  }, [serverDraft?.validationErrors]);
  const fieldError = (field: string): string | undefined =>
    draftValidationByField.get(field);
  const isDraftDirty = useMemo(() => {
    if (!serverDraft) return true;
    return (
      serverDraft.endpoint !== draftEndpoint(kind) ||
      (serverDraft.name ?? "") !== normalizedName() ||
      JSON.stringify(serverDraft.draft) !== JSON.stringify(draftRequestBody())
    );
  }, [
    serverDraft,
    kind,
    name,
    prompt,
    repository,
    pluginSource,
    pluginRef,
    customCode,
    entrypoint,
    setupScriptPath,
    setupScript,
    triggerKind,
    frequency,
    time,
    timezone,
    customSchedule,
    eventSource,
    eventKey,
    eventFilter,
    showTimeout,
    timeoutSeconds,
  ]);

  const ensureCustomWebhookSource = async (): Promise<boolean> => {
    if (triggerKind !== "event" || !customWebhook.enabled) return true;
    const source = eventSource.trim();
    if (!source) {
      setSaveState("error");
      setStatusMessage({
        kind: "error",
        text: t(I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_SOURCE_REQUIRED),
      });
      return false;
    }
    if (customWebhookRegistration?.source === source) return true;

    try {
      const webhook = await AutomationService.createCustomWebhook({
        name:
          customWebhook.name.trim() ||
          t(I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_DEFAULT_NAME, { source }),
        source,
        event_key_expr:
          customWebhook.eventKeyExpr.trim() ||
          DEFAULT_CUSTOM_WEBHOOK_EVENT_KEY_EXPR,
        signature_header:
          customWebhook.signatureHeader.trim() ||
          DEFAULT_CUSTOM_WEBHOOK_SIGNATURE_HEADER,
        signature_scheme: customWebhook.signatureScheme,
        ...(customWebhook.webhookSecret.trim()
          ? { webhook_secret: customWebhook.webhookSecret.trim() }
          : {}),
      });
      setCustomWebhookRegistration(webhook);
      return true;
    } catch (error) {
      if (getResponseStatus(error) === 409) {
        setStatusMessage({
          kind: "success",
          text: t(I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_ALREADY_EXISTS, {
            source,
          }),
        });
        return true;
      }
      setSaveState("error");
      displayErrorToast(error instanceof Error ? error.message : null);
      return false;
    }
  };

  const uploadCustomArchive = async (): Promise<string> => {
    const archive = await packTarGzip([
      { name: MAIN_PY_FILENAME, content: customCode, mode: 0o644 },
      {
        name: setupScriptPath.trim(),
        content: setupScript,
        mode: 0o755,
      },
    ]);
    return AutomationService.uploadAutomationTarball(normalizedName(), archive);
  };
  const persistServerDraft = async (
    tarballPath?: string,
  ): Promise<AutomationDraftApiResponse> => {
    const body = draftRequestBody(tarballPath);
    const request = {
      endpoint: draftEndpoint(kind),
      draft: body,
      name: normalizedName(),
    };
    const saved = serverDraftId
      ? await AutomationService.updateServerDraft(serverDraftId, request)
      : await AutomationService.createServerDraft(request);
    setServerDraft(saved);
    setIsTaggedDraftMissing(false);
    await updateConversationDraftTags(saved.id);
    return saved;
  };
  const runPreflightValidation = async () => {
    const result = await AutomationService.validateDraft({
      endpoint: getAutomationEndpoint(endpointName(kind)),
      draft:
        kind === "custom"
          ? buildCustomBody(PREFLIGHT_TARBALL_PATH)
          : buildPresetBody(),
    });
    setStatusMessage({
      kind: result.valid ? "success" : "error",
      text: result.valid
        ? t(I18nKey.AUTOMATION_SETUP$READY_TO_TEST)
        : result.errors[0]?.message || t(I18nKey.SETUP$SUBMIT_FAILED),
    });
  };
  const validateRequiredFields = (): boolean => {
    if (!prompt.trim() && kind !== "custom") {
      setStatusMessage({
        kind: "error",
        text: t(I18nKey.AUTOMATION_SETUP$PROMPT_REQUIRED),
      });
      return false;
    }
    if (kind === "plugin" && !pluginSource.trim()) {
      setStatusMessage({
        kind: "error",
        text: t(I18nKey.AUTOMATION_SETUP$PLUGIN_REQUIRED),
      });
      return false;
    }
    if (kind === "custom" && !customCode.trim()) {
      setStatusMessage({
        kind: "error",
        text: t(I18nKey.AUTOMATION_SETUP$CODE_REQUIRED),
      });
      return false;
    }
    if (kind === "custom" && !entrypoint.trim()) {
      setStatusMessage({
        kind: "error",
        text: t(I18nKey.AUTOMATION_SETUP$ENTRYPOINT_REQUIRED),
      });
      return false;
    }
    if (kind === "custom" && !setupScriptPath.trim()) {
      setStatusMessage({
        kind: "error",
        text: t(I18nKey.AUTOMATION_SETUP$SETUP_SCRIPT_PATH_REQUIRED),
      });
      return false;
    }
    if (kind === "custom" && !setupScript.trim()) {
      setStatusMessage({
        kind: "error",
        text: t(I18nKey.AUTOMATION_SETUP$SETUP_SCRIPT_REQUIRED),
      });
      return false;
    }
    return true;
  };
  const handleSaveDraft = async () => {
    setIsSubmitting(true);
    setSaveState("saving");
    try {
      if (!(await ensureCustomWebhookSource())) return;
      const saved = await persistServerDraft();
      setSaveState("saved");
      setStatusMessage({
        kind: "success",
        text:
          saved.validationErrors?.[0]?.message ??
          t(I18nKey.AUTOMATION_SETUP$DRAFT_SAVED),
      });
    } catch (error) {
      if (isDraftEndpointUnavailable(error)) {
        setSaveState("saved");
        setStatusMessage({
          kind: "success",
          text: t(I18nKey.AUTOMATION_SETUP$DRAFT_SAVED),
        });
        return;
      }
      setSaveState("error");
      displayErrorToast(error instanceof Error ? error.message : null);
    } finally {
      setIsSubmitting(false);
    }
  };
  const handleTest = async () => {
    if (!validateRequiredFields()) return;
    const eventPayload =
      triggerKind === "event" ? parseEventTestPayload(eventTestPayload) : null;
    if (triggerKind === "event" && eventPayload === null) {
      setStatusMessage({
        kind: "error",
        text: t(I18nKey.AUTOMATION_SETUP$TEST_EVENT_PAYLOAD_INVALID),
      });
      return;
    }
    setIsSubmitting(true);
    setSaveState("saving");
    try {
      if (!(await ensureCustomWebhookSource())) return;
      // Persist the current form state as a draft first, then dispatch it.
      // The service materializes the validated draft body into a disabled
      // automation and starts a manual run; the draft row stays as source
      // of truth for further edits.
      const tarballPath =
        kind === "custom" ? await uploadCustomArchive() : undefined;
      const saved = await persistServerDraft(tarballPath);
      setSaveState("saved");

      if (!saved.dispatchable) {
        setStatusMessage({
          kind: "error",
          text:
            saved.validationErrors?.[0]?.message ??
            t(I18nKey.SETUP$SUBMIT_FAILED),
        });
        return;
      }

      const run = eventPayload
        ? await AutomationService.dispatchServerDraft(saved.id, {
            eventPayload,
          })
        : await AutomationService.dispatchServerDraft(saved.id);
      const materializedAutomationId =
        typeof (run as unknown as Record<string, unknown>).automation_id ===
        "string"
          ? String((run as unknown as Record<string, unknown>).automation_id)
          : saved.materializedAutomationId;
      setServerDraft({
        ...saved,
        materializedAutomationId,
        lastTestRunId: run.id,
      });
      setDraftRuns((previous) => [
        run,
        ...previous.filter((existing) => existing.id !== run.id),
      ]);
      setStatusMessage({
        kind: "success",
        text: t(I18nKey.AUTOMATION_SETUP$TEST_DISPATCHED),
      });
    } catch (error) {
      if (isDraftEndpointUnavailable(error)) {
        setSaveState("saved");
        await runPreflightValidation();
        return;
      }
      setSaveState("error");
      const dispatchError = extractDraftDispatchErrors(error);
      if (dispatchError) {
        setStatusMessage({
          kind: "error",
          text: dispatchError,
        });
      } else {
        displayErrorToast(error instanceof Error ? error.message : null);
      }
    } finally {
      setIsSubmitting(false);
    }
  };
  const handleCreate = async () => {
    if (!validateRequiredFields()) return;
    setIsSubmitting(true);
    setSaveState(serverDraftId ? "saving" : saveState);
    try {
      if (!(await ensureCustomWebhookSource())) return;
      let created: Record<string, unknown>;
      if (kind === "custom") {
        const tarballPath = await uploadCustomArchive();
        created = await AutomationService.createAutomationDraft(
          buildCustomBody(tarballPath),
          kind,
        );
      } else {
        created = await AutomationService.createAutomationDraft(
          buildPresetBody(),
          kind,
        );
      }
      setSaveState("saved");
      toast.success(t(I18nKey.AUTOMATION_SETUP$CREATED));
      // The draft has been finalized into a real automation; drop the
      // persisted draft row so it does not linger as an incomplete setup.
      if (serverDraftId) {
        try {
          await AutomationService.deleteServerDraft(serverDraftId);
        } catch {
          // Cleanup is best-effort; the automation was created either way.
        }
        try {
          await updateConversationDraftTags(null);
        } catch {
          // Tag cleanup is best-effort once the automation exists.
        }
      }
      if (typeof created.id === "string")
        navigate(automationDetailPath(created.id));
    } catch (error) {
      displayErrorToast(error instanceof Error ? error.message : null);
    } finally {
      setIsSubmitting(false);
    }
  };

  useEffect(() => {
    const automationId = serverDraft?.materializedAutomationId;
    if (!automationId) {
      setDraftRuns([]);
      return undefined;
    }

    let cancelled = false;
    AutomationService.listAutomationRuns(automationId, { limit: 10, offset: 0 })
      .then((response) => {
        if (!cancelled && response.runs.length > 0) setDraftRuns(response.runs);
      })
      .catch(() => {
        if (!cancelled) setDraftRuns((previous) => previous);
      });

    return () => {
      cancelled = true;
    };
  }, [serverDraft?.materializedAutomationId]);

  const saveStateLabel = () => {
    if (saveState === "saving") return t(I18nKey.AUTOMATION_SETUP$SAVING);
    if (saveState === "error") return t(I18nKey.AUTOMATION_SETUP$SAVE_FAILED);
    if (serverDraft && !isDraftDirty) {
      return t(I18nKey.AUTOMATION_SETUP$SAVED_JUST_NOW);
    }
    if (isDraftDirty) return t(I18nKey.AUTOMATION_SETUP$UNSAVED_CHANGES);
    return null;
  };

  const renderToolbarActions = () => (
    <div className="flex shrink-0 items-center gap-2">
      <BrandButton
        type="button"
        variant="secondary"
        testId="automation-setup-save-draft"
        isDisabled={isSubmitting}
        onClick={handleSaveDraft}
      >
        {t(I18nKey.AUTOMATION_SETUP$SAVE_DRAFT)}
      </BrandButton>
      <BrandButton
        type="button"
        variant="secondary"
        testId="automation-setup-test"
        isDisabled={isSubmitting}
        onClick={handleTest}
      >
        {isSubmitting
          ? t(I18nKey.AUTOMATION_SETUP$STARTING_TEST)
          : isDraftDirty
            ? t(I18nKey.AUTOMATION_SETUP$SAVE_AND_TEST)
            : t(I18nKey.AUTOMATION_SETUP$TEST_DRAFT)}
      </BrandButton>
      <BrandButton
        type="button"
        variant="primary"
        testId="automation-setup-create"
        isDisabled={isSubmitting}
        onClick={handleCreate}
      >
        {t(I18nKey.AUTOMATIONS$CREATE_AUTOMATION_BUTTON)}
      </BrandButton>
    </div>
  );

  return (
    <>
      {toolbarPortal
        ? createPortal(renderToolbarActions(), toolbarPortal)
        : null}
      <div
        data-testid="automation-setup-panel"
        className="flex h-full min-h-0 flex-col bg-base"
      >
        {showInlineHeader ? (
          <header className="flex h-10 min-h-10 items-center justify-between border-b border-[var(--oh-border)] px-3">
            <div className="flex min-w-0 items-center gap-2">
              <button
                type="button"
                aria-label={t(I18nKey.AUTOMATION_SETUP$BACK_LABEL)}
                onClick={onClose}
                className={cn(
                  "flex size-7 items-center justify-center rounded-lg text-[var(--oh-muted)] hover:bg-white/10 hover:text-white",
                  formControlTransitionClassName,
                )}
              >
                <ArrowLeft className="size-4" aria-hidden />
              </button>
              <h2 className="truncate text-sm font-semibold text-white">
                {t(I18nKey.AUTOMATION_SETUP$TITLE)}
              </h2>
            </div>
            {renderToolbarActions()}
          </header>
        ) : null}

        {saveStateLabel() ? (
          <div
            data-testid="automation-setup-save-state"
            className="border-b border-[var(--oh-border)] px-5 py-2 text-xs text-[var(--oh-muted)]"
          >
            {saveStateLabel()}
          </div>
        ) : null}

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-6">
          <div className="mx-auto flex w-full max-w-5xl flex-col gap-6">
            <div
              role="group"
              aria-label={t(I18nKey.AUTOMATION_SETUP$TYPE_LABEL)}
              className={cn(
                "grid grid-cols-3 gap-2 rounded-xl border border-[var(--oh-border)] bg-base-secondary p-1",
                streamingHighlightClassName(streamingField === "kind"),
              )}
            >
              {AUTOMATION_SETUP_KINDS.map((item) => (
                <button
                  key={item}
                  type="button"
                  aria-pressed={kind === item}
                  data-testid={`automation-setup-kind-${item}`}
                  onClick={() => updateField("kind", item)}
                  className={cn(
                    "flex items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm",
                    formControlTransitionClassName,
                    kind === item
                      ? "bg-white/10 text-white"
                      : "text-[var(--oh-muted)] hover:bg-white/5 hover:text-white",
                  )}
                >
                  {item === "prompt" && (
                    <FileText className="size-4" aria-hidden />
                  )}
                  {item === "plugin" && (
                    <Puzzle className="size-4" aria-hidden />
                  )}
                  {item === "custom" && (
                    <Code2 className="size-4" aria-hidden />
                  )}
                  <span>{t(kindLabelKey(item))}</span>
                </button>
              ))}
            </div>

            <Field
              label={t(I18nKey.AUTOMATIONS$NAME)}
              suffix={agentUpdatedSuffix("name")}
              isStreaming={streamingField === "name"}
              errorText={fieldError("name")}
            >
              <input
                data-testid="automation-setup-name"
                value={name}
                placeholder={t(I18nKey.AUTOMATION_SETUP$NAME_PLACEHOLDER)}
                onChange={(event) => updateField("name", event.target.value)}
                className={formControlFieldClassName}
              />
            </Field>

            {isHydratingServerDraft ? (
              <p
                data-testid="automation-setup-draft-loading"
                className="rounded-2xl border border-[var(--oh-border)] bg-[var(--oh-surface)] px-5 py-4 text-sm text-muted"
              >
                {t(I18nKey.AUTOMATION_SETUP$LOADING_DRAFT)}
              </p>
            ) : null}

            {isTaggedDraftMissing ? (
              <p
                data-testid="automation-setup-draft-missing"
                className="rounded-2xl border border-[var(--oh-warning)]/40 bg-[var(--oh-warning)]/10 px-5 py-4 text-sm text-[var(--oh-warning)]"
              >
                {t(I18nKey.AUTOMATION_SETUP$DRAFT_MISSING)}
              </p>
            ) : null}

            {serverDraft ? (
              <DraftRunDetailsCard draft={serverDraft} runs={draftRuns} />
            ) : null}

            {kind !== "custom" ? (
              <PromptFields
                prompt={prompt}
                updatedSuffix={agentUpdatedSuffix("prompt")}
                isStreaming={streamingField === "prompt"}
                errorText={fieldError("prompt")}
                onPromptChange={(value) => updateField("prompt", value)}
              />
            ) : (
              <CustomCodeFields
                code={customCode}
                entrypoint={entrypoint}
                setupScriptPath={setupScriptPath}
                setupScript={setupScript}
                updatedSuffixes={{
                  customCode: agentUpdatedSuffix("customCode"),
                  entrypoint: agentUpdatedSuffix("entrypoint"),
                  setupScriptPath: agentUpdatedSuffix("setupScriptPath"),
                  setupScript: agentUpdatedSuffix("setupScript"),
                }}
                streamingField={streamingField}
                onCodeChange={(value) => updateField("customCode", value)}
                onEntrypointChange={(value) => updateField("entrypoint", value)}
                onSetupScriptPathChange={(value) =>
                  updateField("setupScriptPath", value)
                }
                onSetupScriptChange={(value) =>
                  updateField("setupScript", value)
                }
              />
            )}

            {kind === "plugin" && (
              <div className="grid gap-3 rounded-xl border border-[var(--oh-border)] bg-base-secondary p-4 md:grid-cols-[2fr_1fr]">
                <Field
                  label={t(I18nKey.AUTOMATION_SETUP$PLUGIN_SOURCE)}
                  suffix={agentUpdatedSuffix("pluginSource")}
                  isStreaming={streamingField === "pluginSource"}
                >
                  <input
                    data-testid="automation-setup-plugin-source"
                    value={pluginSource}
                    placeholder={t(
                      I18nKey.AUTOMATION_SETUP$PLUGIN_SOURCE_PLACEHOLDER,
                    )}
                    onChange={(event) =>
                      updateField("pluginSource", event.target.value)
                    }
                    className={formControlFieldClassName}
                  />
                </Field>
                <Field
                  label={t(I18nKey.AUTOMATION_SETUP$PLUGIN_REF)}
                  suffix={agentUpdatedSuffix("pluginRef")}
                  isStreaming={streamingField === "pluginRef"}
                >
                  <input
                    data-testid="automation-setup-plugin-ref"
                    value={pluginRef}
                    placeholder={t(
                      I18nKey.AUTOMATION_SETUP$PLUGIN_REF_PLACEHOLDER,
                    )}
                    onChange={(event) =>
                      updateField("pluginRef", event.target.value)
                    }
                    className={formControlFieldClassName}
                  />
                </Field>
              </div>
            )}

            {kind !== "custom" && (
              <Field
                label={t(I18nKey.COMMON$REPOSITORIES)}
                suffix={
                  agentUpdatedSuffix("repository") ?? t(I18nKey.COMMON$OPTIONAL)
                }
                isStreaming={streamingField === "repository"}
              >
                <textarea
                  data-testid="automation-setup-repository"
                  value={repository}
                  placeholder={t(I18nKey.SETUP$REPOSITORY_PLACEHOLDER)}
                  onChange={(event) =>
                    updateField("repository", event.target.value)
                  }
                  className={formControlMultilineFieldClassName}
                  rows={2}
                />
              </Field>
            )}

            <section className="flex flex-col gap-3">
              <h3 className="text-sm font-semibold text-white">
                {t(I18nKey.AUTOMATIONS$DETAIL$TRIGGER)}
              </h3>
              <div
                className={cn(
                  "grid gap-3 md:grid-cols-2",
                  streamingHighlightClassName(streamingField === "triggerKind"),
                )}
              >
                <TriggerCard
                  icon={<CalendarDays className="size-4" aria-hidden />}
                  title={t(I18nKey.AUTOMATION_SETUP$SCHEDULE)}
                  description={t(I18nKey.AUTOMATION_SETUP$SCHEDULE_DESCRIPTION)}
                  selected={triggerKind === "cron"}
                  onClick={() => updateField("triggerKind", "cron")}
                />
                <TriggerCard
                  icon={<Zap className="size-4" aria-hidden />}
                  title={t(I18nKey.AUTOMATIONS$DETAIL$TRIGGER_EVENT)}
                  description={t(I18nKey.AUTOMATION_SETUP$EVENT_DESCRIPTION)}
                  selected={triggerKind === "event"}
                  onClick={() => updateField("triggerKind", "event")}
                />
              </div>
            </section>

            {triggerKind === "cron" ? (
              <ScheduleFields
                frequency={frequency}
                time={time}
                timezone={timezone}
                customSchedule={customSchedule}
                updatedSuffixes={{
                  frequency: agentUpdatedSuffix("frequency"),
                  time: agentUpdatedSuffix("time"),
                  timezone: agentUpdatedSuffix("timezone"),
                  customSchedule: agentUpdatedSuffix("customSchedule"),
                }}
                streamingField={streamingField}
                setFrequency={(value) => updateField("frequency", value)}
                setTime={(value) => updateField("time", value)}
                setTimezone={(value) => updateField("timezone", value)}
                setCustomSchedule={(value) =>
                  updateField("customSchedule", value)
                }
              />
            ) : (
              <EventFields
                eventSource={eventSource}
                eventKey={eventKey}
                eventFilter={eventFilter}
                eventTestPayload={eventTestPayload}
                eventSourceOptions={eventSourceOptions}
                eventTypeOptions={eventTypeOptions}
                customWebhook={customWebhook}
                customWebhookRegistration={customWebhookRegistration}
                updatedSuffixes={{
                  eventSource: agentUpdatedSuffix("eventSource"),
                  eventKey: agentUpdatedSuffix("eventKey"),
                  eventFilter: agentUpdatedSuffix("eventFilter"),
                }}
                streamingField={streamingField}
                setEventSource={(value) => updateField("eventSource", value)}
                setEventKey={(value) => updateField("eventKey", value)}
                setEventFilter={(value) => updateField("eventFilter", value)}
                setEventTestPayload={(value) => {
                  setIsEventTestPayloadDirty(true);
                  setEventTestPayload(value);
                }}
                setCustomWebhookField={updateCustomWebhook}
              />
            )}

            <section className="flex flex-col gap-3">
              <h3 className="text-sm font-semibold text-white">
                {t(I18nKey.AUTOMATION_SETUP$ADDITIONAL_OPTIONS)}
              </h3>
              {showTimeout ? (
                <Field
                  label={t(I18nKey.AUTOMATION_SETUP$TIMEOUT_SECONDS)}
                  suffix={agentUpdatedSuffix("timeoutSeconds")}
                  isStreaming={streamingField === "timeoutSeconds"}
                >
                  <input
                    data-testid="automation-setup-timeout"
                    type="number"
                    min="1"
                    value={timeoutSeconds}
                    onChange={(event) =>
                      updateField("timeoutSeconds", event.target.value)
                    }
                    className={formControlFieldClassName}
                  />
                </Field>
              ) : (
                <button
                  type="button"
                  data-testid="automation-setup-add-timeout"
                  onClick={() => updateField("showTimeout", true)}
                  className={cn(
                    "w-fit rounded-full border border-[var(--oh-border)] px-4 py-2 text-sm text-[var(--oh-muted)] hover:bg-white/5 hover:text-white",
                    formControlTransitionClassName,
                    streamingHighlightClassName(
                      streamingField === "showTimeout",
                    ),
                  )}
                >
                  {t(I18nKey.AUTOMATION_SETUP$ADD_TIMEOUT)}
                </button>
              )}
            </section>

            {statusMessage && (
              <p
                role="status"
                data-testid="automation-setup-status"
                className={cn(
                  "text-sm",
                  statusMessage.kind === "success"
                    ? "text-green-400"
                    : "text-red-400",
                )}
              >
                {statusMessage.text}
              </p>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

function PromptFields({
  prompt,
  updatedSuffix,
  isStreaming,
  errorText,
  onPromptChange,
}: {
  prompt: string;
  updatedSuffix?: string;
  isStreaming: boolean;
  errorText?: string;
  onPromptChange: (value: string) => void;
}) {
  const { t } = useTranslation("openhands");
  return (
    <Field
      label={t(I18nKey.AUTOMATIONS$PROMPT)}
      suffix={updatedSuffix}
      isStreaming={isStreaming}
      errorText={errorText}
    >
      <div className="rounded-xl border border-[var(--oh-border)] bg-base-secondary">
        <textarea
          data-testid="automation-setup-prompt"
          rows={7}
          value={prompt}
          onChange={(event) => onPromptChange(event.target.value)}
          className={cn(
            formControlMultilineFieldClassName,
            "min-h-44 resize-none border-0 bg-transparent p-4",
          )}
        />
        <div className="flex items-center justify-between border-t border-[var(--oh-border)] px-4 py-3 text-xs text-[var(--oh-muted)]">
          <span>{t(I18nKey.AUTOMATION_SETUP$MODEL_PLACEHOLDER)}</span>
          <span>{t(I18nKey.AUTOMATION_SETUP$PROMPT_HINT)}</span>
        </div>
      </div>
    </Field>
  );
}

function CustomCodeFields({
  code,
  entrypoint,
  setupScriptPath,
  setupScript,
  updatedSuffixes,
  streamingField,
  onCodeChange,
  onEntrypointChange,
  onSetupScriptPathChange,
  onSetupScriptChange,
}: {
  code: string;
  entrypoint: string;
  setupScriptPath: string;
  setupScript: string;
  updatedSuffixes: Partial<
    Record<
      "customCode" | "entrypoint" | "setupScriptPath" | "setupScript",
      string | undefined
    >
  >;
  streamingField: AutomationSetupField | null;
  onCodeChange: (value: string) => void;
  onEntrypointChange: (value: string) => void;
  onSetupScriptPathChange: (value: string) => void;
  onSetupScriptChange: (value: string) => void;
}) {
  const { t } = useTranslation("openhands");
  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-3 md:grid-cols-[2fr_1fr]">
        <Field
          label={t(I18nKey.AUTOMATION_SETUP$ENTRYPOINT)}
          suffix={updatedSuffixes.entrypoint}
          isStreaming={streamingField === "entrypoint"}
        >
          <input
            data-testid="automation-setup-entrypoint"
            value={entrypoint}
            onChange={(event) => onEntrypointChange(event.target.value)}
            className={formControlFieldClassName}
          />
        </Field>
        <Field
          label={t(I18nKey.AUTOMATION_SETUP$SETUP_SCRIPT_PATH)}
          suffix={updatedSuffixes.setupScriptPath}
          isStreaming={streamingField === "setupScriptPath"}
        >
          <input
            data-testid="automation-setup-setup-script-path"
            value={setupScriptPath}
            onChange={(event) => onSetupScriptPathChange(event.target.value)}
            className={formControlFieldClassName}
          />
        </Field>
      </div>
      <Field
        label={t(I18nKey.AUTOMATION_SETUP$PYTHON_CODE)}
        suffix={updatedSuffixes.customCode}
        isStreaming={streamingField === "customCode"}
      >
        <textarea
          data-testid="automation-setup-custom-code"
          rows={12}
          value={code}
          onChange={(event) => onCodeChange(event.target.value)}
          spellCheck={false}
          className={cn(
            formControlMultilineFieldClassName,
            "font-mono text-xs",
          )}
        />
      </Field>
      <Field
        label={t(I18nKey.AUTOMATION_SETUP$SETUP_SCRIPT)}
        suffix={updatedSuffixes.setupScript}
        isStreaming={streamingField === "setupScript"}
      >
        <textarea
          data-testid="automation-setup-setup-script"
          rows={4}
          value={setupScript}
          onChange={(event) => onSetupScriptChange(event.target.value)}
          spellCheck={false}
          className={cn(
            formControlMultilineFieldClassName,
            "font-mono text-xs",
          )}
        />
      </Field>
    </div>
  );
}

function ScheduleFields({
  frequency,
  time,
  timezone,
  customSchedule,
  updatedSuffixes,
  streamingField,
  setFrequency,
  setTime,
  setTimezone,
  setCustomSchedule,
}: {
  frequency: Frequency;
  time: string;
  timezone: string;
  customSchedule: string;
  updatedSuffixes: Partial<
    Record<
      "frequency" | "time" | "timezone" | "customSchedule",
      string | undefined
    >
  >;
  streamingField: AutomationSetupField | null;
  setFrequency: (value: Frequency) => void;
  setTime: (value: string) => void;
  setTimezone: (value: string) => void;
  setCustomSchedule: (value: string) => void;
}) {
  const { t } = useTranslation("openhands");
  const atUpdatedSuffix = updatedSuffixes.time ?? updatedSuffixes.timezone;
  return (
    <section className="flex flex-col gap-3">
      <h3 className="flex items-center gap-2 text-sm font-semibold text-white">
        <span>{t(I18nKey.AUTOMATION_SETUP$FREQUENCY)}</span>
        {updatedSuffixes.frequency && (
          <span className="text-xs font-normal text-[var(--oh-muted)]">
            {updatedSuffixes.frequency}
          </span>
        )}
      </h3>
      <div
        className={cn(
          "grid grid-cols-2 gap-1 rounded-xl bg-base-secondary p-1 md:grid-cols-6",
          streamingHighlightClassName(streamingField === "frequency"),
        )}
      >
        {FREQUENCIES.map((item) => (
          <button
            key={item}
            type="button"
            data-testid={`automation-setup-frequency-${item}`}
            aria-pressed={frequency === item}
            onClick={() => setFrequency(item)}
            className={cn(
              "rounded-lg px-3 py-2 text-sm",
              formControlTransitionClassName,
              frequency === item
                ? "bg-[var(--oh-interactive-hover)] text-white"
                : "text-[var(--oh-muted)] hover:text-white",
            )}
          >
            {t(frequencyLabelKey(item))}
          </button>
        ))}
      </div>
      {frequency === "custom" ? (
        <Field
          label={t(I18nKey.AUTOMATION_SETUP$TYPE_CUSTOM)}
          suffix={updatedSuffixes.customSchedule}
          isStreaming={streamingField === "customSchedule"}
        >
          <input
            data-testid="automation-setup-custom-schedule"
            value={customSchedule}
            onChange={(event) => setCustomSchedule(event.target.value)}
            className={formControlFieldClassName}
          />
        </Field>
      ) : (
        <div
          data-testid="automation-setup-at-row"
          className="flex flex-col gap-2 md:flex-row md:items-center"
        >
          <span className="shrink-0 text-sm font-semibold text-white">
            {t(I18nKey.AUTOMATION_SETUP$AT)}
          </span>
          <div
            data-streaming-active={
              streamingField === "time" ? "true" : undefined
            }
            className={cn(
              "relative md:w-36",
              streamingHighlightClassName(streamingField === "time"),
            )}
          >
            <input
              aria-label={t(I18nKey.AUTOMATION_SETUP$AT)}
              data-testid="automation-setup-time"
              type="time"
              value={time}
              onChange={(event) => setTime(event.target.value)}
              className={formControlFieldClassName}
            />
            <Clock3
              className="pointer-events-none absolute right-3 top-1/2 size-4 -translate-y-1/2 text-[var(--oh-muted)]"
              aria-hidden
            />
          </div>
          <div
            data-streaming-active={
              streamingField === "timezone" ? "true" : undefined
            }
            className={cn(
              "relative min-w-0 flex-1 md:max-w-96",
              streamingHighlightClassName(streamingField === "timezone"),
            )}
          >
            <input
              aria-label={t(I18nKey.AUTOMATIONS$TIMEZONE)}
              data-testid="automation-setup-timezone"
              value={timezone}
              onChange={(event) => setTimezone(event.target.value)}
              className={cn(formControlFieldClassName, "pl-9")}
            />
            <Globe2
              className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-[var(--oh-muted)]"
              aria-hidden
            />
          </div>
          {atUpdatedSuffix && (
            <span className="shrink-0 text-xs font-normal text-[var(--oh-muted)]">
              {atUpdatedSuffix}
            </span>
          )}
        </div>
      )}
    </section>
  );
}

function EventFields({
  eventSource,
  eventKey,
  eventFilter,
  eventTestPayload,
  eventSourceOptions,
  eventTypeOptions,
  customWebhook,
  customWebhookRegistration,
  updatedSuffixes,
  streamingField,
  setEventSource,
  setEventKey,
  setEventFilter,
  setEventTestPayload,
  setCustomWebhookField,
}: {
  eventSource: string;
  eventKey: string;
  eventFilter: string;
  eventTestPayload: string;
  eventSourceOptions: string[];
  eventTypeOptions: string[];
  customWebhook: CustomWebhookFormState;
  customWebhookRegistration: CustomWebhookCreateResponse | null;
  updatedSuffixes: Partial<
    Record<"eventSource" | "eventKey" | "eventFilter", string | undefined>
  >;
  streamingField: AutomationSetupField | null;
  setEventSource: (value: string) => void;
  setEventKey: (value: string) => void;
  setEventFilter: (value: string) => void;
  setEventTestPayload: (value: string) => void;
  setCustomWebhookField: <FieldName extends keyof CustomWebhookFormState>(
    field: FieldName,
    value: CustomWebhookFormState[FieldName],
  ) => void;
}) {
  const { t } = useTranslation("openhands");
  return (
    <section className="grid gap-3 md:grid-cols-2">
      <Field
        label={t(I18nKey.AUTOMATION_SETUP$EVENT_SOURCE)}
        suffix={updatedSuffixes.eventSource}
        isStreaming={streamingField === "eventSource"}
      >
        <>
          <input
            data-testid="automation-setup-event-source"
            value={eventSource}
            list="automation-setup-event-source-options"
            onChange={(event) => setEventSource(event.target.value)}
            className={formControlFieldClassName}
          />
          <datalist id="automation-setup-event-source-options">
            {eventSourceOptions.map((source) => (
              <option key={source} value={source} />
            ))}
          </datalist>
        </>
      </Field>
      <Field
        label={t(I18nKey.AUTOMATION_SETUP$EVENT_KEY)}
        suffix={updatedSuffixes.eventKey}
        isStreaming={streamingField === "eventKey"}
      >
        <>
          <input
            data-testid="automation-setup-event-key"
            value={eventKey}
            list="automation-setup-event-key-options"
            onChange={(event) => setEventKey(event.target.value)}
            className={formControlFieldClassName}
          />
          <datalist id="automation-setup-event-key-options">
            {eventTypeOptions.map((eventType) => (
              <option key={eventType} value={eventType} />
            ))}
          </datalist>
        </>
      </Field>
      <div className="md:col-span-2">
        <Field
          label={t(I18nKey.AUTOMATION_SETUP$EVENT_FILTER)}
          suffix={updatedSuffixes.eventFilter ?? t(I18nKey.COMMON$OPTIONAL)}
          isStreaming={streamingField === "eventFilter"}
        >
          <input
            data-testid="automation-setup-event-filter"
            value={eventFilter}
            onChange={(event) => setEventFilter(event.target.value)}
            className={formControlFieldClassName}
          />
        </Field>
      </div>
      <div className="md:col-span-2">
        <section className="rounded-xl border border-[var(--oh-border)] p-3">
          <div className="flex items-start gap-3 text-sm font-medium text-white">
            <input
              id="automation-setup-custom-webhook-enabled"
              type="checkbox"
              data-testid="automation-setup-custom-webhook-enabled"
              checked={customWebhook.enabled}
              onChange={(event) =>
                setCustomWebhookField("enabled", event.target.checked)
              }
              className="mt-1"
            />
            <label
              htmlFor="automation-setup-custom-webhook-enabled"
              className="flex flex-col gap-1"
            >
              <span>{t(I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_TITLE)}</span>
              <span className="text-xs font-normal leading-5 text-[var(--oh-muted)]">
                {t(I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_DESCRIPTION)}
              </span>
            </label>
          </div>
          {customWebhook.enabled && (
            <div className="mt-3 grid gap-3 md:grid-cols-2">
              <Field label={t(I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_NAME)}>
                <input
                  data-testid="automation-setup-custom-webhook-name"
                  value={customWebhook.name}
                  onChange={(event) =>
                    setCustomWebhookField("name", event.target.value)
                  }
                  className={formControlFieldClassName}
                  placeholder={t(
                    I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_NAME_PLACEHOLDER,
                  )}
                />
              </Field>
              <Field
                label={t(
                  I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_EVENT_KEY_EXPR,
                )}
              >
                <input
                  data-testid="automation-setup-custom-webhook-event-key-expr"
                  value={customWebhook.eventKeyExpr}
                  onChange={(event) =>
                    setCustomWebhookField("eventKeyExpr", event.target.value)
                  }
                  className={formControlFieldClassName}
                />
              </Field>
              <Field
                label={t(
                  I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_SIGNATURE_HEADER,
                )}
              >
                <input
                  data-testid="automation-setup-custom-webhook-signature-header"
                  value={customWebhook.signatureHeader}
                  onChange={(event) =>
                    setCustomWebhookField("signatureHeader", event.target.value)
                  }
                  className={formControlFieldClassName}
                />
              </Field>
              <Field
                label={t(
                  I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_SIGNATURE_SCHEME,
                )}
              >
                <select
                  data-testid="automation-setup-custom-webhook-signature-scheme"
                  value={customWebhook.signatureScheme}
                  onChange={(event) =>
                    setCustomWebhookField(
                      "signatureScheme",
                      event.target.value as CustomWebhookSignatureScheme,
                    )
                  }
                  className={formControlFieldClassName}
                >
                  <option value="hmac_sha256_hex">
                    {t(
                      I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_SIGNATURE_SCHEME_HMAC,
                    )}
                  </option>
                  <option value="standard_webhooks">
                    {t(
                      I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_SIGNATURE_SCHEME_STANDARD,
                    )}
                  </option>
                  <option value="slack_v0">
                    {t(
                      I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_SIGNATURE_SCHEME_SLACK,
                    )}
                  </option>
                </select>
              </Field>
              <div className="md:col-span-2">
                <Field
                  label={t(I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_SECRET)}
                  suffix={t(I18nKey.COMMON$OPTIONAL)}
                >
                  <input
                    data-testid="automation-setup-custom-webhook-secret"
                    type="password"
                    value={customWebhook.webhookSecret}
                    onChange={(event) =>
                      setCustomWebhookField("webhookSecret", event.target.value)
                    }
                    className={formControlFieldClassName}
                    placeholder={t(
                      I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_SECRET_PLACEHOLDER,
                    )}
                  />
                </Field>
                <p className="mt-2 text-xs leading-5 text-[var(--oh-muted)]">
                  {t(I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_SECRET_HELP)}
                </p>
              </div>
              {customWebhookRegistration && (
                <div
                  data-testid="automation-setup-custom-webhook-result"
                  className="md:col-span-2 rounded-lg border border-[var(--oh-success)]/40 bg-[var(--oh-success)]/10 p-3 text-xs leading-5 text-white"
                >
                  <p className="font-medium">
                    {t(I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_CREATED)}
                  </p>
                  <p className="break-all">
                    {t(I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_URL, {
                      url: customWebhookRegistration.webhook_url,
                    })}
                  </p>
                  {customWebhookRegistration.webhook_secret && (
                    <p className="break-all">
                      {t(
                        I18nKey.AUTOMATION_SETUP$CUSTOM_WEBHOOK_GENERATED_SECRET,
                        {
                          secret: customWebhookRegistration.webhook_secret,
                        },
                      )}
                    </p>
                  )}
                </div>
              )}
            </div>
          )}
        </section>
      </div>
      <div className="md:col-span-2">
        <Field label={t(I18nKey.AUTOMATION_SETUP$TEST_EVENT_PAYLOAD)}>
          <div className="flex flex-col gap-2">
            <textarea
              data-testid="automation-setup-event-test-payload"
              value={eventTestPayload}
              onChange={(event) => setEventTestPayload(event.target.value)}
              className={cn(
                formControlMultilineFieldClassName,
                "min-h-44 font-mono",
              )}
              placeholder={t(
                I18nKey.AUTOMATION_SETUP$TEST_EVENT_PAYLOAD_PLACEHOLDER,
              )}
            />
            <p className="text-xs leading-5 text-[var(--oh-muted)]">
              {t(I18nKey.AUTOMATION_SETUP$TEST_EVENT_PAYLOAD_DESCRIPTION)}
            </p>
          </div>
        </Field>
      </div>
    </section>
  );
}

function Field({
  label,
  suffix,
  horizontal = false,
  isStreaming = false,
  errorText,
  children,
}: {
  label: string;
  suffix?: string;
  horizontal?: boolean;
  isStreaming?: boolean;
  errorText?: string;
  children: ReactNode;
}) {
  return (
    <label
      data-streaming-active={isStreaming ? "true" : undefined}
      className={cn(
        "flex gap-2",
        horizontal ? "items-center" : "flex-col",
        streamingHighlightClassName(isStreaming),
      )}
    >
      <span className="flex items-center gap-2 text-sm font-semibold text-white">
        {label}
        {suffix && (
          <span className="font-normal text-[var(--oh-muted)]">{suffix}</span>
        )}
      </span>
      <div className="min-w-0 flex-1">{children}</div>
      {errorText ? (
        <span
          role="alert"
          className="text-xs leading-5 text-[var(--oh-warning)]"
        >
          {errorText}
        </span>
      ) : null}
    </label>
  );
}

function TriggerCard({
  icon,
  title,
  description,
  selected,
  onClick,
}: {
  icon: ReactNode;
  title: string;
  description: string;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={selected}
      onClick={onClick}
      className={cn(
        "flex items-start gap-3 rounded-xl border p-4 text-left",
        formControlTransitionClassName,
        selected
          ? "border-white/20 bg-white/10 text-white"
          : "border-[var(--oh-border)] bg-base-secondary text-[var(--oh-muted)] hover:bg-white/5 hover:text-white",
      )}
    >
      <span className="mt-0.5 text-[var(--oh-muted)]">{icon}</span>
      <span className="flex flex-col gap-1">
        <span className="text-sm font-semibold">{title}</span>
        <span className="text-xs leading-5 text-[var(--oh-muted)]">
          {description}
        </span>
      </span>
    </button>
  );
}
