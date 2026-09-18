import {
  useCallback,
  useEffect,
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
  Plus,
  Puzzle,
  Zap,
} from "lucide-react";
import AutomationService from "#/api/automation-service/automation-service.api";
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
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import {
  formControlFieldClassName,
  formControlMultilineFieldClassName,
  formControlTransitionClassName,
} from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { useNavigation } from "#/context/navigation-context";
import { useDeploymentCapabilities } from "#/hooks/query/use-manifest-capabilities";
import type {
  AutomationDraftApiResponse,
  AutomationDraftEndpoint,
  InterfaceEndpointName,
  SetupRequestBody,
} from "#/manifests/types";

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
  "once",
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
const NON_CHARACTER_STREAM_FIELDS = new Set<AutomationSetupField>([
  "kind",
  "triggerKind",
  "frequency",
  "showTimeout",
  "time",
]);
const streamingFieldHighlightClassName =
  "rounded-xl ring-2 ring-[#D5C76B]/80 ring-offset-2 ring-offset-base shadow-[0_0_24px_rgba(213,199,107,0.24)]";

type Frequency = (typeof FREQUENCIES)[number];
type StatusMessage = { kind: "success" | "error"; text: string } | null;

interface AutomationSetupPanelProps {
  draft: AutomationSetupDraft;
  conversationId?: string | null;
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

function kindLabelKey(kind: AutomationSetupKind): I18nKey {
  if (kind === "plugin") return I18nKey.AUTOMATION_SETUP$TYPE_PLUGIN;
  if (kind === "custom") return I18nKey.AUTOMATION_SETUP$TYPE_CUSTOM;
  return I18nKey.AUTOMATION_SETUP$TYPE_PROMPT;
}

function frequencyLabelKey(frequency: Frequency): I18nKey {
  switch (frequency) {
    case "once":
      return I18nKey.AUTOMATION_SETUP$FREQUENCY_ONCE;
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
  toolbarPortal,
  showInlineHeader = true,
  onClose,
}: AutomationSetupPanelProps) {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const [form, setForm] = useState(() => buildInitialForm(draft));
  const [fieldMetadata, setFieldMetadata] = useState(
    () => draft.fieldMetadata ?? {},
  );
  const [statusMessage, setStatusMessage] = useState<StatusMessage>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [streamingField, setStreamingField] =
    useState<AutomationSetupField | null>(null);
  // Server-backed draft id from OpenHands/automation PR #417. Null until the
  // first save creates the draft row; the local store stays the reactive layer
  // the agent streams into, and the server row is the persisted source of truth.
  const [serverDraftId, setServerDraftId] = useState<string | null>(null);
  const deploymentCapabilities = useDeploymentCapabilities();
  const serverDraftsSupported = AutomationService.supportsAutomationDrafts(
    deploymentCapabilities.data,
  );
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
    if (repository.trim()) {
      body.repos = [{ url: repository.trim(), provider: "github" }];
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
  const draftRequestBody = (): SetupRequestBody =>
    kind === "custom"
      ? buildCustomBody(PREFLIGHT_TARBALL_PATH)
      : buildPresetBody();
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
    if (!serverDraftsSupported) {
      // Older deployments without the drafts endpoints keep the client-only
      // draft store; there is nothing to persist server-side.
      setStatusMessage({
        kind: "success",
        text: t(I18nKey.AUTOMATION_SETUP$DRAFT_SAVED),
      });
      return;
    }
    setIsSubmitting(true);
    try {
      const body = draftRequestBody();
      let saved: AutomationDraftApiResponse;
      if (serverDraftId) {
        saved = await AutomationService.updateServerDraft(serverDraftId, {
          endpoint: draftEndpoint(kind),
          draft: body,
          name: normalizedName(),
        });
      } else {
        saved = await AutomationService.createServerDraft({
          endpoint: draftEndpoint(kind),
          draft: body,
          name: normalizedName(),
        });
        setServerDraftId(saved.id);
      }
      setStatusMessage({
        kind: "success",
        text:
          saved.validationErrors?.[0]?.message ??
          t(I18nKey.AUTOMATION_SETUP$DRAFT_SAVED),
      });
    } catch (error) {
      displayErrorToast(error instanceof Error ? error.message : null);
    } finally {
      setIsSubmitting(false);
    }
  };
  const handleTest = async () => {
    if (!validateRequiredFields()) return;
    setIsSubmitting(true);
    try {
      if (serverDraftsSupported) {
        // Persist the current form state as a draft first, then dispatch it.
        // The service materializes the validated draft body into a disabled
        // automation and starts a manual run; the draft row stays as source
        // of truth for further edits.
        const body = draftRequestBody();
        let saved: AutomationDraftApiResponse;
        if (serverDraftId) {
          saved = await AutomationService.updateServerDraft(serverDraftId, {
            endpoint: draftEndpoint(kind),
            draft: body,
            name: normalizedName(),
          });
        } else {
          saved = await AutomationService.createServerDraft({
            endpoint: draftEndpoint(kind),
            draft: body,
            name: normalizedName(),
          });
          setServerDraftId(saved.id);
        }

        if (!saved.dispatchable) {
          setStatusMessage({
            kind: "error",
            text:
              saved.validationErrors?.[0]?.message ??
              t(I18nKey.SETUP$SUBMIT_FAILED),
          });
          return;
        }

        const run = await AutomationService.dispatchServerDraft(saved.id);
        setStatusMessage({
          kind: "success",
          text: t(I18nKey.AUTOMATION_SETUP$TEST_DISPATCHED),
        });
        if (run.conversation_id) {
          // A dispatched test run starts a conversation the user can watch.
          navigate(`/conversations/${run.conversation_id}`);
        }
        return;
      }

      // Fallback for deployments without drafts: preflight only.
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
          ? t(I18nKey.AUTOMATION_SETUP$TEST_PASSED)
          : result.errors[0]?.message || t(I18nKey.SETUP$SUBMIT_FAILED),
      });
    } catch (error) {
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
    try {
      let created: Record<string, unknown>;
      if (kind === "custom") {
        const archive = await packTarGzip([
          { name: MAIN_PY_FILENAME, content: customCode, mode: 0o644 },
          {
            name: setupScriptPath.trim(),
            content: setupScript,
            mode: 0o755,
          },
        ]);
        const tarballPath = await AutomationService.uploadAutomationTarball(
          normalizedName(),
          archive,
        );
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
      toast.success(t(I18nKey.AUTOMATION_SETUP$CREATED));
      // The draft has been finalized into a real automation; drop the
      // persisted draft row so it does not linger as an incomplete setup.
      if (serverDraftId) {
        try {
          await AutomationService.deleteServerDraft(serverDraftId);
        } catch {
          // Cleanup is best-effort; the automation was created either way.
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
        {t(I18nKey.AUTOMATION_SETUP$TEST)}
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
            >
              <input
                data-testid="automation-setup-name"
                value={name}
                placeholder={t(I18nKey.AUTOMATION_SETUP$NAME_PLACEHOLDER)}
                onChange={(event) => updateField("name", event.target.value)}
                className={formControlFieldClassName}
              />
            </Field>

            {kind !== "custom" ? (
              <PromptFields
                prompt={prompt}
                updatedSuffix={agentUpdatedSuffix("prompt")}
                isStreaming={streamingField === "prompt"}
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
                <div className="flex items-center gap-2 rounded-xl border border-[var(--oh-border)] bg-base-secondary p-3">
                  <input
                    data-testid="automation-setup-repository"
                    value={repository}
                    placeholder={t(I18nKey.SETUP$REPOSITORY_PLACEHOLDER)}
                    onChange={(event) =>
                      updateField("repository", event.target.value)
                    }
                    className="min-w-0 flex-1 bg-transparent text-sm text-white outline-none placeholder:text-tertiary-alt"
                  />
                  <Plus className="size-4 text-[var(--oh-muted)]" aria-hidden />
                </div>
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
                updatedSuffixes={{
                  eventSource: agentUpdatedSuffix("eventSource"),
                  eventKey: agentUpdatedSuffix("eventKey"),
                  eventFilter: agentUpdatedSuffix("eventFilter"),
                }}
                streamingField={streamingField}
                setEventSource={(value) => updateField("eventSource", value)}
                setEventKey={(value) => updateField("eventKey", value)}
                setEventFilter={(value) => updateField("eventFilter", value)}
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
  onPromptChange,
}: {
  prompt: string;
  updatedSuffix?: string;
  isStreaming: boolean;
  onPromptChange: (value: string) => void;
}) {
  const { t } = useTranslation("openhands");
  return (
    <Field
      label={t(I18nKey.AUTOMATIONS$PROMPT)}
      suffix={updatedSuffix}
      isStreaming={isStreaming}
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
  updatedSuffixes,
  streamingField,
  setEventSource,
  setEventKey,
  setEventFilter,
}: {
  eventSource: string;
  eventKey: string;
  eventFilter: string;
  updatedSuffixes: Partial<
    Record<"eventSource" | "eventKey" | "eventFilter", string | undefined>
  >;
  streamingField: AutomationSetupField | null;
  setEventSource: (value: string) => void;
  setEventKey: (value: string) => void;
  setEventFilter: (value: string) => void;
}) {
  const { t } = useTranslation("openhands");
  return (
    <section className="grid gap-3 md:grid-cols-2">
      <Field
        label={t(I18nKey.AUTOMATION_SETUP$EVENT_SOURCE)}
        suffix={updatedSuffixes.eventSource}
        isStreaming={streamingField === "eventSource"}
      >
        <input
          data-testid="automation-setup-event-source"
          value={eventSource}
          onChange={(event) => setEventSource(event.target.value)}
          className={formControlFieldClassName}
        />
      </Field>
      <Field
        label={t(I18nKey.AUTOMATION_SETUP$EVENT_KEY)}
        suffix={updatedSuffixes.eventKey}
        isStreaming={streamingField === "eventKey"}
      >
        <input
          data-testid="automation-setup-event-key"
          value={eventKey}
          onChange={(event) => setEventKey(event.target.value)}
          className={formControlFieldClassName}
        />
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
    </section>
  );
}

function Field({
  label,
  suffix,
  horizontal = false,
  isStreaming = false,
  children,
}: {
  label: string;
  suffix?: string;
  horizontal?: boolean;
  isStreaming?: boolean;
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
