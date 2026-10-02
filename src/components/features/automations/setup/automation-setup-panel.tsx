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
  CalendarDays,
  Code2,
  FileText,
  Globe2,
  Plus,
  X,
  Zap,
} from "lucide-react";
import AutomationService from "#/api/automation-service/automation-service.api";
import {
  clearAutomationFormSession,
  initializeAutomationFormSession,
  patchAutomationFormSession,
  subscribeAutomationFormSession,
} from "#/api/automation-form-session";
import type {
  AutomationSetupDraft,
  AutomationSetupField,
  AutomationSetupFormPatch,
  AutomationSetupFormValues,
  AutomationSetupKind,
} from "#/api/automation-setup-types";
import {
  automationDetailPath,
  getAutomationEndpoint,
} from "#/manifests/automation-interface";
import { packTarGzip } from "#/utils/tar-gzip";
import { I18nKey } from "#/i18n/declaration";
import { BrandButton } from "#/components/features/settings/brand-button";
import { AutomationSetupPromptStack } from "#/components/features/automations/setup/automation-setup-prompt-stack";
import {
  formControlFieldClassName,
  formControlMultilineFieldClassName,
  formControlTransitionClassName,
} from "#/utils/form-control-classes";
import { cn } from "#/utils/utils";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { useNavigation } from "#/context/navigation-context";
import type {
  InterfaceEndpointName,
  SetupRequestBody,
} from "#/manifests/types";

const DEFAULT_TIMEZONE = "America/New_York";
const DEFAULT_TIME = "09:00";
const DEFAULT_CUSTOM_SCHEDULE = "0 9 * * *";
const DEFAULT_EVENT_SOURCE = "github";
const DEFAULT_EVENT_KEY = "issue_comment.created";
const DEFAULT_CUSTOM_ENTRYPOINT = "python3 main.py";
const addOptionButtonClassName = cn(
  "inline-flex w-fit shrink-0 cursor-pointer items-center rounded-lg border border-[var(--oh-border)] bg-tertiary px-3 py-1.5 text-sm text-content hover:bg-interactive-hover",
  formControlTransitionClassName,
);

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

function buildRepositorySource(url: string): {
  url: string;
  provider?: "github";
} {
  const isExplicitProviderUrl =
    /^[a-z][a-z\d+.-]*:\/\//i.test(url) || url.startsWith("git@");
  return isExplicitProviderUrl ? { url } : { url, provider: "github" };
}

type Frequency = (typeof FREQUENCIES)[number];
type StatusMessage = { kind: "success" | "error"; text: string } | null;

interface AutomationSetupPanelProps {
  draft: AutomationSetupDraft;
  conversationId?: string | null;
  toolbarPortal?: HTMLElement | null;
  showInlineHeader?: boolean;
  reserveComposerSpace?: boolean;
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
  toolbarPortal,
  showInlineHeader = true,
  reserveComposerSpace = false,
}: AutomationSetupPanelProps) {
  const [model, setModel] = useState("");
  const [agentProfileId, setAgentProfileId] = useState("");
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
    initializeAutomationFormSession(conversationId, {
      ...draft,
      form,
      fieldMetadata,
    });
    return () => clearAutomationFormSession(conversationId);
  }, [conversationId]);

  useEffect(() => {
    if (!conversationId) return undefined;
    return subscribeAutomationFormSession(
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
    patchAutomationFormSession(
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
    if (agentProfileId.trim()) {
      body.agent_profile_id = agentProfileId.trim();
    } else if (model.trim()) {
      body.model = model.trim();
    }
    const repositories = parseAutomationSetupRepositories(repository);
    if (repositories.length > 0) {
      body.repos = repositories.map(buildRepositorySource);
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
      enabled: false,
      setup_script_path: setupScriptPath.trim(),
      ...(showTimeout && timeoutSeconds.trim()
        ? { timeout: Number(timeoutSeconds) }
        : {}),
    }) as SetupRequestBody;
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
  const handleSaveDraft = () => {
    setStatusMessage({
      kind: "success",
      text: t(I18nKey.AUTOMATION_SETUP$DRAFT_SAVED),
    });
  };
  const handleTest = async () => {
    if (!validateRequiredFields()) return;
    setIsSubmitting(true);
    try {
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
      displayErrorToast(error instanceof Error ? error.message : null);
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
      if (typeof created.id === "string")
        navigate(automationDetailPath(created.id));
    } catch (error) {
      displayErrorToast(error instanceof Error ? error.message : null);
    } finally {
      setIsSubmitting(false);
    }
  };

  const compactToolbarButtonClassName = "!h-7 !min-h-7 !px-2.5 !text-xs";

  const renderToolbarActions = () => (
    <div className="flex shrink-0 items-center gap-1.5">
      <BrandButton
        type="button"
        variant="secondary"
        testId="automation-setup-save-draft"
        className={compactToolbarButtonClassName}
        isDisabled
        onClick={handleSaveDraft}
      >
        {t(I18nKey.AUTOMATION_SETUP$SAVE_DRAFT)}
      </BrandButton>
      <BrandButton
        type="button"
        variant="secondary"
        testId="automation-setup-test"
        className={compactToolbarButtonClassName}
        isDisabled={isSubmitting}
        onClick={handleTest}
      >
        {t(I18nKey.AUTOMATION_SETUP$TEST)}
      </BrandButton>
      <BrandButton
        type="button"
        variant="primary"
        testId="automation-setup-create"
        className={compactToolbarButtonClassName}
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
          <header className="flex h-10 min-h-10 shrink-0 items-center justify-between gap-2 border-b border-[var(--oh-border)] bg-base px-3">
            <div className="flex min-w-0 items-center gap-2">
              <h2 className="min-w-0 truncate text-sm font-medium text-content">
                {name.trim() || t(I18nKey.AUTOMATION_SETUP$TITLE)}
              </h2>
            </div>
            {renderToolbarActions()}
          </header>
        ) : null}

        <div
          className={cn(
            "custom-scrollbar-always min-h-0 flex-1 overflow-y-auto px-5 pt-5 [scrollbar-gutter:stable]",
            reserveComposerSpace ? "pb-52" : "pb-5",
          )}
        >
          <div className="mx-auto flex w-full min-w-0 max-w-[800px] flex-col gap-6">
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

            <div className="flex flex-col gap-2.5">
              <div className="flex w-full items-center gap-2">
                <span className="flex items-center gap-2 text-sm">
                  {kind === "custom"
                    ? t(I18nKey.AUTOMATION_SETUP$CUSTOM_PYTHON)
                    : t(I18nKey.AUTOMATIONS$PROMPT)}
                </span>
                <button
                  type="button"
                  data-testid={
                    kind === "custom"
                      ? "automation-setup-kind-prompt"
                      : "automation-setup-kind-custom"
                  }
                  onClick={() =>
                    updateField("kind", kind === "custom" ? "prompt" : "custom")
                  }
                  className={cn(
                    addOptionButtonClassName,
                    "ml-auto gap-1.5",
                    streamingHighlightClassName(streamingField === "kind"),
                  )}
                >
                  {kind === "custom" ? (
                    <FileText className="size-4" aria-hidden />
                  ) : (
                    <Code2 className="size-4" aria-hidden />
                  )}
                  {kind === "custom"
                    ? t(I18nKey.AUTOMATIONS$PROMPT)
                    : t(I18nKey.AUTOMATION_SETUP$TYPE_CUSTOM)}
                </button>
              </div>
              {kind !== "custom" ? (
                <AutomationSetupPromptStack
                  prompt={prompt}
                  repository={repository}
                  updatedSuffix={agentUpdatedSuffix("prompt")}
                  repositorySuffix={agentUpdatedSuffix("repository")}
                  showTitle={false}
                  isStreaming={streamingField === "prompt"}
                  onPromptChange={(value) => updateField("prompt", value)}
                  onRepositoryChange={(value) =>
                    updateField("repository", value)
                  }
                  model={model}
                  onModelChange={setModel}
                  agentProfileId={agentProfileId}
                  onAgentProfileChange={setAgentProfileId}
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
                  onEntrypointChange={(value) =>
                    updateField("entrypoint", value)
                  }
                  onSetupScriptPathChange={(value) =>
                    updateField("setupScriptPath", value)
                  }
                  onSetupScriptChange={(value) =>
                    updateField("setupScript", value)
                  }
                />
              )}
            </div>

            <section className="flex flex-col gap-2.5">
              <div
                role="radiogroup"
                aria-label={t(I18nKey.AUTOMATIONS$DETAIL$TRIGGER)}
                className={cn(
                  "grid grid-cols-2 gap-2",
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

            <section className="flex flex-col gap-2.5">
              <span className="text-sm">
                {t(I18nKey.AUTOMATION_SETUP$ADDITIONAL_OPTIONS)}
              </span>
              {kind === "plugin" ? (
                <div
                  data-testid="automation-setup-plugin-module"
                  className="flex flex-col gap-3 rounded-xl border border-[var(--oh-border)] bg-base-secondary p-4"
                >
                  <div className="flex min-w-0 items-end gap-2">
                    <div className="grid min-w-0 flex-1 gap-3 md:grid-cols-[2fr_1fr]">
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
                    <button
                      type="button"
                      data-testid="automation-setup-plugin-remove"
                      aria-label={t(I18nKey.COMMON$REMOVE)}
                      onClick={() => {
                        updateField("pluginSource", "");
                        updateField("pluginRef", "");
                        updateField("kind", "prompt");
                      }}
                      className={cn(
                        "flex size-9 shrink-0 items-center justify-center rounded-lg border border-[var(--oh-border)] text-[var(--oh-muted)] hover:bg-white/5 hover:text-white",
                        formControlTransitionClassName,
                      )}
                    >
                      <X className="size-4" aria-hidden />
                    </button>
                  </div>
                </div>
              ) : null}
              {showTimeout ? (
                <div className="flex min-w-0 items-end gap-2">
                  <div className="min-w-0 flex-1">
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
                  </div>
                  <button
                    type="button"
                    data-testid="automation-setup-timeout-remove"
                    aria-label={t(I18nKey.COMMON$REMOVE)}
                    onClick={() => updateField("showTimeout", false)}
                    className={cn(
                      "flex size-9 shrink-0 items-center justify-center rounded-lg border border-[var(--oh-border)] text-[var(--oh-muted)] hover:bg-white/5 hover:text-white",
                      formControlTransitionClassName,
                    )}
                  >
                    <X className="size-4" aria-hidden />
                  </button>
                </div>
              ) : null}
              <div className="flex flex-wrap items-center gap-2">
                {kind !== "plugin" ? (
                  <button
                    type="button"
                    data-testid="automation-setup-add-plugin"
                    onClick={() => updateField("kind", "plugin")}
                    className={cn(addOptionButtonClassName, "gap-1.5")}
                  >
                    <Plus className="size-4" aria-hidden />
                    {`${t(I18nKey.BUTTON$ADD)} ${t(I18nKey.AUTOMATION_SETUP$TYPE_PLUGIN)}`}
                  </button>
                ) : null}
                {!showTimeout ? (
                  <button
                    type="button"
                    data-testid="automation-setup-add-timeout"
                    onClick={() => updateField("showTimeout", true)}
                    className={cn(
                      addOptionButtonClassName,
                      "gap-1.5",
                      streamingHighlightClassName(
                        streamingField === "showTimeout",
                      ),
                    )}
                  >
                    <Plus className="size-4" aria-hidden />
                    {t(I18nKey.AUTOMATION_SETUP$ADD_TIMEOUT)}
                  </button>
                ) : null}
              </div>
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
    <section className="flex flex-col gap-2.5">
      <div className="flex min-w-0 items-center gap-2">
        <span className="shrink-0 text-sm">
          {t(I18nKey.AUTOMATION_SETUP$FREQUENCY)}
        </span>
        {updatedSuffixes.frequency ? (
          <span className="shrink-0 text-xs text-[var(--oh-muted)]">
            {updatedSuffixes.frequency}
          </span>
        ) : null}
        <div
          role="radiogroup"
          aria-label={t(I18nKey.AUTOMATION_SETUP$FREQUENCY)}
          className={cn(
            "inline-flex max-w-full min-w-0 items-center gap-0.5 overflow-x-auto rounded-lg bg-[var(--oh-surface-raised)] p-0.5",
            streamingHighlightClassName(streamingField === "frequency"),
          )}
        >
          {FREQUENCIES.map((item) => (
            <button
              key={item}
              type="button"
              role="radio"
              data-testid={`automation-setup-frequency-${item}`}
              aria-checked={frequency === item}
              onClick={() => setFrequency(item)}
              className={cn(
                "inline-flex h-8 shrink-0 items-center rounded-md px-3 text-sm",
                formControlTransitionClassName,
                frequency === item
                  ? "border border-[var(--oh-interactive-hover)] bg-base-secondary text-content"
                  : "border border-transparent text-[var(--oh-muted)] hover:text-content",
              )}
            >
              {t(frequencyLabelKey(item))}
            </button>
          ))}
        </div>
      </div>
      <div className="flex w-full min-w-0 items-center gap-4 overflow-x-auto">
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
              className={cn(formControlFieldClassName, "w-[14rem]")}
            />
          </Field>
        ) : (
          <>
            <label className="flex shrink-0 items-center gap-2.5">
              <span className="shrink-0 text-sm text-content">
                {t(I18nKey.AUTOMATION_SETUP$AT)}
              </span>
              <div
                data-streaming-active={
                  streamingField === "time" ? "true" : undefined
                }
                className={cn(
                  "relative",
                  streamingHighlightClassName(streamingField === "time"),
                )}
              >
                <input
                  data-testid="automation-setup-time"
                  type="time"
                  value={time}
                  onChange={(event) => setTime(event.target.value)}
                  className={cn(formControlFieldClassName, "w-[9.5rem]")}
                />
              </div>
            </label>
            <div
              data-streaming-active={
                streamingField === "timezone" ? "true" : undefined
              }
              className={cn(
                "relative shrink-0",
                streamingHighlightClassName(streamingField === "timezone"),
              )}
            >
              <select
                aria-label={t(I18nKey.AUTOMATIONS$TIMEZONE)}
                data-testid="automation-setup-timezone"
                value={timezone}
                onChange={(event) => setTimezone(event.target.value)}
                className={cn(formControlFieldClassName, "w-[15rem] pl-9")}
              >
                <option value={timezone}>{timezone}</option>
                {timezone !== DEFAULT_TIMEZONE ? (
                  <option value={DEFAULT_TIMEZONE}>{DEFAULT_TIMEZONE}</option>
                ) : null}
              </select>
              <Globe2
                className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-[var(--oh-muted)]"
                aria-hidden
              />
            </div>
            {atUpdatedSuffix ? (
              <span className="shrink-0 text-xs text-[var(--oh-muted)]">
                {atUpdatedSuffix}
              </span>
            ) : null}
          </>
        )}
      </div>
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
