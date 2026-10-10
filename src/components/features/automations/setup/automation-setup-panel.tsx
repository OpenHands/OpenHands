import { useState, type ReactNode } from "react";
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
import type {
  AutomationSetupDraft,
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

const FREQUENCIES = [
  "hourly",
  "daily",
  "weekdays",
  "weekly",
  "custom",
] as const;

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
type TriggerKind = "cron" | "event";
type StatusMessage = { kind: "success" | "error"; text: string } | null;

interface AutomationSetupPanelProps {
  draft: AutomationSetupDraft;
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
  toolbarPortal,
  showInlineHeader = true,
  reserveComposerSpace = false,
}: AutomationSetupPanelProps) {
  const [model, setModel] = useState("");
  const [agentProfileId, setAgentProfileId] = useState("");
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const [kind, setKind] = useState<AutomationSetupKind>(draft.kind);
  const [name, setName] = useState(() => deriveName(draft.prompt));
  const [prompt, setPrompt] = useState(draft.prompt);
  const [repository, setRepository] = useState("");
  const [pluginSource, setPluginSource] = useState(draft.plugins?.[0] ?? "");
  const [pluginRef, setPluginRef] = useState("");
  const [customCode, setCustomCode] = useState(() =>
    buildStarterPython(draft.prompt),
  );
  const [entrypoint, setEntrypoint] = useState(DEFAULT_CUSTOM_ENTRYPOINT);
  const [setupScriptPath, setSetupScriptPath] = useState(
    DEFAULT_CUSTOM_SETUP_SCRIPT_PATH,
  );
  const [setupScript, setSetupScript] = useState(DEFAULT_CUSTOM_SETUP_SCRIPT);
  const [triggerKind, setTriggerKind] = useState<TriggerKind>("cron");
  const [frequency, setFrequency] = useState<Frequency>("daily");
  const [time, setTime] = useState(DEFAULT_TIME);
  const [timezone, setTimezone] = useState(DEFAULT_TIMEZONE);
  const [customSchedule, setCustomSchedule] = useState(DEFAULT_CUSTOM_SCHEDULE);
  const [eventSource, setEventSource] = useState(DEFAULT_EVENT_SOURCE);
  const [eventKey, setEventKey] = useState(DEFAULT_EVENT_KEY);
  const [eventFilter, setEventFilter] = useState("");
  const [showTimeout, setShowTimeout] = useState(false);
  const [timeoutSeconds, setTimeoutSeconds] = useState(DEFAULT_TIMEOUT_SECONDS);
  const [statusMessage, setStatusMessage] = useState<StatusMessage>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

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
            <Field label={t(I18nKey.AUTOMATIONS$NAME)}>
              <input
                data-testid="automation-setup-name"
                value={name}
                placeholder={t(I18nKey.AUTOMATION_SETUP$NAME_PLACEHOLDER)}
                onChange={(event) => setName(event.target.value)}
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
                    setKind(kind === "custom" ? "prompt" : "custom")
                  }
                  className={cn(addOptionButtonClassName, "ml-auto gap-1.5")}
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
                  showTitle={false}
                  isStreaming={false}
                  onPromptChange={setPrompt}
                  onRepositoryChange={setRepository}
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
                  onCodeChange={setCustomCode}
                  onEntrypointChange={setEntrypoint}
                  onSetupScriptPathChange={setSetupScriptPath}
                  onSetupScriptChange={setSetupScript}
                />
              )}
            </div>

            <section className="flex flex-col gap-2.5">
              <div
                role="radiogroup"
                aria-label={t(I18nKey.AUTOMATIONS$DETAIL$TRIGGER)}
                className="grid grid-cols-2 gap-2"
              >
                <TriggerCard
                  icon={<CalendarDays className="size-4" aria-hidden />}
                  title={t(I18nKey.AUTOMATION_SETUP$SCHEDULE)}
                  description={t(I18nKey.AUTOMATION_SETUP$SCHEDULE_DESCRIPTION)}
                  selected={triggerKind === "cron"}
                  onClick={() => setTriggerKind("cron")}
                />
                <TriggerCard
                  icon={<Zap className="size-4" aria-hidden />}
                  title={t(I18nKey.AUTOMATIONS$DETAIL$TRIGGER_EVENT)}
                  description={t(I18nKey.AUTOMATION_SETUP$EVENT_DESCRIPTION)}
                  selected={triggerKind === "event"}
                  onClick={() => setTriggerKind("event")}
                />
              </div>
            </section>

            {triggerKind === "cron" ? (
              <ScheduleFields
                frequency={frequency}
                time={time}
                timezone={timezone}
                customSchedule={customSchedule}
                setFrequency={setFrequency}
                setTime={setTime}
                setTimezone={setTimezone}
                setCustomSchedule={setCustomSchedule}
              />
            ) : (
              <EventFields
                eventSource={eventSource}
                eventKey={eventKey}
                eventFilter={eventFilter}
                setEventSource={setEventSource}
                setEventKey={setEventKey}
                setEventFilter={setEventFilter}
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
                      <Field label={t(I18nKey.AUTOMATION_SETUP$PLUGIN_SOURCE)}>
                        <input
                          data-testid="automation-setup-plugin-source"
                          value={pluginSource}
                          placeholder={t(
                            I18nKey.AUTOMATION_SETUP$PLUGIN_SOURCE_PLACEHOLDER,
                          )}
                          onChange={(event) =>
                            setPluginSource(event.target.value)
                          }
                          className={formControlFieldClassName}
                        />
                      </Field>
                      <Field label={t(I18nKey.AUTOMATION_SETUP$PLUGIN_REF)}>
                        <input
                          data-testid="automation-setup-plugin-ref"
                          value={pluginRef}
                          placeholder={t(
                            I18nKey.AUTOMATION_SETUP$PLUGIN_REF_PLACEHOLDER,
                          )}
                          onChange={(event) => setPluginRef(event.target.value)}
                          className={formControlFieldClassName}
                        />
                      </Field>
                    </div>
                    <button
                      type="button"
                      data-testid="automation-setup-plugin-remove"
                      aria-label={t(I18nKey.COMMON$REMOVE)}
                      onClick={() => {
                        setPluginSource("");
                        setPluginRef("");
                        setKind("prompt");
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
                    <Field label={t(I18nKey.AUTOMATION_SETUP$TIMEOUT_SECONDS)}>
                      <input
                        data-testid="automation-setup-timeout"
                        type="number"
                        min="1"
                        value={timeoutSeconds}
                        onChange={(event) =>
                          setTimeoutSeconds(event.target.value)
                        }
                        className={formControlFieldClassName}
                      />
                    </Field>
                  </div>
                  <button
                    type="button"
                    data-testid="automation-setup-timeout-remove"
                    aria-label={t(I18nKey.COMMON$REMOVE)}
                    onClick={() => setShowTimeout(false)}
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
                    onClick={() => setKind("plugin")}
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
                    onClick={() => setShowTimeout(true)}
                    className={cn(addOptionButtonClassName, "gap-1.5")}
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
  onCodeChange,
  onEntrypointChange,
  onSetupScriptPathChange,
  onSetupScriptChange,
}: {
  code: string;
  entrypoint: string;
  setupScriptPath: string;
  setupScript: string;
  onCodeChange: (value: string) => void;
  onEntrypointChange: (value: string) => void;
  onSetupScriptPathChange: (value: string) => void;
  onSetupScriptChange: (value: string) => void;
}) {
  const { t } = useTranslation("openhands");
  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-3 md:grid-cols-[2fr_1fr]">
        <Field label={t(I18nKey.AUTOMATION_SETUP$ENTRYPOINT)}>
          <input
            data-testid="automation-setup-entrypoint"
            value={entrypoint}
            onChange={(event) => onEntrypointChange(event.target.value)}
            className={formControlFieldClassName}
          />
        </Field>
        <Field label={t(I18nKey.AUTOMATION_SETUP$SETUP_SCRIPT_PATH)}>
          <input
            data-testid="automation-setup-setup-script-path"
            value={setupScriptPath}
            onChange={(event) => onSetupScriptPathChange(event.target.value)}
            className={formControlFieldClassName}
          />
        </Field>
      </div>
      <Field label={t(I18nKey.AUTOMATION_SETUP$PYTHON_CODE)}>
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
      <Field label={t(I18nKey.AUTOMATION_SETUP$SETUP_SCRIPT)}>
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
  setFrequency,
  setTime,
  setTimezone,
  setCustomSchedule,
}: {
  frequency: Frequency;
  time: string;
  timezone: string;
  customSchedule: string;
  setFrequency: (value: Frequency) => void;
  setTime: (value: string) => void;
  setTimezone: (value: string) => void;
  setCustomSchedule: (value: string) => void;
}) {
  const { t } = useTranslation("openhands");
  return (
    <section className="flex flex-col gap-2.5">
      <div className="flex min-w-0 items-center gap-2">
        <span className="shrink-0 text-sm">
          {t(I18nKey.AUTOMATION_SETUP$FREQUENCY)}
        </span>
        <div
          role="radiogroup"
          aria-label={t(I18nKey.AUTOMATION_SETUP$FREQUENCY)}
          className="inline-flex max-w-full min-w-0 items-center gap-0.5 overflow-x-auto rounded-lg bg-[var(--oh-surface-raised)] p-0.5"
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
          <label className="flex shrink-0 items-center gap-2.5">
            <span className="shrink-0 text-sm text-content">
              {t(I18nKey.AUTOMATION_SETUP$TYPE_CUSTOM)}
            </span>
            <input
              data-testid="automation-setup-custom-schedule"
              value={customSchedule}
              onChange={(event) => setCustomSchedule(event.target.value)}
              className={cn(formControlFieldClassName, "w-[14rem]")}
            />
          </label>
        ) : (
          <>
            <label className="flex shrink-0 items-center gap-2.5">
              <span className="shrink-0 text-sm text-content">
                {t(I18nKey.AUTOMATION_SETUP$AT)}
              </span>
              <div className="relative">
                <input
                  data-testid="automation-setup-time"
                  type="time"
                  value={time}
                  onChange={(event) => setTime(event.target.value)}
                  className={cn(formControlFieldClassName, "w-[9.5rem]")}
                />
              </div>
            </label>
            <div className="relative shrink-0">
              <select
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
  setEventSource,
  setEventKey,
  setEventFilter,
}: {
  eventSource: string;
  eventKey: string;
  eventFilter: string;
  setEventSource: (value: string) => void;
  setEventKey: (value: string) => void;
  setEventFilter: (value: string) => void;
}) {
  const { t } = useTranslation("openhands");
  return (
    <section className="grid gap-3 md:grid-cols-2">
      <Field label={t(I18nKey.AUTOMATION_SETUP$EVENT_SOURCE)}>
        <input
          data-testid="automation-setup-event-source"
          value={eventSource}
          onChange={(event) => setEventSource(event.target.value)}
          className={formControlFieldClassName}
        />
      </Field>
      <Field label={t(I18nKey.AUTOMATION_SETUP$EVENT_KEY)}>
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
          suffix={t(I18nKey.COMMON$OPTIONAL)}
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
  children,
}: {
  label: string;
  suffix?: string;
  horizontal?: boolean;
  children: ReactNode;
}) {
  return (
    <label
      className={cn("flex gap-2", horizontal ? "items-center" : "flex-col")}
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
