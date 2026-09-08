import { useState, type ReactNode } from "react";
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
import type {
  AutomationSetupDraft,
  AutomationSetupKind,
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
const MAIN_PY_FILENAME = "main.py";
const DEFAULT_CUSTOM_SETUP_SCRIPT_PATH = "setup.sh";
const DEFAULT_CUSTOM_SETUP_SCRIPT = `#!/usr/bin/env bash
:
`;
const DEFAULT_TIMEOUT_SECONDS = "600";
const PREFLIGHT_TARBALL_PATH =
  "oh-internal://uploads/00000000-0000-0000-0000-000000000000";

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

type Frequency = (typeof FREQUENCIES)[number];
type TriggerKind = "cron" | "event";
type StatusMessage = { kind: "success" | "error"; text: string } | null;

interface AutomationSetupPanelProps {
  draft: AutomationSetupDraft;
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

function endpointName(kind: AutomationSetupKind): InterfaceEndpointName {
  if (kind === "plugin") return "createPlugin";
  if (kind === "custom") return "createBundle";
  return "createPrompt";
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
  onClose,
}: AutomationSetupPanelProps) {
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

  return (
    <div
      data-testid="automation-setup-panel"
      className="flex h-full min-h-0 flex-col bg-base"
    >
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
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-6">
        <div className="mx-auto flex w-full max-w-5xl flex-col gap-6">
          <div
            role="group"
            aria-label={t(I18nKey.AUTOMATION_SETUP$TYPE_LABEL)}
            className="grid grid-cols-3 gap-2 rounded-xl border border-[var(--oh-border)] bg-base-secondary p-1"
          >
            {AUTOMATION_SETUP_KINDS.map((item) => (
              <button
                key={item}
                type="button"
                aria-pressed={kind === item}
                data-testid={`automation-setup-kind-${item}`}
                onClick={() => setKind(item)}
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
                {item === "plugin" && <Puzzle className="size-4" aria-hidden />}
                {item === "custom" && <Code2 className="size-4" aria-hidden />}
                <span>{t(kindLabelKey(item))}</span>
              </button>
            ))}
          </div>

          <Field label={t(I18nKey.AUTOMATIONS$NAME)}>
            <input
              data-testid="automation-setup-name"
              value={name}
              placeholder={t(I18nKey.AUTOMATION_SETUP$NAME_PLACEHOLDER)}
              onChange={(event) => setName(event.target.value)}
              className={formControlFieldClassName}
            />
          </Field>

          {kind !== "custom" ? (
            <PromptFields prompt={prompt} onPromptChange={setPrompt} />
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

          {kind === "plugin" && (
            <div className="grid gap-3 rounded-xl border border-[var(--oh-border)] bg-base-secondary p-4 md:grid-cols-[2fr_1fr]">
              <Field label={t(I18nKey.AUTOMATION_SETUP$PLUGIN_SOURCE)}>
                <input
                  data-testid="automation-setup-plugin-source"
                  value={pluginSource}
                  placeholder={t(
                    I18nKey.AUTOMATION_SETUP$PLUGIN_SOURCE_PLACEHOLDER,
                  )}
                  onChange={(event) => setPluginSource(event.target.value)}
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
          )}

          {kind !== "custom" && (
            <Field
              label={t(I18nKey.COMMON$REPOSITORIES)}
              suffix={t(I18nKey.COMMON$OPTIONAL)}
            >
              <div className="flex items-center gap-2 rounded-xl border border-[var(--oh-border)] bg-base-secondary p-3">
                <input
                  data-testid="automation-setup-repository"
                  value={repository}
                  placeholder={t(I18nKey.SETUP$REPOSITORY_PLACEHOLDER)}
                  onChange={(event) => setRepository(event.target.value)}
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
            <div className="grid gap-3 md:grid-cols-2">
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

          <section className="flex flex-col gap-3">
            <h3 className="text-sm font-semibold text-white">
              {t(I18nKey.AUTOMATION_SETUP$ADDITIONAL_OPTIONS)}
            </h3>
            {showTimeout ? (
              <Field label={t(I18nKey.AUTOMATION_SETUP$TIMEOUT_SECONDS)}>
                <input
                  data-testid="automation-setup-timeout"
                  type="number"
                  min="1"
                  value={timeoutSeconds}
                  onChange={(event) => setTimeoutSeconds(event.target.value)}
                  className={formControlFieldClassName}
                />
              </Field>
            ) : (
              <button
                type="button"
                data-testid="automation-setup-add-timeout"
                onClick={() => setShowTimeout(true)}
                className={cn(
                  "w-fit rounded-full border border-[var(--oh-border)] px-4 py-2 text-sm text-[var(--oh-muted)] hover:bg-white/5 hover:text-white",
                  formControlTransitionClassName,
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
  );
}

function PromptFields({
  prompt,
  onPromptChange,
}: {
  prompt: string;
  onPromptChange: (value: string) => void;
}) {
  const { t } = useTranslation("openhands");
  return (
    <Field label={t(I18nKey.AUTOMATIONS$PROMPT)}>
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
    <section className="flex flex-col gap-3">
      <h3 className="text-sm font-semibold text-white">
        {t(I18nKey.AUTOMATION_SETUP$FREQUENCY)}
      </h3>
      <div className="grid grid-cols-2 gap-1 rounded-xl bg-base-secondary p-1 md:grid-cols-6">
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
        <input
          data-testid="automation-setup-custom-schedule"
          value={customSchedule}
          onChange={(event) => setCustomSchedule(event.target.value)}
          className={formControlFieldClassName}
        />
      ) : (
        <div className="grid gap-3 md:grid-cols-[16rem_1fr]">
          <Field label={t(I18nKey.AUTOMATION_SETUP$AT)} horizontal>
            <div className="relative">
              <input
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
          </Field>
          <div className="relative">
            <input
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
        </div>
      )}
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
