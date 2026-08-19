import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import XMarkIcon from "#/icons/x-mark.svg?react";
import {
  useBashCommandLogs,
  type SandboxIssue,
} from "#/hooks/query/use-bash-command-logs";
import type { BashOutput } from "@openhands/typescript-client";
import { cn } from "#/utils/utils";
import { modalTitleLgMediumClassName } from "#/utils/modal-classes";
import {
  AutomationRunStatus,
  type Automation,
  type AutomationRun,
} from "#/types/automation";
import { DebugAutomationButton } from "./debug-automation-button";

/**
 * Localized empty-state message key for each `SandboxIssue` reason.
 * Centralised so we don't sprinkle conditional renders for each code.
 */
const SANDBOX_ISSUE_I18N: Record<SandboxIssue, I18nKey> = {
  missing: I18nKey.AUTOMATIONS$DETAIL$LOGS_SANDBOX_MISSING,
  paused: I18nKey.AUTOMATIONS$DETAIL$LOGS_SANDBOX_PAUSED,
  starting: I18nKey.AUTOMATIONS$DETAIL$LOGS_SANDBOX_STARTING,
  errored: I18nKey.AUTOMATIONS$DETAIL$LOGS_SANDBOX_ERROR,
  unreachable: I18nKey.AUTOMATIONS$DETAIL$LOGS_SANDBOX_UNREACHABLE,
};

type LogTab = "stdout" | "stderr";

interface RunLogsModalProps {
  /** Conversation that owns the bash command. */
  conversationId: string | null;
  /** Bash command id to fetch logs for. */
  bashCommandId: string | null;
  isOpen: boolean;
  onClose: () => void;
  /** The run these logs belong to; enables the debug action for failed runs. */
  run?: AutomationRun;
  /** The parent automation, used to add context to the debug prompt. */
  automation?: Automation;
}

function concatStream(outputs: BashOutput[], key: "stdout" | "stderr"): string {
  // Outputs come back from the API sorted by timestamp, but pages can
  // arrive out-of-order, so re-sort by (timestamp, order) before
  // concatenating to keep the stream chronological.
  return [...outputs]
    .sort((a, b) => {
      const ts = a.timestamp.localeCompare(b.timestamp);
      if (ts !== 0) return ts;
      return (a.order ?? 0) - (b.order ?? 0);
    })
    .map((output) => output[key] ?? "")
    .join("");
}

const STATUS_DETAIL_ROW_KEYS = ["phase", "kind", "transient"] as const;

type StatusDetailRowKey = (typeof STATUS_DETAIL_ROW_KEYS)[number];

const STATUS_DETAIL_RENDERED_KEYS = new Set<string>([
  "detail",
  ...STATUS_DETAIL_ROW_KEYS,
]);

interface ParsedDetailText {
  message: string | null;
  stdout: string | null;
  stderr: string | null;
}

interface DetailSection {
  key: string;
  label: string;
  value: string;
  tone?: "danger" | "content";
}

interface DetailRow {
  key: string;
  label: string;
  value: string;
}

function humanizeToken(value: string): string {
  return value
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatStatusDetailLabel(key: string): string {
  return humanizeToken(key);
}

function formatStatusDetailValue(value: unknown): string | null {
  if (value == null) return null;
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") return value;
  return null;
}

function parseDetailText(value: unknown): ParsedDetailText | null {
  if (typeof value !== "string" || value.trim().length === 0) return null;

  const sections: Record<keyof ParsedDetailText, string[]> = {
    message: [],
    stdout: [],
    stderr: [],
  };
  let current: keyof ParsedDetailText = "message";

  for (const line of value.split(/\r?\n/)) {
    const marker = /^(stdout|stderr):\s*(.*)$/i.exec(line);
    if (marker) {
      current = marker[1].toLowerCase() as "stdout" | "stderr";
      if (marker[2]) sections[current].push(marker[2]);
      continue;
    }
    sections[current].push(line);
  }

  const clean = (lines: string[]) => {
    const text = lines.join("\n").trim();
    return text.length > 0 ? text : null;
  };

  const message = clean(sections.message);
  const stdout = clean(sections.stdout);
  const stderr = clean(sections.stderr);
  return message || stdout || stderr ? { message, stdout, stderr } : null;
}

function parseExitCode(
  detail: ParsedDetailText | null,
  statusDetail: AutomationRun["status_detail"],
): string | null {
  const code = formatStatusDetailValue(statusDetail?.code);
  if (code) return code;
  const match = detail?.message?.match(/(?:^|\n)exit_code=(\d+)/);
  return match?.[1] ?? null;
}

function stringField(
  statusDetail: AutomationRun["status_detail"],
  key: string,
): string | null {
  const value = statusDetail?.[key];
  return typeof value === "string" && value.trim().length > 0
    ? value.trim()
    : null;
}

function firstSentence(value: string): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  const match = normalized.match(/^(.+?[.!?])(?:\s|$)/);
  return (match?.[1] ?? normalized).trim();
}

function formatHumanErrorReason(value: string): string | null {
  const cleaned = value.trim();
  if (!cleaned || /^execution failed\.?$/i.test(cleaned)) return null;

  if (
    /authenticationerror/i.test(cleaned) &&
    /incorrect api key/i.test(cleaned)
  ) {
    return "LLM authentication failed: incorrect API key provided.";
  }
  if (/rate.?limit/i.test(cleaned)) {
    return "LLM provider rate limit reached.";
  }

  const withoutCommonPrefixes = cleaned
    .replace(/^litellm\.[A-Za-z]+Error:\s*/i, "")
    .replace(/^[A-Za-z]+Error:\s*/i, "")
    .replace(/^OpenAIException\s*-\s*/i, "")
    .trim();

  return firstSentence(withoutCommonPrefixes);
}

function getStatusDetailReason(
  statusDetail: AutomationRun["status_detail"],
): string | null {
  const candidates = [
    stringField(statusDetail, "reason"),
    stringField(statusDetail, "message"),
    stringField(statusDetail, "formatted_detail"),
  ];

  for (const candidate of candidates) {
    if (!candidate) continue;
    const parsedCandidate = parseDetailText(candidate);
    const reason = formatHumanErrorReason(
      parsedCandidate?.message ?? candidate,
    );
    if (reason) return reason;
  }
  return null;
}

function getStatusDetailSummary(
  statusDetail: AutomationRun["status_detail"],
  parsedDetail: ParsedDetailText | null,
): string | null {
  if (!statusDetail) return null;

  const kind = typeof statusDetail.kind === "string" ? statusDetail.kind : "";
  const phase =
    typeof statusDetail.phase === "string" ? statusDetail.phase : "";
  const statusCode = statusDetail.status_code;
  const exitCode = parseExitCode(parsedDetail, statusDetail);
  const transient = statusDetail.transient === true;
  const reason = getStatusDetailReason(statusDetail);

  if (reason) return reason;

  if (statusCode === 429 || kind === "api_rate_limited") {
    return `${transient ? "Temporary" : "HTTP"} API rate limit${
      statusCode ? ` (HTTP ${statusCode})` : ""
    }.`;
  }
  if (kind.includes("execution") || phase === "execution") {
    return exitCode
      ? `Execution failed with exit code ${exitCode}.`
      : "Execution failed.";
  }
  if (statusCode) {
    return `${transient ? "Temporary" : "HTTP"} ${
      kind ? humanizeToken(kind).toLowerCase() : "API issue"
    } (HTTP ${statusCode}).`;
  }
  if (kind) {
    return `${humanizeToken(kind)}${transient ? " (temporary)" : ""}.`;
  }
  return parsedDetail?.message ?? null;
}

function getStatusDetailRowLabel(key: StatusDetailRowKey): string {
  switch (key) {
    case "kind":
      return "Error";
    case "transient":
      return "Transient";
    default:
      return formatStatusDetailLabel(key);
  }
}

function formatMetadataLabel(key: string): string {
  switch (key) {
    case "code":
      return "Code";
    case "status_code":
      return "HTTP status";
    case "count":
      return "Occurrences";
    case "first_seen_at":
      return "First seen";
    case "last_seen_at":
      return "Last seen";
    default:
      return formatStatusDetailLabel(key);
  }
}

function formatMetadataValue(key: string, value: unknown): string | null {
  const formatted = formatStatusDetailValue(value);
  if (!formatted) return null;
  return ["source", "operation"].includes(key) && typeof value === "string"
    ? humanizeToken(value)
    : formatted;
}

function getStatusDetailRows(
  statusDetail: AutomationRun["status_detail"],
): DetailRow[] {
  if (!statusDetail) return [];
  return STATUS_DETAIL_ROW_KEYS.flatMap((key) => {
    const rawValue = statusDetail[key];
    const value = formatStatusDetailValue(rawValue);
    if (!value) return [];
    const displayValue =
      ["phase", "kind", "source", "operation"].includes(key) &&
      typeof rawValue === "string"
        ? humanizeToken(rawValue)
        : value;
    return [
      {
        key,
        label: getStatusDetailRowLabel(key),
        value: displayValue,
      },
    ];
  });
}

function getDetailSections(
  parsedDetail: ParsedDetailText | null,
  summary: string | null,
): DetailSection[] {
  if (!parsedDetail) return [];

  const message = parsedDetail.message
    ?.split(/\r?\n/)
    .filter((line) => !/^exit_code=\d+$/.test(line.trim()))
    .join("\n")
    .trim();

  return [
    ...(message && message !== summary
      ? [{ key: "message", label: "Message", value: message }]
      : []),
    ...(parsedDetail.stderr
      ? [
          {
            key: "stderr",
            label: "Error output",
            value: parsedDetail.stderr,
            tone: "danger" as const,
          },
        ]
      : []),
    ...(parsedDetail.stdout
      ? [
          {
            key: "stdout",
            label: "Output",
            value: parsedDetail.stdout,
            tone: "content" as const,
          },
        ]
      : []),
  ];
}

function flattenMetadataRows(value: unknown, path: string[] = []): DetailRow[] {
  if (value === undefined || value === null) return [];
  if (["string", "number", "boolean"].includes(typeof value)) {
    const key = path.join(".");
    const labelPath = path[0] === "metadata" ? path.slice(1) : path;
    const labelKey = labelPath.join("_") || key;
    return [
      {
        key,
        label: formatMetadataLabel(labelKey),
        value: formatMetadataValue(labelKey, value) ?? "",
      },
    ];
  }
  if (Array.isArray(value)) {
    return [
      {
        key: path.join("."),
        label: formatStatusDetailLabel(path.join("_")),
        value: JSON.stringify(value),
      },
    ];
  }
  if (typeof value === "object") {
    return Object.entries(value).flatMap(([key, nested]) =>
      flattenMetadataRows(nested, [...path, key]),
    );
  }
  return [];
}

function getStatusDetailExtraRows(
  statusDetail: AutomationRun["status_detail"],
): DetailRow[] {
  if (!statusDetail) return [];
  return Object.entries(statusDetail).flatMap(([key, value]) =>
    STATUS_DETAIL_RENDERED_KEYS.has(key) || value === undefined
      ? []
      : flattenMetadataRows(value, [key]),
  );
}

function DetailSections({ sections }: { sections: DetailSection[] }) {
  if (sections.length === 0) return null;

  return (
    <div className="space-y-2">
      {sections.map(({ key, label, value, tone = "content" }) => (
        <div key={key}>
          <div className="mb-1 text-xs font-medium uppercase tracking-wide text-muted">
            {label}
          </div>
          <pre
            className={cn(
              "whitespace-pre-wrap break-words rounded-md bg-black/30 p-3 font-mono text-xs",
              tone === "danger" ? "text-danger" : "text-content",
            )}
          >
            {value}
          </pre>
        </div>
      ))}
    </div>
  );
}

export function RunLogsModal({
  conversationId,
  bashCommandId,
  isOpen,
  onClose,
  run,
  automation,
}: RunLogsModalProps) {
  const { t } = useTranslation("openhands");
  const [activeTab, setActiveTab] = useState<LogTab>("stdout");

  const {
    data: outputs,
    isFetching,
    isResolvingConversation,
    sandboxIssue,
    conversationMissing,
    error,
  } = useBashCommandLogs({
    conversationId,
    bashCommandId,
    enabled: isOpen,
  });

  // Reset to the default tab whenever the modal opens for a different run.
  useEffect(() => {
    if (isOpen) setActiveTab("stdout");
  }, [isOpen, bashCommandId]);

  // Close on Escape.
  useEffect(() => {
    if (!isOpen) return undefined;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [isOpen, onClose]);

  const { stdout, stderr } = useMemo(() => {
    if (!outputs) return { stdout: "", stderr: "" };
    return {
      stdout: concatStream(outputs, "stdout"),
      stderr: concatStream(outputs, "stderr"),
    };
  }, [outputs]);

  if (!isOpen) return null;

  const loading = isResolvingConversation || (isFetching && !outputs);
  const hasBashCommand = !!bashCommandId;
  const noBashCommand = !hasBashCommand;
  const activeBody = activeTab === "stdout" ? stdout : stderr;
  const statusDetail = run?.status_detail ?? null;
  const rawStatusDetail =
    typeof statusDetail?.detail === "string" ? statusDetail.detail : null;
  const parsedStatusDetail = parseDetailText(rawStatusDetail);
  const statusDetailSummary = getStatusDetailSummary(
    statusDetail,
    parsedStatusDetail,
  );
  const statusDetailSections = hasBashCommand
    ? []
    : getDetailSections(parsedStatusDetail, statusDetailSummary);
  const statusDetailRows = getStatusDetailRows(statusDetail);
  const statusDetailExtraRows = getStatusDetailExtraRows(statusDetail);
  const errorDetail = run?.error_detail ?? null;
  const errorDetailDuplicatesStatusDetail =
    !!errorDetail &&
    !!rawStatusDetail &&
    errorDetail.trim() === rawStatusDetail.trim();
  const errorDetailSections = getDetailSections(
    parseDetailText(errorDetail),
    null,
  );
  const hasRunDetails = !!errorDetail || !!statusDetail;
  const titleKey = hasBashCommand
    ? I18nKey.AUTOMATIONS$DETAIL$LOGS_TITLE
    : I18nKey.AUTOMATIONS$DETAIL$RUN_DETAILS_TITLE;

  const tabBaseClass =
    "border-b-2 px-3 py-2 text-sm font-normal transition-colors focus:outline-none";
  const tabActiveClass = "border-[var(--oh-primary)] text-white";
  const tabInactiveClass = "border-transparent text-muted hover:text-content";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-label={t(titleKey)}
    >
      <div
        className="absolute inset-0 bg-black/60"
        onClick={onClose}
        onKeyDown={(e) => {
          if (e.key === "Escape") onClose();
        }}
        role="presentation"
      />
      <div className="relative flex max-h-[85vh] w-full max-w-3xl flex-col overflow-y-auto rounded-xl border border-[var(--oh-border)] bg-[var(--oh-surface)] p-6">
        <button
          type="button"
          onClick={onClose}
          className="absolute right-4 top-4 text-muted hover:text-foreground"
          aria-label={t(I18nKey.AUTOMATIONS$CANCEL)}
        >
          <XMarkIcon className="size-5" />
        </button>

        <h2 className={cn("pr-8", modalTitleLgMediumClassName)}>
          {t(titleKey)}
        </h2>

        {hasRunDetails && (
          <div
            data-testid="run-status-details"
            className="mt-4 space-y-3 rounded-lg border border-[var(--oh-border)] bg-black/20 p-4 text-sm"
          >
            {errorDetail && !errorDetailDuplicatesStatusDetail && (
              <div>
                <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-muted">
                  {t(I18nKey.AUTOMATIONS$DETAIL$RUN_ERROR_DETAIL_LABEL)}
                </h3>
                <DetailSections sections={errorDetailSections} />
              </div>
            )}

            {statusDetail && (
              <div className="space-y-3">
                <h3 className="text-xs font-medium uppercase tracking-wide text-muted">
                  {t(I18nKey.AUTOMATIONS$DETAIL$RUN_STATUS_DETAIL_LABEL)}
                </h3>
                {statusDetailSummary && (
                  <p className="rounded-md bg-black/20 px-3 py-2 text-content">
                    {statusDetailSummary}
                  </p>
                )}
                <DetailSections sections={statusDetailSections} />
                {statusDetailRows.length > 0 && (
                  <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-xs">
                    {statusDetailRows.map(({ key, label, value }) => (
                      <div key={key} className="contents">
                        <dt className="text-muted">{label}</dt>
                        <dd className="break-words text-content">{value}</dd>
                      </div>
                    ))}
                  </dl>
                )}
                {statusDetailExtraRows.length > 0 && (
                  <details className="text-xs">
                    <summary className="cursor-pointer text-muted hover:text-content">
                      {t(I18nKey.AUTOMATIONS$DETAIL$RUN_STATUS_DETAIL_METADATA)}
                    </summary>
                    <dl className="mt-2 grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 rounded-md bg-black/30 p-3">
                      {statusDetailExtraRows.map(({ key, label, value }) => (
                        <div key={key} className="contents">
                          <dt className="text-muted">{label}</dt>
                          <dd className="break-words text-content">{value}</dd>
                        </div>
                      ))}
                    </dl>
                  </details>
                )}
              </div>
            )}
          </div>
        )}

        <div
          role="tablist"
          aria-label={t(titleKey)}
          className="mt-4 flex gap-1 border-b border-[var(--oh-border)]"
        >
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "stdout"}
            aria-controls="run-logs-panel-stdout"
            id="run-logs-tab-stdout"
            tabIndex={activeTab === "stdout" ? 0 : -1}
            onClick={() => setActiveTab("stdout")}
            className={`${tabBaseClass} ${
              activeTab === "stdout" ? tabActiveClass : tabInactiveClass
            }`}
          >
            {t(I18nKey.AUTOMATIONS$DETAIL$LOGS_TAB_OUTPUT)}
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={activeTab === "stderr"}
            aria-controls="run-logs-panel-stderr"
            id="run-logs-tab-stderr"
            tabIndex={activeTab === "stderr" ? 0 : -1}
            onClick={() => setActiveTab("stderr")}
            className={`${tabBaseClass} ${
              activeTab === "stderr" ? tabActiveClass : tabInactiveClass
            }`}
          >
            {t(I18nKey.AUTOMATIONS$DETAIL$LOGS_TAB_ERROR)}
          </button>
        </div>

        <div
          role="tabpanel"
          id={`run-logs-panel-${activeTab}`}
          aria-labelledby={`run-logs-tab-${activeTab}`}
          className="mt-3 min-h-[12rem] flex-1 overflow-auto rounded-lg border border-[var(--oh-border)] bg-black/40 p-4 font-mono text-xs"
        >
          {noBashCommand && (
            <p className="text-muted italic">
              {t(I18nKey.AUTOMATIONS$DETAIL$LOGS_NO_COMMAND)}
            </p>
          )}

          {!noBashCommand && conversationMissing && (
            <p className="text-muted italic">
              {t(I18nKey.AUTOMATIONS$DETAIL$LOGS_CONVERSATION_MISSING)}
            </p>
          )}

          {!noBashCommand && !conversationMissing && sandboxIssue && (
            <p
              data-testid={`run-logs-sandbox-issue-${sandboxIssue}`}
              className="text-muted italic"
            >
              {t(SANDBOX_ISSUE_I18N[sandboxIssue])}
            </p>
          )}

          {!noBashCommand &&
            !conversationMissing &&
            !sandboxIssue &&
            loading && (
              <p className="text-muted italic">
                {t(I18nKey.AUTOMATIONS$DETAIL$LOGS_LOADING)}
              </p>
            )}

          {!noBashCommand &&
            !conversationMissing &&
            !sandboxIssue &&
            !loading &&
            error &&
            !outputs && (
              <p className="text-danger">
                {t(I18nKey.AUTOMATIONS$DETAIL$LOGS_ERROR)}: {String(error)}
              </p>
            )}

          {!loading && !sandboxIssue && outputs && (
            <pre
              data-testid={`run-logs-output-${activeTab}`}
              className={`whitespace-pre-wrap break-words ${
                activeTab === "stderr" ? "text-danger" : "text-content"
              }`}
            >
              {activeBody.length > 0 ? (
                activeBody
              ) : (
                <span className="text-muted italic">
                  {t(I18nKey.AUTOMATIONS$DETAIL$LOGS_EMPTY)}
                </span>
              )}
            </pre>
          )}
        </div>

        {run?.status === AutomationRunStatus.FAILED && (
          <div className="mt-4 flex justify-end">
            <DebugAutomationButton
              run={run}
              automation={automation}
              stderr={stderr}
            />
          </div>
        )}
      </div>
    </div>
  );
}
