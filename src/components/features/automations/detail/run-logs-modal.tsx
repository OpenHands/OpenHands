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
  const noBashCommand = !bashCommandId;
  const activeBody = activeTab === "stdout" ? stdout : stderr;

  const tabBaseClass =
    "border-b-2 px-3 py-2 text-sm font-normal transition-colors focus:outline-none";
  const tabActiveClass = "border-[var(--oh-primary)] text-white";
  const tabInactiveClass = "border-transparent text-muted hover:text-content";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-label={t(I18nKey.AUTOMATIONS$DETAIL$LOGS_TITLE)}
    >
      <div
        className="absolute inset-0 bg-black/60"
        onClick={onClose}
        onKeyDown={(e) => {
          if (e.key === "Escape") onClose();
        }}
        role="presentation"
      />
      <div className="relative flex max-h-[80vh] w-full max-w-3xl flex-col rounded-xl border border-[var(--oh-border)] bg-[var(--oh-surface)] p-6">
        <button
          type="button"
          onClick={onClose}
          className="absolute right-4 top-4 text-muted hover:text-foreground"
          aria-label={t(I18nKey.AUTOMATIONS$CANCEL)}
        >
          <XMarkIcon className="size-5" />
        </button>

        <h2 className={cn("pr-8", modalTitleLgMediumClassName)}>
          {t(I18nKey.AUTOMATIONS$DETAIL$LOGS_TITLE)}
        </h2>

        <div
          role="tablist"
          aria-label={t(I18nKey.AUTOMATIONS$DETAIL$LOGS_TITLE)}
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
