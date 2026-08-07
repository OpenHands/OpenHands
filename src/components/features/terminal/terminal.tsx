import React, { useCallback } from "react";
import { useTranslation } from "react-i18next";
import { Plus, X } from "lucide-react";
import { useTerminal } from "#/hooks/use-terminal";
import "@xterm/xterm/css/xterm.css";
import { RUNTIME_INACTIVE_STATES } from "#/types/agent-state";
import { cn } from "#/utils/utils";
import { WaitingForRuntimeMessage } from "../chat/waiting-for-runtime-message";
import { useAgentState } from "#/hooks/use-agent-state";
import {
  selectActiveConversationTerminals,
  useCommandStore,
} from "#/stores/command-store";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useBashCommandRunner } from "#/hooks/use-bash-command-runner";
import { DEFAULT_WORKING_DIR } from "#/api/agent-server-config";
import { I18nKey } from "#/i18n/declaration";

/** Default timeout (seconds) for interactive shell commands. */
const INTERACTIVE_COMMAND_TIMEOUT_SEC = 120;

function TerminalSession({
  bashEnabled,
  onSubmitCommand,
}: {
  bashEnabled: boolean;
  onSubmitCommand: (command: string) => Promise<void>;
}) {
  const ref = useTerminal({
    onSubmitCommand: bashEnabled ? onSubmitCommand : undefined,
  });

  return (
    <div ref={ref} className="h-full w-full" data-testid="terminal-xterm-host" />
  );
}

function Terminal() {
  const { t } = useTranslation("openhands");
  const { curAgentState } = useAgentState();
  const { data: conversation } = useActiveConversation();
  const appendOutput = useCommandStore((state) => state.appendOutput);
  const setActiveConversation = useCommandStore(
    (state) => state.setActiveConversation,
  );
  const terminals = useCommandStore(selectActiveConversationTerminals);
  const addTab = useCommandStore((state) => state.addTab);
  const closeTab = useCommandStore((state) => state.closeTab);
  const setActiveTab = useCommandStore((state) => state.setActiveTab);

  const isRuntimeInactive = RUNTIME_INACTIVE_STATES.includes(curAgentState);
  const conversationId = conversation?.id ?? null;
  const conversationUrl = conversation?.conversation_url ?? null;
  const sessionApiKey = conversation?.session_api_key ?? null;
  const workingDir =
    conversation?.workspace?.working_dir?.trim() || DEFAULT_WORKING_DIR;

  // Ensure this conversation owns the active terminal session even if the
  // route effect has not run yet (e.g. Terminal tab opened first).
  React.useEffect(() => {
    if (conversationId) {
      setActiveConversation(conversationId);
    }
  }, [conversationId, setActiveConversation]);

  const bashEnabled = !isRuntimeInactive && !!conversationUrl;
  const runCommand = useBashCommandRunner(
    conversationUrl,
    sessionApiKey,
    bashEnabled,
  );

  const onSubmitCommand = useCallback(
    async (command: string) => {
      const result = await runCommand(
        command,
        workingDir,
        INTERACTIVE_COMMAND_TIMEOUT_SEC,
      );
      const stdout = result.stdout?.trimEnd() ?? "";
      const stderr = result.stderr?.trimEnd() ?? "";
      const combined = [stdout, stderr].filter(Boolean).join("\n");

      if (combined) {
        appendOutput(combined);
      } else if (result.exit_code !== 0) {
        appendOutput(`Exit code: ${result.exit_code}`);
      }
    },
    [appendOutput, runCommand, workingDir],
  );

  const activeTabId = terminals?.activeTabId ?? "default";
  const sessionKey = `${conversationId ?? "none"}:${activeTabId}`;
  const canCloseTabs = (terminals?.tabs.length ?? 0) > 1;

  return (
    <div
      className="relative flex h-full min-h-0 flex-col"
      data-testid="terminal"
    >
      <div
        className="flex items-center gap-1 border-b border-[var(--oh-border)] px-2 py-1"
        data-testid="terminal-tabs"
      >
        <div className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto custom-scrollbar-always">
          {(terminals?.tabs ?? []).map((tab) => {
            const isActive = tab.id === terminals?.activeTabId;
            return (
              <div
                key={tab.id}
                className={cn(
                  "group flex shrink-0 items-center gap-1 rounded-md px-2 py-1 text-xs",
                  isActive
                    ? "bg-[var(--oh-interactive-hover)] text-white"
                    : "text-[var(--oh-muted)] hover:bg-[var(--oh-interactive-hover)]/60 hover:text-white",
                )}
              >
                <button
                  type="button"
                  data-testid={`terminal-tab-${tab.number}`}
                  className="max-w-[9rem] truncate"
                  onClick={() => setActiveTab(tab.id)}
                >
                  {t(I18nKey.TERMINAL$TAB_LABEL, { number: tab.number })}
                </button>
                {canCloseTabs && (
                  <button
                    type="button"
                    data-testid={`terminal-tab-close-${tab.number}`}
                    aria-label={t(I18nKey.TERMINAL$CLOSE_TAB)}
                    className="rounded p-0.5 opacity-70 hover:bg-black/30 hover:opacity-100"
                    onClick={(event) => {
                      event.stopPropagation();
                      closeTab(tab.id);
                    }}
                  >
                    <X className="size-3" aria-hidden />
                  </button>
                )}
              </div>
            );
          })}
        </div>
        <button
          type="button"
          data-testid="terminal-add-tab"
          aria-label={t(I18nKey.TERMINAL$NEW_TAB)}
          title={t(I18nKey.TERMINAL$NEW_TAB)}
          className="flex shrink-0 items-center justify-center rounded-md p-1 text-white hover:bg-[var(--oh-interactive-hover)]"
          onClick={() => addTab()}
        >
          <Plus className="size-3.5" aria-hidden />
        </button>
      </div>

      {isRuntimeInactive && <WaitingForRuntimeMessage className="pt-16" />}

      <div
        className={cn(
          "flex-1 min-h-0 p-4",
          isRuntimeInactive &&
            "pointer-events-none absolute inset-0 h-0 w-0 overflow-hidden p-0 opacity-0",
        )}
      >
        <TerminalSession
          key={sessionKey}
          bashEnabled={bashEnabled}
          onSubmitCommand={onSubmitCommand}
        />
      </div>
    </div>
  );
}

export default Terminal;
