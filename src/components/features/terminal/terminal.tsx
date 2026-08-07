import { useCallback } from "react";
import { useTerminal } from "#/hooks/use-terminal";
import "@xterm/xterm/css/xterm.css";
import { RUNTIME_INACTIVE_STATES } from "#/types/agent-state";
import { cn } from "#/utils/utils";
import { WaitingForRuntimeMessage } from "../chat/waiting-for-runtime-message";
import { useAgentState } from "#/hooks/use-agent-state";
import { useCommandStore } from "#/stores/command-store";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useBashCommandRunner } from "#/hooks/use-bash-command-runner";
import { DEFAULT_WORKING_DIR } from "#/api/agent-server-config";

/** Default timeout (seconds) for interactive shell commands. */
const INTERACTIVE_COMMAND_TIMEOUT_SEC = 120;

function Terminal() {
  const { curAgentState } = useAgentState();
  const { data: conversation } = useActiveConversation();
  const appendOutput = useCommandStore((state) => state.appendOutput);

  const isRuntimeInactive = RUNTIME_INACTIVE_STATES.includes(curAgentState);
  const conversationUrl = conversation?.conversation_url ?? null;
  const sessionApiKey = conversation?.session_api_key ?? null;
  const workingDir =
    conversation?.workspace?.working_dir?.trim() || DEFAULT_WORKING_DIR;

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

  const ref = useTerminal({
    onSubmitCommand: bashEnabled ? onSubmitCommand : undefined,
  });

  return (
    <div className="relative flex h-full min-h-0 flex-col">
      {isRuntimeInactive && <WaitingForRuntimeMessage className="pt-16" />}

      <div
        className={cn(
          "flex-1 min-h-0 p-4",
          isRuntimeInactive &&
            "pointer-events-none absolute inset-0 h-0 w-0 overflow-hidden p-0 opacity-0",
        )}
      >
        <div ref={ref} className="h-full w-full" />
      </div>
    </div>
  );
}

export default Terminal;
