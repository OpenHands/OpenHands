import { useCallback } from "react";
import { BashClient } from "@openhands/typescript-client/clients";
import type { CommandResult } from "#/api/runtime-service/agent-server-runtime-service";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";

export type BashCommandRunner = (
  command: string,
  cwd: string,
  timeout: number,
) => Promise<CommandResult>;

/** Run local Git probes through the SDK's conversation-scoped command API. */
export function useBashCommandRunner(
  conversationUrl: string | null | undefined,
  sessionApiKey: string | null | undefined,
  enabled: boolean,
  conversationId?: string,
): BashCommandRunner {
  return useCallback(
    async (command: string, cwd: string, timeout: number) => {
      if (!enabled) throw new Error("Bash command runner is disabled");
      const client = new BashClient(
        getAgentServerClientOptions({
          conversationUrl,
          sessionApiKey,
          conversationId,
        }),
      );
      try {
        const output = await client.executeCommand({ command, cwd, timeout });
        return {
          exit_code: output.exit_code ?? -1,
          stdout: output.stdout ?? "",
          stderr: output.stderr ?? "",
        };
      } finally {
        client.close();
      }
    },
    [enabled, conversationUrl, sessionApiKey, conversationId],
  );
}
