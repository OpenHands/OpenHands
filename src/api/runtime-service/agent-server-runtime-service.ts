import { FileClient } from "@openhands/typescript-client/clients";
import { RemoteWorkspace } from "@openhands/typescript-client/workspace/remote-workspace";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { getActiveBackend } from "#/api/backend-registry/active-store";
import { callCloudProxy } from "#/api/cloud/proxy";
import { buildHttpBaseUrl } from "#/utils/websocket-url";

export interface CommandResult {
  exit_code: number;
  stdout: string;
  stderr: string;
}

/**
 * Cloud-aware runtime operations for agent-server conversations.
 *
 * In **local** mode the runtime is reachable directly from the browser
 * (e.g. `127.0.0.1:18000`) so the SDK's typed clients work fine.
 * In **cloud** mode the runtime lives at `*.prod-runtime.all-hands.dev`,
 * which doesn't allow CORS from `localhost`, so all calls go through
 * `callCloudProxy` with the runtime URL as `hostOverride` and the
 * conversation's `session_api_key` as auth — server-side hop, no CORS.
 */
class AgentServerRuntimeService {
  static async executeCommand(
    conversationUrl: string | null | undefined,
    sessionApiKey: string | null | undefined,
    command: string,
    cwd?: string,
    timeout = 30,
  ): Promise<CommandResult> {
    const active = getActiveBackend().backend;

    if (active.kind === "cloud" && conversationUrl) {
      const output = await callCloudProxy<{
        exit_code?: number;
        stdout?: string;
        stderr?: string;
      }>({
        backend: active,
        method: "POST",
        hostOverride: buildHttpBaseUrl(conversationUrl),
        path: "/api/bash/execute_bash_command",
        body: {
          command,
          ...(cwd ? { cwd } : {}),
          timeout: Math.floor(timeout),
        },
        authMode: "session-api-key",
        sessionApiKey,
        timeoutSeconds: timeout + 10,
      });
      return {
        exit_code: output.exit_code ?? -1,
        stdout: output.stdout ?? "",
        stderr: output.stderr ?? "",
      };
    }

    const result = await new RemoteWorkspace(
      getAgentServerClientOptions({ conversationUrl, sessionApiKey }),
    ).executeCommand(command, cwd, timeout);
    return {
      exit_code: result.exit_code,
      stdout: result.stdout,
      stderr: result.stderr,
    };
  }

  static async downloadFile(
    conversationUrl: string | null | undefined,
    sessionApiKey: string | null | undefined,
    path: string,
  ): Promise<ArrayBuffer> {
    const active = getActiveBackend().backend;

    if (active.kind === "cloud" && conversationUrl) {
      const blob = await callCloudProxy<Blob>({
        backend: active,
        method: "GET",
        hostOverride: buildHttpBaseUrl(conversationUrl),
        path: `/api/file/download?path=${encodeURIComponent(path)}`,
        authMode: "session-api-key",
        sessionApiKey,
        responseType: "blob",
      });
      return blob.arrayBuffer();
    }

    return new FileClient(
      getAgentServerClientOptions({ conversationUrl, sessionApiKey }),
    ).downloadFile(path);
  }
}

export default AgentServerRuntimeService;
