import { FileClient } from "@openhands/typescript-client/clients";
import { RemoteWorkspace } from "@openhands/typescript-client/workspace/remote-workspace";
import { getAgentServerClientOptions } from "#/api/agent-server-client-options";
import { getActiveBackend } from "#/api/backend-registry/active-store";

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
 * whose CORS allowlist (`OH_ALLOW_CORS_ORIGINS`, set to the Canvas origin
 * in saas-deploy) permits direct browser calls authenticated with the
 * conversation's `X-Session-API-Key`. So both modes now use the same
 * direct typed-client path against the per-conversation runtime URL.
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

    if (active.kind === "cloud" && !conversationUrl) {
      throw new Error(
        "AgentServerRuntimeService.executeCommand requires a conversation URL on cloud backends",
      );
    }

    // `RemoteWorkspace.executeCommand` forwards `timeout` (seconds) to the
    // agent-server as the command timeout AND sets the HTTP request timeout to
    // `(timeout + 10) * 1000` ms — the same +10s buffer the old cloud
    // `callCloudProxy` branch used (`timeoutSeconds: timeout + 10`). We
    // deliberately do NOT pass `timeout` into `getAgentServerClientOptions`:
    // that option is the SDK HttpClient's default timeout in *milliseconds*,
    // so a seconds value (e.g. 30) would be read as 30 ms. The per-request
    // timeout the SDK sets on `executeCommand` overrides the client default
    // anyway, but passing a ms-mismatched default is a latent footgun if any
    // method on the client ever lacks a per-request timeout.
    const result = await new RemoteWorkspace(
      getAgentServerClientOptions({ conversationUrl, sessionApiKey }),
    ).executeCommand(command, cwd, timeout);
    // The SDK already coerces these (`exit_code ?? 0`, `stdout || ''`,
    // `stderr || ''`); keep a defensive fallback so a contract drift in the
    // client can never surface `undefined` to callers.
    return {
      exit_code: result.exit_code ?? -1,
      stdout: result.stdout ?? "",
      stderr: result.stderr ?? "",
    };
  }

  static async downloadFile(
    conversationUrl: string | null | undefined,
    sessionApiKey: string | null | undefined,
    path: string,
  ): Promise<ArrayBuffer> {
    const active = getActiveBackend().backend;

    if (active.kind === "cloud" && !conversationUrl) {
      throw new Error(
        "AgentServerRuntimeService.downloadFile requires a conversation URL on cloud backends",
      );
    }

    return new FileClient(
      getAgentServerClientOptions({ conversationUrl, sessionApiKey }),
    ).downloadFile(path);
  }
}

export default AgentServerRuntimeService;
